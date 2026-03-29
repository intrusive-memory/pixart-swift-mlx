#!/usr/bin/env python3
"""
Convert PixArt-Sigma DiT weights from PyTorch (HuggingFace diffusers) to MLX safetensors.

Source: PixArt-alpha/PixArt-Sigma-XL-2-1024-MS (diffusers format)
Output: int4 quantized MLX safetensors (~300 MB)

Key remapping matches WeightMapping.swift exactly.
Conv2d weight transposition: [O,I,kH,kW] -> [O,kH,kW,I] for patchEmbed.weight.
Linear layers quantized to int4 (group_size=64).
Kept as float16 (NOT quantized): scale_shift_table, LayerNorm weights, Conv2d, Embedding.
Discarded: pos_embed, y_embedder.y_embedding.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import save_file as save_safetensors


# ---------------------------------------------------------------------------
# Key mapping: HuggingFace diffusers keys -> MLX module paths
# This MUST match WeightMapping.swift exactly.
# ---------------------------------------------------------------------------

NUM_BLOCKS = 28


def build_key_mapping() -> dict[str, str]:
    """Build the complete key mapping table matching WeightMapping.swift."""
    table = {}

    # Global keys (~22 pairs)

    # Patch embedding Conv2d: pos_embed.proj -> patchEmbed
    # Weight requires Conv2d transposition: [O,I,kH,kW] -> [O,kH,kW,I]
    table["pos_embed.proj.weight"] = "patchEmbed.weight"
    table["pos_embed.proj.bias"] = "patchEmbed.bias"

    # Caption projection: Linear(4096,1152) -> GELU(tanh) -> Linear(1152,1152)
    table["caption_projection.linear_1.weight"] = "captionProjection.linear1.weight"
    table["caption_projection.linear_1.bias"] = "captionProjection.linear1.bias"
    table["caption_projection.linear_2.weight"] = "captionProjection.linear2.weight"
    table["caption_projection.linear_2.bias"] = "captionProjection.linear2.bias"

    # Timestep embedder MLP: Linear(256,1152) -> SiLU -> Linear(1152,1152)
    table["adaln_single.emb.timestep_embedder.linear_1.weight"] = (
        "timestepEmbedder.linear1.weight"
    )
    table["adaln_single.emb.timestep_embedder.linear_1.bias"] = (
        "timestepEmbedder.linear1.bias"
    )
    table["adaln_single.emb.timestep_embedder.linear_2.weight"] = (
        "timestepEmbedder.linear2.weight"
    )
    table["adaln_single.emb.timestep_embedder.linear_2.bias"] = (
        "timestepEmbedder.linear2.bias"
    )

    # Resolution embedder
    table["adaln_single.emb.resolution_embedder.linear_1.weight"] = (
        "sizeEmbedder.embedder.linear1.weight"
    )
    table["adaln_single.emb.resolution_embedder.linear_1.bias"] = (
        "sizeEmbedder.embedder.linear1.bias"
    )
    table["adaln_single.emb.resolution_embedder.linear_2.weight"] = (
        "sizeEmbedder.embedder.linear2.weight"
    )
    table["adaln_single.emb.resolution_embedder.linear_2.bias"] = (
        "sizeEmbedder.embedder.linear2.bias"
    )

    # Aspect ratio embedder
    table["adaln_single.emb.aspect_ratio_embedder.linear_1.weight"] = (
        "arEmbedder.embedder.linear1.weight"
    )
    table["adaln_single.emb.aspect_ratio_embedder.linear_1.bias"] = (
        "arEmbedder.embedder.linear1.bias"
    )
    table["adaln_single.emb.aspect_ratio_embedder.linear_2.weight"] = (
        "arEmbedder.embedder.linear2.weight"
    )
    table["adaln_single.emb.aspect_ratio_embedder.linear_2.bias"] = (
        "arEmbedder.embedder.linear2.bias"
    )

    # t_block: SiLU -> Linear(1152, 6*1152)
    table["adaln_single.linear.weight"] = "t_block_linear.weight"
    table["adaln_single.linear.bias"] = "t_block_linear.bias"

    # Final layer projection: Linear(1152, patchSize^2 * outChannels)
    table["proj_out.weight"] = "finalLayer.linear.weight"
    table["proj_out.bias"] = "finalLayer.linear.bias"

    # Final layer AdaLN scale_shift_table: [2, 1152]
    table["scale_shift_table"] = "finalLayer.scaleShiftTable"

    # Per-block keys: 28 blocks x ~8 groups
    for i in range(NUM_BLOCKS):
        hf = f"transformer_blocks.{i}"
        mlx = f"blocks.{i}"

        # scale_shift_table: [6, 1152]
        table[f"{hf}.scale_shift_table"] = f"{mlx}.scaleShiftTable"

        # Self-attention Q/K/V/out projections
        table[f"{hf}.attn1.to_q.weight"] = f"{mlx}.attn.to_q.weight"
        table[f"{hf}.attn1.to_q.bias"] = f"{mlx}.attn.to_q.bias"
        table[f"{hf}.attn1.to_k.weight"] = f"{mlx}.attn.to_k.weight"
        table[f"{hf}.attn1.to_k.bias"] = f"{mlx}.attn.to_k.bias"
        table[f"{hf}.attn1.to_v.weight"] = f"{mlx}.attn.to_v.weight"
        table[f"{hf}.attn1.to_v.bias"] = f"{mlx}.attn.to_v.bias"
        table[f"{hf}.attn1.to_out.0.weight"] = f"{mlx}.attn.to_out.weight"
        table[f"{hf}.attn1.to_out.0.bias"] = f"{mlx}.attn.to_out.bias"

        # Self-attention QK norms: LayerNorm
        table[f"{hf}.attn1.q_norm.weight"] = f"{mlx}.attn.q_norm.weight"
        table[f"{hf}.attn1.q_norm.bias"] = f"{mlx}.attn.q_norm.bias"
        table[f"{hf}.attn1.k_norm.weight"] = f"{mlx}.attn.k_norm.weight"
        table[f"{hf}.attn1.k_norm.bias"] = f"{mlx}.attn.k_norm.bias"

        # Cross-attention Q/K/V/out projections
        table[f"{hf}.attn2.to_q.weight"] = f"{mlx}.cross_attn.to_q.weight"
        table[f"{hf}.attn2.to_q.bias"] = f"{mlx}.cross_attn.to_q.bias"
        table[f"{hf}.attn2.to_k.weight"] = f"{mlx}.cross_attn.to_k.weight"
        table[f"{hf}.attn2.to_k.bias"] = f"{mlx}.cross_attn.to_k.bias"
        table[f"{hf}.attn2.to_v.weight"] = f"{mlx}.cross_attn.to_v.weight"
        table[f"{hf}.attn2.to_v.bias"] = f"{mlx}.cross_attn.to_v.bias"
        table[f"{hf}.attn2.to_out.0.weight"] = f"{mlx}.cross_attn.to_out.weight"
        table[f"{hf}.attn2.to_out.0.bias"] = f"{mlx}.cross_attn.to_out.bias"

        # FFN: GEGLU
        table[f"{hf}.ff.net.0.proj.weight"] = f"{mlx}.mlp.fc1.weight"
        table[f"{hf}.ff.net.0.proj.bias"] = f"{mlx}.mlp.fc1.bias"
        table[f"{hf}.ff.net.2.weight"] = f"{mlx}.mlp.fc2.weight"
        table[f"{hf}.ff.net.2.bias"] = f"{mlx}.mlp.fc2.bias"

    return table


# Keys to explicitly discard (silently dropped)
DISCARDED_KEYS = {"pos_embed", "y_embedder.y_embedding"}


# ---------------------------------------------------------------------------
# Quantization: int4, group_size=64
# ---------------------------------------------------------------------------

GROUP_SIZE = 64


def quantize_int4(weight: np.ndarray) -> dict[str, np.ndarray]:
    """
    Quantize a 2D weight matrix [M, N] to int4 with group_size=64.

    Returns dict with:
      - "weight": [M, N/8] uint32 (packed 4-bit values)
      - "scales": [M, N/group_size] float16
      - "biases": [M, N/group_size] float16 (quantization zero-points)
    """
    assert weight.ndim == 2, f"Expected 2D weight, got shape {weight.shape}"
    M, N = weight.shape
    assert N % GROUP_SIZE == 0, (
        f"Weight dim N={N} must be divisible by group_size={GROUP_SIZE}"
    )

    # Reshape into groups: [M, N/group_size, group_size]
    weight_fp32 = weight.astype(np.float32)
    num_groups = N // GROUP_SIZE
    grouped = weight_fp32.reshape(M, num_groups, GROUP_SIZE)

    # Compute per-group min/max
    g_min = grouped.min(axis=2)  # [M, num_groups]
    g_max = grouped.max(axis=2)  # [M, num_groups]

    # Compute scales and zero-points (biases)
    # Map [min, max] -> [0, 15] for 4-bit unsigned
    scales = (g_max - g_min) / 15.0
    # Avoid division by zero
    scales = np.where(scales == 0, 1.0, scales)
    biases = g_min  # zero-point = min value

    # Quantize to 4-bit unsigned integers [0, 15]
    quantized = np.clip(
        np.round((grouped - biases[:, :, np.newaxis]) / scales[:, :, np.newaxis]),
        0,
        15,
    ).astype(np.uint32)

    # Pack 8 x 4-bit values into uint32
    # quantized shape: [M, num_groups, GROUP_SIZE]
    # Reshape to [M, N/8, 8] for packing
    quantized_flat = quantized.reshape(M, -1)  # [M, N]
    assert quantized_flat.shape[1] % 8 == 0
    packed_cols = quantized_flat.shape[1] // 8
    quantized_for_pack = quantized_flat.reshape(M, packed_cols, 8)

    packed = np.zeros((M, packed_cols), dtype=np.uint32)
    for bit_idx in range(8):
        packed |= quantized_for_pack[:, :, bit_idx] << (4 * bit_idx)

    return {
        "weight": packed,  # [M, N/8] uint32
        "scales": scales.astype(np.float16),  # [M, N/group_size] float16
        "biases": biases.astype(np.float16),  # [M, N/group_size] float16
    }


# ---------------------------------------------------------------------------
# Keys that should NOT be quantized (kept as float16)
# ---------------------------------------------------------------------------

def should_skip_quantization(mlx_key: str) -> bool:
    """
    Return True if this key should NOT be quantized (kept as float16).

    Skipped:
    - scaleShiftTable (all instances)
    - LayerNorm weights (q_norm, k_norm)
    - Conv2d patch embed weights (patchEmbed)
    - Embedding weights
    - All bias tensors (bias is singular, not "biases")
    """
    # scale_shift_table entries
    if "scaleShiftTable" in mlx_key:
        return True
    # LayerNorm weights (q_norm, k_norm)
    if "q_norm" in mlx_key or "k_norm" in mlx_key:
        return True
    # Conv2d patch embed
    if "patchEmbed" in mlx_key:
        return True
    # Bias tensors (the model bias, not quantization biases)
    if mlx_key.endswith(".bias"):
        return True
    return False


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------

def load_pytorch_state_dict(model_id: str) -> dict[str, torch.Tensor]:
    """Load PixArt-Sigma state dict from HuggingFace."""
    from diffusers import PixArtSigmaPipeline

    print(f"Loading PixArt-Sigma model from {model_id}...")
    pipe = PixArtSigmaPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
    )
    state_dict = pipe.transformer.state_dict()
    print(f"Loaded {len(state_dict)} keys from transformer state dict")
    return state_dict


def convert_weights(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, np.ndarray]:
    """
    Convert PyTorch state dict to MLX-compatible numpy arrays.

    Applies:
    1. Key remapping (HuggingFace diffusers -> MLX module paths)
    2. Conv2d transposition for patchEmbed.weight
    3. int4 quantization for Linear weight tensors
    4. float16 for everything else
    """
    key_map = build_key_mapping()
    output = {}
    unmapped_keys = []
    mapped_count = 0
    discarded_count = 0
    quantized_count = 0
    fp16_count = 0

    for hf_key, tensor in state_dict.items():
        # Check if key should be discarded
        if hf_key in DISCARDED_KEYS:
            print(f"  DISCARD: {hf_key}")
            discarded_count += 1
            continue

        # Look up MLX key
        mlx_key = key_map.get(hf_key)
        if mlx_key is None:
            unmapped_keys.append(hf_key)
            continue

        mapped_count += 1
        arr = tensor.detach().cpu().float().numpy()

        # Conv2d transposition for patch embed: [O,I,kH,kW] -> [O,kH,kW,I]
        if mlx_key == "patchEmbed.weight" and arr.ndim == 4:
            arr = arr.transpose(0, 2, 3, 1)
            print(f"  TRANSPOSE: {hf_key} -> {mlx_key} {arr.shape}")

        # Decide: quantize or keep as float16
        if not should_skip_quantization(mlx_key) and arr.ndim == 2:
            # Quantize Linear weight to int4
            q = quantize_int4(arr)
            output[mlx_key] = q["weight"]
            output[mlx_key.replace(".weight", ".scales")] = q["scales"]
            output[mlx_key.replace(".weight", ".biases")] = q["biases"]
            quantized_count += 1
            print(
                f"  QUANTIZE: {hf_key} -> {mlx_key} "
                f"[{arr.shape}] -> packed [{q['weight'].shape}]"
            )
        else:
            # Keep as float16
            output[mlx_key] = arr.astype(np.float16)
            fp16_count += 1
            print(f"  FP16: {hf_key} -> {mlx_key} {arr.shape}")

    print(f"\nSummary:")
    print(f"  Mapped:     {mapped_count}")
    print(f"  Discarded:  {discarded_count}")
    print(f"  Quantized:  {quantized_count} (int4, group_size={GROUP_SIZE})")
    print(f"  Float16:    {fp16_count}")

    if unmapped_keys:
        print(f"\n  WARNING: {len(unmapped_keys)} unmapped keys:")
        for k in unmapped_keys:
            print(f"    - {k}")
        print(
            "\n  These keys exist in the PyTorch model but have no mapping in "
            "WeightMapping.swift. This may indicate a diffusers version mismatch."
        )
    else:
        print(f"  Unmapped:   0 (all keys accounted for)")

    return output


def build_config() -> dict:
    """Build config.json for the PixArt-Sigma DiT model."""
    return {
        "model_type": "pixart-sigma-dit",
        "hidden_size": 1152,
        "num_attention_heads": 16,
        "num_layers": 28,
        "patch_size": 2,
        "in_channels": 4,
        "out_channels": 8,
        "caption_channels": 4096,
        "sample_size": 128,
        "quantization": {
            "group_size": GROUP_SIZE,
            "bits": 4,
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Convert PixArt-Sigma DiT weights to int4 MLX safetensors"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for MLX safetensors and config.json",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
        help="HuggingFace model ID (default: PixArt-alpha/PixArt-Sigma-XL-2-1024-MS)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print key mapping without downloading model",
    )
    args = parser.parse_args()

    if args.dry_run:
        key_map = build_key_mapping()
        print(f"Key mapping table ({len(key_map)} entries):\n")
        for hf_key in sorted(key_map.keys()):
            print(f"  {hf_key}\n    -> {key_map[hf_key]}")
        print(f"\nDiscarded keys: {DISCARDED_KEYS}")
        return

    # Create output directory
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load and convert
    state_dict = load_pytorch_state_dict(args.model_id)
    converted = convert_weights(state_dict)

    # Save safetensors
    safetensors_path = out_dir / "model.safetensors"
    print(f"\nSaving {len(converted)} tensors to {safetensors_path}...")
    save_safetensors(converted, str(safetensors_path))
    size_mb = safetensors_path.stat().st_size / (1024 * 1024)
    print(f"Saved: {size_mb:.1f} MB")

    # Save config.json
    config_path = out_dir / "config.json"
    config = build_config()
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved config to {config_path}")

    print(f"\nConversion complete. Output directory: {out_dir}")


if __name__ == "__main__":
    main()
