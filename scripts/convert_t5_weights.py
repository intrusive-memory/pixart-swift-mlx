#!/usr/bin/env python3
"""
Convert T5-XXL weights from PyTorch (HuggingFace) to MLX safetensors.

Source: google/t5-v1_1-xxl
Output: int4 quantized MLX safetensors (~1.2 GB)

Linear layers quantized to int4 (group_size=64).
Kept as float16 (NOT quantized): shared.weight (Embedding), relative_attention_bias (Embedding).
Saves config.json, tokenizer.json, tokenizer_config.json alongside weights.
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import save_file as save_safetensors
from transformers import T5EncoderModel, T5Tokenizer, AutoTokenizer


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

    # Pad N to be divisible by GROUP_SIZE if needed
    pad_n = 0
    if N % GROUP_SIZE != 0:
        pad_n = GROUP_SIZE - (N % GROUP_SIZE)
        weight = np.pad(weight, ((0, 0), (0, pad_n)), mode="constant")
        N = weight.shape[1]

    weight_fp32 = weight.astype(np.float32)
    num_groups = N // GROUP_SIZE
    grouped = weight_fp32.reshape(M, num_groups, GROUP_SIZE)

    g_min = grouped.min(axis=2)
    g_max = grouped.max(axis=2)

    scales = (g_max - g_min) / 15.0
    scales = np.where(scales == 0, 1.0, scales)
    biases = g_min

    quantized = np.clip(
        np.round((grouped - biases[:, :, np.newaxis]) / scales[:, :, np.newaxis]),
        0,
        15,
    ).astype(np.uint32)

    quantized_flat = quantized.reshape(M, -1)
    assert quantized_flat.shape[1] % 8 == 0
    packed_cols = quantized_flat.shape[1] // 8
    quantized_for_pack = quantized_flat.reshape(M, packed_cols, 8)

    packed = np.zeros((M, packed_cols), dtype=np.uint32)
    for bit_idx in range(8):
        packed |= quantized_for_pack[:, :, bit_idx] << (4 * bit_idx)

    return {
        "weight": packed,
        "scales": scales.astype(np.float16),
        "biases": biases.astype(np.float16),
    }


# ---------------------------------------------------------------------------
# Keys that should NOT be quantized
# ---------------------------------------------------------------------------

def should_skip_quantization(key: str, tensor: np.ndarray) -> bool:
    """
    Return True if this key should NOT be quantized (kept as float16).

    Skipped:
    - shared.weight (Embedding layer)
    - relative_attention_bias (Embedding layer)
    - All bias tensors
    - LayerNorm weights
    - 1D tensors (biases, norms)
    """
    # Embedding weights
    if "shared.weight" in key:
        return True
    if "relative_attention_bias" in key:
        return True
    # LayerNorm / layer_norm
    if "layer_norm" in key:
        return True
    if "final_layer_norm" in key:
        return True
    # Bias tensors
    if key.endswith(".bias"):
        return True
    # Non-2D tensors can't be quantized
    if tensor.ndim != 2:
        return True
    return False


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------

def load_t5_state_dict(model_id: str) -> tuple[dict[str, torch.Tensor], dict]:
    """Load T5-XXL encoder state dict and config from HuggingFace."""
    print(f"Loading T5 encoder model from {model_id}...")
    model = T5EncoderModel.from_pretrained(model_id, torch_dtype=torch.float32)
    state_dict = model.state_dict()
    config = model.config.to_dict()
    print(f"Loaded {len(state_dict)} keys from T5 encoder state dict")
    return state_dict, config


def convert_weights(state_dict: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    """
    Convert T5 PyTorch state dict to MLX-compatible numpy arrays.

    Applies int4 quantization to Linear weight tensors, keeps Embedding
    and LayerNorm weights as float16.
    """
    output = {}
    quantized_count = 0
    fp16_count = 0

    for key, tensor in sorted(state_dict.items()):
        arr = tensor.detach().cpu().float().numpy()

        if should_skip_quantization(key, arr):
            output[key] = arr.astype(np.float16)
            fp16_count += 1
            print(f"  FP16: {key} {arr.shape}")
        elif arr.ndim == 2:
            q = quantize_int4(arr)
            output[key] = q["weight"]
            output[key.replace(".weight", ".scales")] = q["scales"]
            output[key.replace(".weight", ".biases")] = q["biases"]
            quantized_count += 1
            print(
                f"  QUANTIZE: {key} [{arr.shape}] -> packed [{q['weight'].shape}]"
            )
        else:
            output[key] = arr.astype(np.float16)
            fp16_count += 1
            print(f"  FP16: {key} {arr.shape}")

    print(f"\nSummary:")
    print(f"  Quantized: {quantized_count} (int4, group_size={GROUP_SIZE})")
    print(f"  Float16:   {fp16_count}")

    return output


def save_tokenizer_files(model_id: str, out_dir: Path):
    """Save tokenizer.json and tokenizer_config.json."""
    print(f"Saving tokenizer files from {model_id}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.save_pretrained(str(out_dir))
        # Clean up unnecessary files, keep only what we need
        keep_files = {"tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"}
        for f in out_dir.iterdir():
            if f.name not in keep_files and f.suffix != ".safetensors" and f.name != "config.json":
                if f.is_file():
                    f.unlink()
                    print(f"  Removed extra tokenizer file: {f.name}")
        print(f"  Saved tokenizer files to {out_dir}")
    except Exception as e:
        print(f"  WARNING: Could not save tokenizer: {e}")
        print(f"  You may need to copy tokenizer files manually.")


def main():
    parser = argparse.ArgumentParser(
        description="Convert T5-XXL weights to int4 MLX safetensors"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for MLX safetensors, config.json, and tokenizer files",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="google/t5-v1_1-xxl",
        help="HuggingFace model ID (default: google/t5-v1_1-xxl)",
    )
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    state_dict, config = load_t5_state_dict(args.model_id)

    # Convert weights
    converted = convert_weights(state_dict)

    # Save safetensors (may need to split for large models)
    # T5-XXL int4 is ~1.2 GB, split into 2 shards at ~700 MB each
    keys = sorted(converted.keys())
    total_bytes = sum(v.nbytes for v in converted.values())
    shard_limit = 700 * 1024 * 1024  # 700 MB per shard

    if total_bytes > shard_limit:
        print(f"\nTotal size: {total_bytes / (1024**2):.1f} MB, splitting into shards...")
        shard_idx = 0
        shard_data = {}
        shard_bytes = 0
        shard_files = []

        for key in keys:
            tensor_bytes = converted[key].nbytes
            if shard_bytes + tensor_bytes > shard_limit and shard_data:
                shard_name = f"model-{shard_idx:05d}-of-TOTAL.safetensors"
                shard_path = out_dir / shard_name
                save_safetensors(shard_data, str(shard_path))
                shard_files.append(shard_name)
                size_mb = shard_path.stat().st_size / (1024 * 1024)
                print(f"  Saved shard {shard_idx}: {shard_name} ({size_mb:.1f} MB)")
                shard_idx += 1
                shard_data = {}
                shard_bytes = 0

            shard_data[key] = converted[key]
            shard_bytes += tensor_bytes

        # Save remaining
        if shard_data:
            shard_name = f"model-{shard_idx:05d}-of-TOTAL.safetensors"
            shard_path = out_dir / shard_name
            save_safetensors(shard_data, str(shard_path))
            shard_files.append(shard_name)
            size_mb = shard_path.stat().st_size / (1024 * 1024)
            print(f"  Saved shard {shard_idx}: {shard_name} ({size_mb:.1f} MB)")

        # Rename shards with correct total
        total_shards = len(shard_files)
        weight_map = {}
        for idx, old_name in enumerate(shard_files):
            new_name = f"model-{idx:05d}-of-{total_shards:05d}.safetensors"
            old_path = out_dir / old_name
            new_path = out_dir / new_name
            if old_name != new_name:
                old_path.rename(new_path)
            # Build weight map for index
            # Re-read the shard to get key list
            from safetensors import safe_open
            with safe_open(str(new_path), framework="numpy") as f:
                for key in f.keys():
                    weight_map[key] = new_name

        # Save model.safetensors.index.json
        index = {
            "metadata": {"total_size": total_bytes},
            "weight_map": weight_map,
        }
        index_path = out_dir / "model.safetensors.index.json"
        with open(index_path, "w") as f:
            json.dump(index, f, indent=2)
        print(f"  Saved index to {index_path}")
    else:
        safetensors_path = out_dir / "model.safetensors"
        print(f"\nSaving {len(converted)} tensors to {safetensors_path}...")
        save_safetensors(converted, str(safetensors_path))
        size_mb = safetensors_path.stat().st_size / (1024 * 1024)
        print(f"Saved: {size_mb:.1f} MB")

    # Save config.json
    config["quantization"] = {
        "group_size": GROUP_SIZE,
        "bits": 4,
    }
    config_path = out_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved config to {config_path}")

    # Save tokenizer files
    save_tokenizer_files(args.model_id, out_dir)

    print(f"\nConversion complete. Output directory: {out_dir}")


if __name__ == "__main__":
    main()
