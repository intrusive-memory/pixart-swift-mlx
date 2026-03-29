#!/usr/bin/env python3
"""
Convert SDXL VAE weights from PyTorch (HuggingFace) to float16 MLX safetensors.

Source: stabilityai/sdxl-vae
Output: float16 MLX safetensors (~160 MB)

NO quantization -- Conv2d layers do not benefit from weight-only quantization.
All weights converted to float16.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import save_file as save_safetensors


# ---------------------------------------------------------------------------
# Conv2d transposition for MLX NHWC layout
# ---------------------------------------------------------------------------

def transpose_conv2d(weight: np.ndarray) -> np.ndarray:
    """
    Transpose Conv2d weight from PyTorch [O,I,kH,kW] to MLX [O,kH,kW,I].
    """
    if weight.ndim == 4:
        return weight.transpose(0, 2, 3, 1)
    return weight


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------

def load_vae_state_dict(model_id: str) -> tuple[dict[str, torch.Tensor], dict]:
    """Load SDXL VAE state dict and config from HuggingFace."""
    from diffusers import AutoencoderKL

    print(f"Loading SDXL VAE model from {model_id}...")
    vae = AutoencoderKL.from_pretrained(model_id, torch_dtype=torch.float32)
    state_dict = vae.state_dict()
    config = dict(vae.config)
    print(f"Loaded {len(state_dict)} keys from VAE state dict")
    return state_dict, config


def convert_weights(state_dict: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    """
    Convert VAE PyTorch state dict to MLX-compatible numpy arrays.

    All weights are kept as float16 (no quantization).
    Conv2d weights are transposed from [O,I,kH,kW] to [O,kH,kW,I] for MLX.
    """
    output = {}
    transposed_count = 0
    total_count = 0

    for key, tensor in sorted(state_dict.items()):
        arr = tensor.detach().cpu().float().numpy()
        total_count += 1

        # Transpose Conv2d weights for MLX NHWC layout
        if arr.ndim == 4:
            arr = transpose_conv2d(arr)
            transposed_count += 1
            print(f"  TRANSPOSE: {key} -> {arr.shape}")
        else:
            print(f"  FP16: {key} {arr.shape}")

        output[key] = arr.astype(np.float16)

    print(f"\nSummary:")
    print(f"  Total keys:  {total_count}")
    print(f"  Transposed:  {transposed_count} (Conv2d [O,I,kH,kW] -> [O,kH,kW,I])")
    print(f"  All stored as float16 (no quantization)")

    return output


def build_config(original_config: dict) -> dict:
    """Build config.json for the SDXL VAE."""
    # Keep original config and add our metadata
    config = dict(original_config)
    config["model_type"] = "sdxl-vae"
    config["quantization"] = None  # Explicitly mark as not quantized
    return config


def main():
    parser = argparse.ArgumentParser(
        description="Convert SDXL VAE weights to float16 MLX safetensors"
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
        default="stabilityai/sdxl-vae",
        help="HuggingFace model ID (default: stabilityai/sdxl-vae)",
    )
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    state_dict, original_config = load_vae_state_dict(args.model_id)

    # Convert weights
    converted = convert_weights(state_dict)

    # Save safetensors
    safetensors_path = out_dir / "model.safetensors"
    print(f"\nSaving {len(converted)} tensors to {safetensors_path}...")
    save_safetensors(converted, str(safetensors_path))
    size_mb = safetensors_path.stat().st_size / (1024 * 1024)
    print(f"Saved: {size_mb:.1f} MB")

    # Save config.json
    config = build_config(original_config)
    config_path = out_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2, default=str)
    print(f"Saved config to {config_path}")

    print(f"\nConversion complete. Output directory: {out_dir}")


if __name__ == "__main__":
    main()
