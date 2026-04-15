#!/usr/bin/env python3
"""
Dequantize PixArt-Sigma DiT int4 safetensors → fp16 safetensors.

Source: /tmp/vinetas-test-models/pixart-sigma-xl-dit-int4/model.safetensors
Output: /tmp/vinetas-test-models/pixart-sigma-xl-dit-fp16/model.safetensors

The int4-quantized safetensors stores Linear weight tensors as triplets:
  - <key>.weight  — uint32 packed, shape [outDim, inDim/8]
  - <key>.scales  — float16, shape [outDim, numGroups]  (numGroups = inDim/64)
  - <key>.biases  — float16, shape [outDim, numGroups]  (zero-point = min value)

This script dequantizes every such triplet back to float16 and drops the
.scales / .biases sidecar keys. Non-quantized tensors (scaleShiftTable, LayerNorm
weights, Conv2d patchEmbed, biases) are copied through unchanged as float16.

Usage:
    python3 dequantize_dit_to_fp16.py [--input DIR] [--output DIR]

Requires:
    pip install mlx safetensors numpy
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import mlx.core as mx
import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file as save_safetensors


# ---------------------------------------------------------------------------
# Dequantization
# ---------------------------------------------------------------------------

GROUP_SIZE = 64
BITS = 4


def dequantize_tensor(
    weight: mx.array,
    scales: mx.array,
    biases: mx.array,
) -> np.ndarray:
    """
    Dequantize a packed int4 weight tensor using MLX's built-in dequantize.

    mx.dequantize(w, scales, biases, group_size, bits) reconstructs:
        float_weight[i, j] = packed_int4_value[i, j] * scales[i, g] + biases[i, g]
    where g = j // group_size.

    Returns a float16 numpy array of shape [outDim, inDim].
    """
    float_weight = mx.dequantize(weight, scales, biases, GROUP_SIZE, BITS)
    mx.eval(float_weight)
    return np.array(float_weight, dtype=np.float16)


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------

def dequantize_safetensors(input_path: Path, output_path: Path) -> None:
    """
    Load int4 safetensors, dequantize all quantized Linear weights, write fp16.
    """
    print(f"Loading {input_path} ...")

    # First pass: collect all tensor names and identify quantization sidecars
    with safe_open(str(input_path), framework="numpy") as f:
        all_keys = list(f.keys())

    scales_keys: set[str] = {k for k in all_keys if k.endswith(".scales")}
    biases_keys: set[str] = {k for k in all_keys if k.endswith(".biases")}

    # Base paths that have both .scales and .biases (quantized Linear weights)
    quantized_bases: set[str] = set()
    for k in scales_keys:
        base = k[: -len(".scales")]
        if f"{base}.biases" in biases_keys:
            quantized_bases.add(base)

    print(f"  Total tensors : {len(all_keys)}")
    print(f"  Quantized Linear weight triplets : {len(quantized_bases)}")
    print(f"  Sidecar keys to drop : {len(scales_keys) + len(biases_keys)}")

    # Second pass: load, dequantize, and collect output tensors
    output_tensors: dict[str, np.ndarray] = {}
    dequantized_count = 0
    passthrough_count = 0

    with safe_open(str(input_path), framework="numpy") as f:
        for key in all_keys:
            # Drop .scales and .biases sidecar keys — they have no equivalent in
            # the fp16 model (plain Linear has only .weight + .bias).
            if key.endswith(".scales") or key.endswith(".biases"):
                continue

            if key.endswith(".weight"):
                base = key[: -len(".weight")]
                if base in quantized_bases:
                    # Load the triplet for dequantization
                    w_np = f.get_tensor(key)        # uint32 packed
                    s_np = f.get_tensor(f"{base}.scales")   # float16 scales
                    b_np = f.get_tensor(f"{base}.biases")   # float16 biases

                    w_mlx = mx.array(w_np)
                    s_mlx = mx.array(s_np.astype(np.float32)).astype(mx.float16)
                    b_mlx = mx.array(b_np.astype(np.float32)).astype(mx.float16)

                    fp16_weight = dequantize_tensor(w_mlx, s_mlx, b_mlx)
                    output_tensors[key] = fp16_weight
                    dequantized_count += 1
                    print(
                        f"  DEQUANTIZE: {key} "
                        f"{w_np.shape} uint32 -> {fp16_weight.shape} float16"
                    )
                    continue

            # Pass through everything else unchanged (scaleShiftTable, LayerNorm
            # weights, Conv2d patchEmbed, bias tensors, etc.)
            tensor = f.get_tensor(key)
            # Ensure float tensors are stored as float16
            if tensor.dtype in (np.float32, np.float64):
                tensor = tensor.astype(np.float16)
            output_tensors[key] = tensor
            passthrough_count += 1

    print(f"\nConversion summary:")
    print(f"  Dequantized : {dequantized_count}")
    print(f"  Passthrough : {passthrough_count}")
    print(f"  Output keys : {len(output_tensors)}")

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving {len(output_tensors)} tensors to {output_path} ...")
    save_safetensors(output_tensors, str(output_path))
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Saved: {size_mb:.1f} MB")


def copy_config(input_dir: Path, output_dir: Path) -> None:
    """Copy config.json from input_dir to output_dir if present."""
    config_src = input_dir / "config.json"
    if not config_src.exists():
        print("No config.json found — skipping")
        return

    config_dst = output_dir / "config.json"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(config_src) as f:
        config = json.load(f)

    # Update config to reflect fp16 (remove quantization field if present)
    config.pop("quantization", None)

    with open(config_dst, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Copied config.json (quantization field removed) -> {config_dst}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dequantize PixArt-Sigma DiT int4 safetensors to fp16"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="/tmp/vinetas-test-models/pixart-sigma-xl-dit-int4",
        help="Directory containing int4 model.safetensors and config.json",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/tmp/vinetas-test-models/pixart-sigma-xl-dit-fp16",
        help="Output directory for fp16 model.safetensors and config.json",
    )
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    input_safetensors = input_dir / "model.safetensors"
    if not input_safetensors.exists():
        print(f"ERROR: input not found: {input_safetensors}")
        raise SystemExit(1)

    output_safetensors = output_dir / "model.safetensors"

    dequantize_safetensors(input_safetensors, output_safetensors)
    copy_config(input_dir, output_dir)

    print(f"\nDequantization complete. Output: {output_dir}")
    print(
        "Next step: run `make test-fixtures-fp16` to test the fp16 DiT against int4."
    )


if __name__ == "__main__":
    main()
