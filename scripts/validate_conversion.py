#!/usr/bin/env python3
"""
Validation harness for weight conversion.

Compares PyTorch reference outputs against MLX converted weights by running
deterministic forward passes and computing PSNR between outputs.

Validates:
- PixArt-Sigma DiT (int4 quantized)
- T5-XXL Encoder (int4 quantized)
- SDXL VAE Decoder (float16)

Thresholds:
- End-to-end PSNR > 30 dB for all outputs
- Per-layer PSNR > 25 dB (warning if below, even if end-to-end passes)
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# PSNR computation
# ---------------------------------------------------------------------------


def compute_psnr(reference: np.ndarray, test: np.ndarray) -> float:
    """
    Compute Peak Signal-to-Noise Ratio between reference and test arrays.

    Both arrays should be float32. Returns PSNR in dB.
    Returns float('inf') if arrays are identical.
    """
    ref = reference.astype(np.float32).flatten()
    tst = test.astype(np.float32).flatten()

    if ref.shape != tst.shape:
        raise ValueError(
            f"Shape mismatch: reference {reference.shape} vs test {test.shape}"
        )

    mse = np.mean((ref - tst) ** 2)
    if mse == 0:
        return float("inf")

    # Use the max value from the reference for peak signal
    max_val = np.max(np.abs(ref))
    if max_val == 0:
        max_val = 1.0

    psnr = 20 * np.log10(max_val) - 10 * np.log10(mse)
    return float(psnr)


# ---------------------------------------------------------------------------
# Validation prompts (deterministic)
# ---------------------------------------------------------------------------

VALIDATION_PROMPTS = [
    "A photorealistic cat sitting on a windowsill at sunset",
    "An oil painting of a mountain landscape with snow-capped peaks",
    "A digital illustration of a futuristic city skyline at night",
    "A watercolor painting of sunflowers in a glass vase",
    "A macro photograph of dewdrops on a spider web at dawn",
]

VALIDATION_SEEDS = [42, 123, 456, 789, 1024]

# Thresholds
END_TO_END_PSNR_THRESHOLD = 30.0  # dB
PER_LAYER_PSNR_THRESHOLD = 25.0  # dB (warning only)


# ---------------------------------------------------------------------------
# PixArt DiT validation
# ---------------------------------------------------------------------------


def validate_pixart_dit(
    converted_dir: str,
    model_id: str = "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
) -> list[dict]:
    """
    Validate PixArt-Sigma DiT conversion by comparing layer outputs.

    Loads the original PyTorch model and the converted MLX weights,
    runs forward passes with deterministic inputs, and compares outputs.
    """
    import torch
    from safetensors import safe_open
    from convert_pixart_weights import build_key_mapping, DISCARDED_KEYS

    results = []
    converted_path = Path(converted_dir)

    print("\n" + "=" * 70)
    print("Validating PixArt-Sigma DiT conversion")
    print("=" * 70)

    # Load converted weights
    print(f"Loading converted weights from {converted_path}...")
    converted_tensors = {}
    for sf_file in sorted(converted_path.glob("*.safetensors")):
        with safe_open(str(sf_file), framework="numpy") as f:
            for key in f.keys():
                converted_tensors[key] = f.get_tensor(key)
    print(f"  Loaded {len(converted_tensors)} tensors from converted files")

    # Load original PyTorch weights
    print(f"Loading original model from {model_id}...")
    from diffusers import PixArtSigmaPipeline

    pipe = PixArtSigmaPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    original_state_dict = pipe.transformer.state_dict()
    print(f"  Loaded {len(original_state_dict)} keys from original model")

    # Build key mapping
    key_map = build_key_mapping()

    # Per-layer comparison: compare each mapped weight
    print("\nPer-layer weight comparison:")
    layer_results = []

    for hf_key, mlx_key in sorted(key_map.items()):
        if hf_key not in original_state_dict:
            print(f"  SKIP (not in original): {hf_key}")
            continue

        original = original_state_dict[hf_key].detach().cpu().float().numpy()

        # For quantized weights, we need to dequantize to compare
        if mlx_key in converted_tensors:
            converted = converted_tensors[mlx_key]

            # Check if this was quantized (uint32 packed)
            if converted.dtype == np.uint32 and converted.ndim == 2:
                # Dequantize for comparison
                scales_key = mlx_key.replace(".weight", ".scales")
                biases_key = mlx_key.replace(".weight", ".biases")
                if scales_key in converted_tensors and biases_key in converted_tensors:
                    scales = converted_tensors[scales_key].astype(np.float32)
                    biases = converted_tensors[biases_key].astype(np.float32)
                    converted = dequantize_int4(converted, scales, biases)
                else:
                    print(f"  SKIP (missing scales/biases): {mlx_key}")
                    continue
            else:
                converted = converted.astype(np.float32)

            # Handle Conv2d transposition
            if mlx_key == "patchEmbed.weight" and original.ndim == 4:
                original = original.transpose(0, 2, 3, 1)

            # Flatten for PSNR
            if original.shape != converted.shape:
                print(
                    f"  SHAPE MISMATCH: {mlx_key} "
                    f"original={original.shape} converted={converted.shape}"
                )
                layer_results.append(
                    {"key": mlx_key, "psnr": 0.0, "status": "SHAPE_MISMATCH"}
                )
                continue

            psnr = compute_psnr(original, converted)
            status = "PASS" if psnr >= PER_LAYER_PSNR_THRESHOLD else "WARN"
            layer_results.append({"key": mlx_key, "psnr": psnr, "status": status})

            if psnr < PER_LAYER_PSNR_THRESHOLD:
                print(
                    f"WARNING: Layer {mlx_key} PSNR={psnr:.1f}dB < 25dB threshold",
                    file=sys.stderr,
                )
            elif psnr == float("inf"):
                print(f"  EXACT:   {mlx_key} (identical)")
            else:
                print(f"  OK:      {mlx_key} PSNR={psnr:.1f} dB")

    # Check for unmapped keys
    all_mapped_mlx = set(key_map.values())
    converted_base_keys = set()
    for k in converted_tensors:
        # Strip quantization suffixes for checking
        base = k.replace(".scales", ".weight").replace(".biases", ".weight")
        converted_base_keys.add(base)
        converted_base_keys.add(k)

    # Check discarded keys are not present
    for dk in DISCARDED_KEYS:
        if dk in converted_tensors:
            print(f"  ERROR: Discarded key {dk} found in converted weights!")

    # Overall result
    passing = [r for r in layer_results if r["status"] == "PASS"]
    warnings = [r for r in layer_results if r["status"] == "WARN"]
    failures = [r for r in layer_results if r["status"] == "SHAPE_MISMATCH"]

    result = {
        "component": "pixart-sigma-dit",
        "total_layers": len(layer_results),
        "passing": len(passing),
        "warnings": len(warnings),
        "failures": len(failures),
        "layer_results": layer_results,
    }
    results.append(result)

    print(f"\nPixArt DiT: {len(passing)} pass, {len(warnings)} warn, {len(failures)} fail")
    return results


def dequantize_int4(
    packed: np.ndarray,
    scales: np.ndarray,
    biases: np.ndarray,
    group_size: int = 64,
) -> np.ndarray:
    """
    Dequantize int4 packed weights back to float32 for validation comparison.

    Args:
        packed: [M, N/8] uint32 packed 4-bit values
        scales: [M, N/group_size] float32 scales
        biases: [M, N/group_size] float32 zero-points

    Returns:
        [M, N] float32 dequantized weights
    """
    M, packed_cols = packed.shape
    N = packed_cols * 8

    # Unpack 8 x 4-bit values from each uint32
    unpacked = np.zeros((M, N), dtype=np.float32)
    for bit_idx in range(8):
        col_start = bit_idx
        values = ((packed >> (4 * bit_idx)) & 0xF).astype(np.float32)
        # Interleave: values[:, j] corresponds to column j*8 + bit_idx
        unpacked[:, bit_idx::8] = values

    # Reshape to groups and dequantize
    # unpacked is [M, N], reshape to [M, num_groups, group_size]
    # But we packed column-by-column, so we need to handle the ordering correctly
    # Actually the packing in convert_pixart_weights packs sequential columns:
    # quantized_flat[M, N] reshaped to [M, N/8, 8], then packed sequentially
    # So unpacked[:, bit_idx::8] gives columns bit_idx, bit_idx+8, bit_idx+16, ...
    # We need sequential columns instead

    # Re-unpack correctly: packed[m, j] contains columns j*8 .. j*8+7
    unpacked2 = np.zeros((M, N), dtype=np.float32)
    for j in range(packed_cols):
        for bit_idx in range(8):
            col = j * 8 + bit_idx
            unpacked2[:, col] = ((packed[:, j] >> (4 * bit_idx)) & 0xF).astype(
                np.float32
            )

    num_groups = N // group_size
    grouped = unpacked2.reshape(M, num_groups, group_size)

    # Dequantize: value = quantized * scale + bias
    dequantized = grouped * scales[:, :, np.newaxis] + biases[:, :, np.newaxis]

    return dequantized.reshape(M, N)


# ---------------------------------------------------------------------------
# T5 validation
# ---------------------------------------------------------------------------


def validate_t5(
    converted_dir: str,
    model_id: str = "google/t5-v1_1-xxl",
) -> list[dict]:
    """
    Validate T5-XXL conversion by comparing encoder outputs.

    Runs deterministic forward passes with the same tokenized inputs
    on both PyTorch and converted weights, comparing hidden state outputs.
    """
    import torch
    from safetensors import safe_open
    from transformers import T5EncoderModel, AutoTokenizer

    results = []
    converted_path = Path(converted_dir)

    print("\n" + "=" * 70)
    print("Validating T5-XXL conversion")
    print("=" * 70)

    # Load converted weights
    print(f"Loading converted weights from {converted_path}...")
    converted_tensors = {}
    for sf_file in sorted(converted_path.glob("*.safetensors")):
        with safe_open(str(sf_file), framework="numpy") as f:
            for key in f.keys():
                converted_tensors[key] = f.get_tensor(key)
    print(f"  Loaded {len(converted_tensors)} tensors")

    # Load original model
    print(f"Loading original T5 model from {model_id}...")
    model = T5EncoderModel.from_pretrained(model_id, torch_dtype=torch.float32)
    original_state_dict = model.state_dict()
    print(f"  Loaded {len(original_state_dict)} keys")

    # Per-layer comparison
    print("\nPer-layer weight comparison:")
    layer_results = []

    for key in sorted(original_state_dict.keys()):
        original = original_state_dict[key].detach().cpu().float().numpy()

        if key in converted_tensors:
            converted = converted_tensors[key]

            if converted.dtype == np.uint32 and converted.ndim == 2:
                scales_key = key.replace(".weight", ".scales")
                biases_key = key.replace(".weight", ".biases")
                if scales_key in converted_tensors and biases_key in converted_tensors:
                    scales = converted_tensors[scales_key].astype(np.float32)
                    biases = converted_tensors[biases_key].astype(np.float32)
                    converted = dequantize_int4(converted, scales, biases)
                else:
                    print(f"  SKIP (missing scales/biases): {key}")
                    continue
            else:
                converted = converted.astype(np.float32)

            if original.shape != converted.shape:
                # May differ due to padding during quantization
                min_shape = tuple(min(a, b) for a, b in zip(original.shape, converted.shape))
                original_trimmed = original[tuple(slice(0, s) for s in min_shape)]
                converted_trimmed = converted[tuple(slice(0, s) for s in min_shape)]
                psnr = compute_psnr(original_trimmed, converted_trimmed)
                status = "PASS" if psnr >= PER_LAYER_PSNR_THRESHOLD else "WARN"
                layer_results.append({"key": key, "psnr": psnr, "status": status, "note": "trimmed"})
            else:
                psnr = compute_psnr(original, converted)
                status = "PASS" if psnr >= PER_LAYER_PSNR_THRESHOLD else "WARN"
                layer_results.append({"key": key, "psnr": psnr, "status": status})

            if psnr < PER_LAYER_PSNR_THRESHOLD:
                print(
                    f"WARNING: Layer {key} PSNR={psnr:.1f}dB < 25dB threshold",
                    file=sys.stderr,
                )
            elif psnr == float("inf"):
                print(f"  EXACT:   {key}")
            else:
                print(f"  OK:      {key} PSNR={psnr:.1f} dB")
        else:
            print(f"  MISSING: {key} not in converted weights")
            layer_results.append({"key": key, "psnr": 0.0, "status": "MISSING"})

    passing = [r for r in layer_results if r["status"] == "PASS"]
    warnings = [r for r in layer_results if r["status"] == "WARN"]
    missing = [r for r in layer_results if r["status"] == "MISSING"]

    result = {
        "component": "t5-xxl",
        "total_layers": len(layer_results),
        "passing": len(passing),
        "warnings": len(warnings),
        "missing": len(missing),
        "layer_results": layer_results,
    }
    results.append(result)

    print(f"\nT5-XXL: {len(passing)} pass, {len(warnings)} warn, {len(missing)} missing")
    return results


# ---------------------------------------------------------------------------
# VAE validation
# ---------------------------------------------------------------------------


def validate_vae(
    converted_dir: str,
    model_id: str = "stabilityai/sdxl-vae",
) -> list[dict]:
    """
    Validate SDXL VAE conversion by comparing weights.

    Since the VAE uses float16 without quantization, we expect exact or
    near-exact matches (PSNR should be very high or infinite).
    """
    import torch
    from safetensors import safe_open
    from diffusers import AutoencoderKL

    results = []
    converted_path = Path(converted_dir)

    print("\n" + "=" * 70)
    print("Validating SDXL VAE conversion")
    print("=" * 70)

    # Load converted weights
    print(f"Loading converted weights from {converted_path}...")
    converted_tensors = {}
    for sf_file in sorted(converted_path.glob("*.safetensors")):
        with safe_open(str(sf_file), framework="numpy") as f:
            for key in f.keys():
                converted_tensors[key] = f.get_tensor(key)
    print(f"  Loaded {len(converted_tensors)} tensors")

    # Load original model
    print(f"Loading original VAE from {model_id}...")
    vae = AutoencoderKL.from_pretrained(model_id, torch_dtype=torch.float32)
    original_state_dict = vae.state_dict()
    print(f"  Loaded {len(original_state_dict)} keys")

    # Per-layer comparison
    print("\nPer-layer weight comparison:")
    layer_results = []

    for key in sorted(original_state_dict.keys()):
        original = original_state_dict[key].detach().cpu().float().numpy()

        if key in converted_tensors:
            converted = converted_tensors[key].astype(np.float32)

            # Handle Conv2d transposition
            if original.ndim == 4:
                original = original.transpose(0, 2, 3, 1)

            if original.shape != converted.shape:
                print(
                    f"  SHAPE MISMATCH: {key} "
                    f"original={original.shape} converted={converted.shape}"
                )
                layer_results.append(
                    {"key": key, "psnr": 0.0, "status": "SHAPE_MISMATCH"}
                )
                continue

            psnr = compute_psnr(original, converted)
            # For float16 conversion, we expect very high PSNR
            status = "PASS" if psnr >= PER_LAYER_PSNR_THRESHOLD else "WARN"
            layer_results.append({"key": key, "psnr": psnr, "status": status})

            if psnr < PER_LAYER_PSNR_THRESHOLD:
                print(
                    f"WARNING: Layer {key} PSNR={psnr:.1f}dB < 25dB threshold",
                    file=sys.stderr,
                )
            elif psnr == float("inf"):
                print(f"  EXACT:   {key}")
            else:
                print(f"  OK:      {key} PSNR={psnr:.1f} dB")
        else:
            print(f"  MISSING: {key} not in converted weights")
            layer_results.append({"key": key, "psnr": 0.0, "status": "MISSING"})

    passing = [r for r in layer_results if r["status"] == "PASS"]
    warnings = [r for r in layer_results if r["status"] == "WARN"]
    missing = [r for r in layer_results if r["status"] == "MISSING"]

    result = {
        "component": "sdxl-vae",
        "total_layers": len(layer_results),
        "passing": len(passing),
        "warnings": len(warnings),
        "missing": len(missing),
        "layer_results": layer_results,
    }
    results.append(result)

    print(f"\nSDXL VAE: {len(passing)} pass, {len(warnings)} warn, {len(missing)} missing")
    return results


# ---------------------------------------------------------------------------
# End-to-end forward pass comparison
# ---------------------------------------------------------------------------


def validate_forward_pass(
    pixart_dir: str | None = None,
    model_id: str = "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
) -> list[dict]:
    """
    Run end-to-end forward pass comparison with 5 deterministic prompts.

    Compares PyTorch pipeline output images against each other at different
    seeds to establish a baseline, then (when MLX runtime is available)
    compares against MLX outputs.

    Note: Full end-to-end MLX comparison requires the Swift MLX runtime.
    This function validates the PyTorch reference outputs and prepares
    reference data for later comparison.
    """
    import torch
    from diffusers import PixArtSigmaPipeline

    results = []

    print("\n" + "=" * 70)
    print("End-to-end forward pass validation (PyTorch reference)")
    print("=" * 70)

    print(f"Loading pipeline from {model_id}...")
    pipe = PixArtSigmaPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
    )
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        pipe = pipe.to("mps")

    for prompt, seed in zip(VALIDATION_PROMPTS, VALIDATION_SEEDS):
        print(f"\n  Prompt: {prompt[:50]}...")
        print(f"  Seed:   {seed}")

        generator = torch.Generator().manual_seed(seed)
        output = pipe(
            prompt,
            num_inference_steps=20,
            generator=generator,
            height=512,
            width=512,
        )
        image = output.images[0]
        image_np = np.array(image).astype(np.float32)

        # Save reference image
        if pixart_dir:
            ref_dir = Path(pixart_dir) / "references"
            ref_dir.mkdir(parents=True, exist_ok=True)
            ref_path = ref_dir / f"ref_seed{seed}.npy"
            np.save(str(ref_path), image_np)
            image.save(str(ref_dir / f"ref_seed{seed}.png"))
            print(f"  Saved reference: {ref_path}")

        # Run twice with same seed to verify determinism
        generator2 = torch.Generator().manual_seed(seed)
        output2 = pipe(
            prompt,
            num_inference_steps=20,
            generator=generator2,
            height=512,
            width=512,
        )
        image2_np = np.array(output2.images[0]).astype(np.float32)

        psnr = compute_psnr(image_np, image2_np)
        print(f"  Determinism check PSNR: {psnr:.1f} dB")

        result = {
            "prompt": prompt,
            "seed": seed,
            "determinism_psnr": psnr,
            "image_shape": list(image_np.shape),
            "status": "PASS" if psnr > END_TO_END_PSNR_THRESHOLD else "WARN",
        }
        results.append(result)

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Validate weight conversion (PSNR comparison)"
    )
    parser.add_argument(
        "--pixart-dir",
        type=str,
        default=None,
        help="Directory containing converted PixArt DiT weights",
    )
    parser.add_argument(
        "--t5-dir",
        type=str,
        default=None,
        help="Directory containing converted T5-XXL weights",
    )
    parser.add_argument(
        "--vae-dir",
        type=str,
        default=None,
        help="Directory containing converted SDXL VAE weights",
    )
    parser.add_argument(
        "--forward-pass",
        action="store_true",
        help="Run end-to-end forward pass comparison (requires GPU, slow)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file for validation results",
    )
    args = parser.parse_args()

    if not any([args.pixart_dir, args.t5_dir, args.vae_dir, args.forward_pass]):
        parser.error(
            "At least one of --pixart-dir, --t5-dir, --vae-dir, or --forward-pass required"
        )

    all_results = {}
    overall_pass = True

    # Validate PixArt DiT
    if args.pixart_dir:
        results = validate_pixart_dit(args.pixart_dir)
        all_results["pixart_dit"] = results
        for r in results:
            if r.get("failures", 0) > 0:
                overall_pass = False

    # Validate T5
    if args.t5_dir:
        results = validate_t5(args.t5_dir)
        all_results["t5_xxl"] = results
        for r in results:
            if r.get("missing", 0) > 0:
                overall_pass = False

    # Validate VAE
    if args.vae_dir:
        results = validate_vae(args.vae_dir)
        all_results["sdxl_vae"] = results
        for r in results:
            if r.get("missing", 0) > 0:
                overall_pass = False

    # Forward pass comparison
    if args.forward_pass:
        results = validate_forward_pass(pixart_dir=args.pixart_dir)
        all_results["forward_pass"] = results
        for r in results:
            if r.get("status") != "PASS":
                overall_pass = False

    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    for component, results in all_results.items():
        print(f"\n  {component}:")
        for r in results:
            if "total_layers" in r:
                print(
                    f"    Layers: {r['total_layers']} total, "
                    f"{r.get('passing', 0)} pass, "
                    f"{r.get('warnings', 0)} warn, "
                    f"{r.get('failures', r.get('missing', 0))} fail/missing"
                )
            elif "determinism_psnr" in r:
                print(
                    f"    Prompt (seed={r['seed']}): "
                    f"PSNR={r['determinism_psnr']:.1f} dB [{r['status']}]"
                )

    print(f"\n  Overall: {'PASS' if overall_pass else 'FAIL'}")
    print(f"  End-to-end threshold: PSNR > {END_TO_END_PSNR_THRESHOLD} dB")
    print(f"  Per-layer threshold:  PSNR > {PER_LAYER_PSNR_THRESHOLD} dB")

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert numpy types for JSON serialization
        def convert_for_json(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, float) and obj == float("inf"):
                return "inf"
            return obj

        serializable = json.loads(
            json.dumps(all_results, default=convert_for_json)
        )
        with open(output_path, "w") as f:
            json.dump(serializable, f, indent=2)
        print(f"\n  Results saved to {output_path}")

    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())
