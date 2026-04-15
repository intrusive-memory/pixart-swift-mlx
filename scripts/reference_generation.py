#!/usr/bin/env python3
"""
Reference generation for PixArt-Sigma XL using diffusers.

Runs the same generation as the Swift fixture test (seed=42, 1024×1024,
"A red car parked on a cobblestone street") and saves:
  - Per-step latent statistics (to detect channel divergence)
  - The final decoded image as PNG
  - A JSON sidecar with all intermediate values

Run from the pixart-swift-mlx directory:
    python3 scripts/reference_generation.py [--steps 20] [--size 512]

Outputs to:
    /tmp/pixart-reference/
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

MODEL_ID = "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS"
CACHE_DIR = Path("~/.cache/huggingface/hub").expanduser()
OUTPUT_DIR = Path("/tmp/pixart-reference")

PROMPT = "A red car parked on a cobblestone street"
SEED = 42
STEPS = 20
SIZE = 1024
GUIDANCE = 4.5


def compute_channel_stats(latent: torch.Tensor, label: str) -> dict:
    """Compute per-channel mean/std of a latent tensor [B, C, H, W]."""
    b, c, h, w = latent.shape
    stats = {"label": label, "shape": list(latent.shape)}
    for ch in range(c):
        ch_data = latent[0, ch].float()
        stats[f"ch{ch}_mean"] = float(ch_data.mean())
        stats[f"ch{ch}_std"] = float(ch_data.std())
        stats[f"ch{ch}_min"] = float(ch_data.min())
        stats[f"ch{ch}_max"] = float(ch_data.max())
    return stats


def compute_image_stats(image: Image.Image) -> dict:
    """Compute channel means for a PIL RGB image."""
    arr = np.array(image).astype(float)
    return {
        "meanRed": float(arr[:, :, 0].mean()),
        "meanGreen": float(arr[:, :, 1].mean()),
        "meanBlue": float(arr[:, :, 2].mean()),
        "meanLuminance": float(
            0.299 * arr[:, :, 0].mean()
            + 0.587 * arr[:, :, 1].mean()
            + 0.114 * arr[:, :, 2].mean()
        ),
    }


def run_reference(steps: int = STEPS, size: int = SIZE) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading PixArt-Sigma from {MODEL_ID} ...")
    from diffusers import PixArtSigmaPipeline

    pipe = PixArtSigmaPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
    )
    pipe = pipe.to("cpu")
    pipe.scheduler.set_timesteps(steps)

    print(f"Scheduler: {pipe.scheduler.__class__.__name__}")
    print(f"Beta schedule: {pipe.scheduler.config.beta_schedule}")
    print(f"Timesteps ({len(pipe.scheduler.timesteps)}): {pipe.scheduler.timesteps.tolist()}")

    # ── Beta schedule reference ──────────────────────────────────────────────
    acs = pipe.scheduler.alphas_cumprod.numpy()
    sigmas = np.sqrt((1 - acs) / np.maximum(acs, 1e-8))
    sched_ref = {
        "beta_schedule": pipe.scheduler.config.beta_schedule,
        "timesteps": pipe.scheduler.timesteps.tolist(),
        "alphas_cumprod_spot": {
            str(t): float(acs[t]) for t in [0, 100, 250, 500, 750, 999]
        },
        "sigmas_at_inference_timesteps": [
            float(sigmas[t]) for t in pipe.scheduler.timesteps.tolist()
        ],
    }
    print("\nalphas_cumprod spot check:")
    for t, v in sched_ref["alphas_cumprod_spot"].items():
        print(f"  t={t:>4}: {v:.8f}")

    # ── Encode prompt ────────────────────────────────────────────────────────
    print(f"\nEncoding prompt: '{PROMPT}'")
    generator = torch.Generator().manual_seed(SEED)

    # We'll capture intermediate latents via a hook on the scheduler
    step_data = []

    original_step = pipe.scheduler.step

    def step_hook(model_output, timestep, sample, **kwargs):
        # Call WITHOUT return_dict override so we can capture both prev_sample and x0 pred
        result_dict = original_step(model_output, timestep, sample, return_dict=True)
        prev_sample = result_dict.prev_sample  # [1, 4, H/8, W/8]
        predicted_x0 = getattr(result_dict, "pred_original_sample", None)

        step_info = {
            "timestep": int(timestep),
            "latent_stats": compute_channel_stats(prev_sample, f"t={int(timestep)}"),
        }
        if predicted_x0 is not None:
            step_info["x0_pred_stats"] = compute_channel_stats(
                predicted_x0, f"x0_pred_t={int(timestep)}"
            )
        step_data.append(step_info)

        idx = len(step_data)
        s = prev_sample[0]
        ch_means = [float(s[c].mean()) for c in range(4)]
        print(f"  step {idx:>2} t={int(timestep):>4}: latent ch_means={[f'{v:.4f}' for v in ch_means]}")
        # Pipeline called with return_dict=False, so return tuple
        return (prev_sample,)

    pipe.scheduler.step = step_hook

    print(f"\nGenerating {size}×{size} image, seed={SEED}, steps={steps}, guidance={GUIDANCE} ...")
    output = pipe(
        prompt=PROMPT,
        height=size,
        width=size,
        num_inference_steps=steps,
        guidance_scale=GUIDANCE,
        generator=generator,
        output_type="pil",
    )

    image = output.images[0]
    img_stats = compute_image_stats(image)
    print(f"\nImage channel means: R={img_stats['meanRed']:.1f} G={img_stats['meanGreen']:.1f} B={img_stats['meanBlue']:.1f}")
    print(f"Luminance: {img_stats['meanLuminance']:.1f}")

    # Save image
    img_path = OUTPUT_DIR / f"pixart-ref-seed{SEED}-{size}px.png"
    image.save(str(img_path))
    print(f"Image saved: {img_path}")

    # Save full report JSON
    report = {
        "prompt": PROMPT,
        "seed": SEED,
        "steps": steps,
        "size": size,
        "guidance_scale": GUIDANCE,
        "scheduler": sched_ref,
        "per_step": step_data,
        "final_image_stats": img_stats,
    }
    json_path = OUTPUT_DIR / f"pixart-ref-seed{SEED}-{size}px.json"
    with open(str(json_path), "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report saved: {json_path}")


def main():
    parser = argparse.ArgumentParser(description="PixArt-Sigma reference generation")
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--size", type=int, default=SIZE)
    args = parser.parse_args()
    run_reference(steps=args.steps, size=args.size)


if __name__ == "__main__":
    main()
