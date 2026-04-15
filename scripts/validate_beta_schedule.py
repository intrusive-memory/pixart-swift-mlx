#!/usr/bin/env python3
"""
Validate PixArt-Sigma DPM-Solver beta schedule and key intermediate values.

This script produces reference values that can be compared against Swift
DPMSolverScheduler outputs to confirm the schedule implementation is correct.

No diffusers required — only numpy and safetensors.

Usage:
    python3 scripts/validate_beta_schedule.py

Output:
    - alphas_cumprod at key timesteps (compare against Swift scheduler)
    - sigmas at key timesteps (compare against Swift scheduler.configure())
    - Timestep sinusoidal embedding for t=500 (compare against Swift timestepSinusoidalEmbedding)
    - Source model weight stats (confirm shapes match conversion)
"""

import numpy as np
from pathlib import Path

SRC_SAFETENSORS = Path(
    "~/.cache/huggingface/hub/models--PixArt-alpha--PixArt-Sigma-XL-2-1024-MS/"
    "snapshots/e102b3591cc82e97071b8b4cb90d834d0c487207/transformer/diffusion_pytorch_model.safetensors"
).expanduser()

TRAIN_TIMESTEPS = 1000
BETA_START = 0.0001
BETA_END = 0.02


# ─── Beta Schedule ────────────────────────────────────────────────────────────

def compute_linear_schedule(train_steps: int, beta_start: float, beta_end: float):
    betas = np.linspace(beta_start, beta_end, train_steps, dtype=np.float64)
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas)
    return betas, alphas_cumprod


def compute_scaled_linear_schedule(train_steps: int, beta_start: float, beta_end: float):
    betas = np.linspace(np.sqrt(beta_start), np.sqrt(beta_end), train_steps, dtype=np.float64) ** 2
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas)
    return betas, alphas_cumprod


def compute_sigmas(alphas_cumprod):
    return np.sqrt((1.0 - alphas_cumprod) / np.maximum(alphas_cumprod, 1e-8))


# ─── Timestep Sinusoidal Embedding (matches PixArt-Sigma Timesteps module) ───

def timestep_sinusoidal_embedding(timestep: float, dim: int = 256) -> np.ndarray:
    """
    Reference implementation matching diffusers get_timestep_embedding for PixArt-Sigma:
      - downscale_freq_shift=0 → denominator is halfDim (not halfDim-1)
      - flip_sin_to_cos=True  → [cos, sin] order
    """
    half_dim = dim // 2
    freqs = np.exp(
        -np.log(10000.0) * np.arange(half_dim, dtype=np.float64) / half_dim
    )
    angles = timestep * freqs  # [halfDim]
    # flip_sin_to_cos → [cos, sin]
    return np.concatenate([np.cos(angles), np.sin(angles)])  # [256]


# ─── DPM-Solver Timestep Schedule ────────────────────────────────────────────

def compute_dpm_timesteps(train_steps: int, inference_steps: int):
    """Matches DPMSolverScheduler.configure() in Swift."""
    step_ratio = float(train_steps) / float(inference_steps)
    timesteps = [
        int(float(train_steps - 1) - float(i) * step_ratio + 0.5)
        for i in range(inference_steps)
    ]
    return [max(0, t) for t in timesteps]


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("PixArt-Sigma DPM-Solver Validation Reference")
    print("=" * 70)

    # ── Beta Schedule Comparison ──
    print("\n── Beta Schedule: linear vs scaledLinear ──")
    betas_lin, ac_lin = compute_linear_schedule(TRAIN_TIMESTEPS, BETA_START, BETA_END)
    betas_sl, ac_sl = compute_scaled_linear_schedule(TRAIN_TIMESTEPS, BETA_START, BETA_END)

    check_timesteps = [0, 100, 250, 499, 500, 750, 998, 999]
    print(f"{'t':>5}  {'linear ᾱ':>14}  {'scaledLinear ᾱ':>16}  {'ratio':>8}")
    for t in check_timesteps:
        ratio = ac_sl[t] / ac_lin[t] if ac_lin[t] > 0 else float("inf")
        print(f"{t:>5}  {ac_lin[t]:>14.8f}  {ac_sl[t]:>16.8f}  {ratio:>8.3f}x")

    print(f"\n[CORRECT] Use linear schedule: betas = linspace({BETA_START}, {BETA_END}, {TRAIN_TIMESTEPS})")
    print(f"[WRONG]   scaledLinear at t=500 is {ac_sl[500]/ac_lin[500]:.2f}x too slow to denoise")

    # ── Sigmas for 20 inference steps ──
    print("\n── Sigmas (20 inference steps, linear schedule) ──")
    sigmas_lin = compute_sigmas(ac_lin)
    dpm_timesteps = compute_dpm_timesteps(TRAIN_TIMESTEPS, 20)
    print(f"Timesteps (20 steps): {dpm_timesteps}")
    print(f"{'step':>5}  {'t':>5}  {'sigma':>12}  {'alpha_cumprod':>14}")
    for i, t in enumerate(dpm_timesteps):
        print(f"{i:>5}  {t:>5}  {sigmas_lin[t]:>12.6f}  {ac_lin[t]:>14.8f}")

    # ── Timestep Sinusoidal Embedding Spot-Check ──
    print("\n── Timestep Sinusoidal Embedding (t=500, first 8 of 256 values) ──")
    emb_500 = timestep_sinusoidal_embedding(500.0, dim=256)
    print(f"emb[0:8] = {emb_500[:8].tolist()}")
    print(f"emb[128:136] = {emb_500[128:136].tolist()}")
    print(f"emb norm = {np.linalg.norm(emb_500):.6f}")
    print("(first 128 values are cos, last 128 are sin — flip_sin_to_cos=True)")

    # ── Source Model Weight Validation ──
    print("\n── Source Model Weights ──")
    if SRC_SAFETENSORS.exists():
        try:
            from safetensors import safe_open
            with safe_open(str(SRC_SAFETENSORS), framework="numpy", device="cpu") as f:
                keys = sorted(f.keys())
            global_keys = [k for k in keys if not k.startswith("transformer_blocks.")]
            print(f"Source: {SRC_SAFETENSORS}")
            print(f"Total keys: {len(keys)} ({len(global_keys)} global, {len(keys)-len(global_keys)} block)")

            with safe_open(str(SRC_SAFETENSORS), framework="numpy", device="cpu") as f:
                for key in ["adaln_single.linear.weight", "adaln_single.linear.bias",
                            "adaln_single.emb.timestep_embedder.linear_1.weight",
                            "adaln_single.emb.timestep_embedder.linear_2.weight"]:
                    t = f.get_tensor(key)
                    print(f"  {key}: {t.shape} {t.dtype}")

            micro_keys = [k for k in keys if "resolution_embedder" in k or "aspect_ratio_embedder" in k]
            if micro_keys:
                print(f"\n  [PRESENT] Micro-conditioning keys: {micro_keys}")
            else:
                print("\n  [ABSENT]  No micro-conditioning keys — model runs without resolution/AR conditioning")
                print("  This is expected for PixArt-alpha/PixArt-Sigma-XL-2-1024-MS (adaln_single.linear is [6912, 1152])")
        except ImportError:
            print("safetensors not installed — skipping weight validation")
    else:
        print(f"Source not found: {SRC_SAFETENSORS}")
        print("Run: huggingface-cli download PixArt-alpha/PixArt-Sigma-XL-2-1024-MS")

    print("\n" + "=" * 70)
    print("Expected Swift DPMSolverScheduler.alphasCumprod[500] = 0.07780 (±0.0001)")
    print("Expected Swift DPMSolverScheduler.sigmas[step=10]   ≈", f"{sigmas_lin[dpm_timesteps[10]]:.6f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
