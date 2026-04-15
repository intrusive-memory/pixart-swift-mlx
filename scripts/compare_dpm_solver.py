#!/usr/bin/env python3
"""
Direct comparison: diffusers DPMSolverMultistepScheduler vs our Swift port.

Runs both implementations with identical inputs and reports where they diverge.
Uses only the scheduler math — no model forward pass needed.

The Swift DPMSolverScheduler implements:
  - First-order: DDIM-like step
  - Second-order: DPM-Solver++(2M) midpoint update

We verify each step matches diffusers exactly.

Usage:
    python3 scripts/compare_dpm_solver.py
"""

import math
import numpy as np
import torch

TRAIN_TIMESTEPS = 1000
BETA_START = 0.0001
BETA_END = 0.02
INFERENCE_STEPS = 20
LATENT_SHAPE = (1, 4, 64, 64)  # 512×512 → 64×64 latents
SEED = 42


# ─── Reference: linear beta schedule ─────────────────────────────────────────

def make_alphas_cumprod():
    betas = np.linspace(BETA_START, BETA_END, TRAIN_TIMESTEPS, dtype=np.float64)
    alphas = 1.0 - betas
    return np.cumprod(alphas)


ALPHAS_CUMPROD = make_alphas_cumprod()


# ─── Reference: diffusers timestep schedule ───────────────────────────────────

def diffusers_timesteps(train_steps: int, inference_steps: int):
    """Matches DPMSolverMultistepScheduler.set_timesteps()."""
    step_ratio = train_steps / inference_steps
    timesteps = (np.arange(0, inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
    return timesteps.tolist()


def swift_timesteps(train_steps: int, inference_steps: int):
    """Matches DPMSolverScheduler.configure() in Swift."""
    step_ratio = float(train_steps) / float(inference_steps)
    ts = [int(float(train_steps - 1) - float(i) * step_ratio + 0.5) for i in range(inference_steps)]
    return [max(0, t) for t in ts]


# ─── Swift DPM-Solver port ────────────────────────────────────────────────────

def swift_alpha(t: int) -> float:
    idx = min(max(0, t), len(ALPHAS_CUMPROD) - 1)
    return float(ALPHAS_CUMPROD[idx])


def swift_first_order(x0_pred: np.ndarray, timestep: int, sample: np.ndarray,
                       plan_timesteps: list) -> np.ndarray:
    """Swift dpmSolverFirstOrderStep."""
    ac_t = swift_alpha(timestep)
    sqrt_alpha_t = math.sqrt(ac_t)
    sqrt_one_minus_t = math.sqrt(max(1.0 - ac_t, 1e-8))

    # Find prev timestep in plan (next lower in denoising direction)
    try:
        idx = plan_timesteps.index(timestep)
        prev_t = plan_timesteps[idx + 1] if idx + 1 < len(plan_timesteps) else 0
    except ValueError:
        prev_t = 0

    ac_prev = swift_alpha(prev_t) if prev_t > 0 else 1.0
    sqrt_alpha_prev = math.sqrt(ac_prev)
    sqrt_one_minus_prev = math.sqrt(max(1.0 - ac_prev, 1e-8))

    pred_noise = (sample - sqrt_alpha_t * x0_pred) / sqrt_one_minus_t
    return sqrt_alpha_prev * x0_pred + sqrt_one_minus_prev * pred_noise


def swift_second_order(x0_pred: np.ndarray, prev_x0_pred: np.ndarray,
                        timestep: int, prev_timestep: int, sample: np.ndarray,
                        plan_timesteps: list) -> np.ndarray:
    """Swift dpmSolverSecondOrderStep."""
    try:
        idx = plan_timesteps.index(timestep)
        target_t = plan_timesteps[idx + 1] if idx + 1 < len(plan_timesteps) else 0
    except ValueError:
        target_t = 0

    ac_t  = swift_alpha(target_t)
    ac_s0 = swift_alpha(timestep)
    ac_s1 = swift_alpha(prev_timestep)

    alpha_t  = math.sqrt(ac_t)
    sigma_t  = math.sqrt(max(1.0 - ac_t, 1e-8))
    alpha_s0 = math.sqrt(ac_s0)
    sigma_s0 = math.sqrt(max(1.0 - ac_s0, 1e-8))
    alpha_s1 = math.sqrt(ac_s1)
    sigma_s1 = math.sqrt(max(1.0 - ac_s1, 1e-8))

    lambda_t  = math.log(alpha_t)  - math.log(sigma_t)
    lambda_s0 = math.log(alpha_s0) - math.log(sigma_s0)
    lambda_s1 = math.log(alpha_s1) - math.log(sigma_s1)

    h  = lambda_t  - lambda_s0
    h0 = lambda_s0 - lambda_s1
    r0 = max(h0 / h, 1e-8)

    d1_coeff = 1.0 / r0
    d1 = d1_coeff * (x0_pred - prev_x0_pred)

    exp_minus_h = math.exp(-h)
    coeff = float(-alpha_t * (exp_minus_h - 1.0))
    sigma_ratio = float(sigma_t / sigma_s0)

    return sigma_ratio * sample + coeff * (x0_pred + 0.5 * d1)


def epsilon_to_x0(epsilon: np.ndarray, sample: np.ndarray, timestep: int) -> np.ndarray:
    """Swift epsilon prediction conversion."""
    ac = swift_alpha(timestep)
    sqrt_alpha = math.sqrt(ac)
    sqrt_one_minus = math.sqrt(max(1.0 - ac, 1e-8))
    return (sample - sqrt_one_minus * epsilon) / sqrt_alpha


# ─── Diffusers DPM-Solver reference ──────────────────────────────────────────

def run_diffusers_scheduler(latent: np.ndarray, epsilon_fn):
    """Run diffusers DPMSolverMultistepScheduler and return per-step latent means."""
    from diffusers import DPMSolverMultistepScheduler

    sched = DPMSolverMultistepScheduler(
        beta_start=BETA_START,
        beta_end=BETA_END,
        beta_schedule="linear",
        num_train_timesteps=TRAIN_TIMESTEPS,
        solver_order=2,
        prediction_type="epsilon",
    )
    sched.set_timesteps(INFERENCE_STEPS)

    x = torch.from_numpy(latent.copy())
    results = []
    for t in sched.timesteps:
        eps = torch.from_numpy(epsilon_fn(x.numpy(), int(t)))
        x = sched.step(eps, t, x, return_dict=False)[0]
        means = [float(x[0, c].mean()) for c in range(4)]
        results.append({"t": int(t), "means": means})
        print(f"  diffusers step t={int(t):>4}: ch_means={[f'{v:.5f}' for v in means]}")

    return results, x.numpy()


def run_swift_scheduler(latent: np.ndarray, epsilon_fn):
    """Run Swift DPM-Solver port and return per-step latent means."""
    plan = swift_timesteps(TRAIN_TIMESTEPS, INFERENCE_STEPS)

    x = latent.copy()
    previous_outputs = []  # (x0_pred, timestep) pairs
    results = []

    for step_i, t in enumerate(plan):
        eps = epsilon_fn(x, t)
        x0_pred = epsilon_to_x0(eps, x, t)

        if len(previous_outputs) == 0:
            x_new = swift_first_order(x0_pred, t, x, plan)
        else:
            prev_x0, prev_t = previous_outputs[-1]
            x_new = swift_second_order(x0_pred, prev_x0, t, prev_t, x, plan)

        previous_outputs.append((x0_pred, t))
        if len(previous_outputs) > 2:
            previous_outputs.pop(0)

        x = x_new
        means = [float(x[0, c].mean()) for c in range(4)]
        results.append({"t": t, "means": means})
        print(f"  swift    step t={t:>4}: ch_means={[f'{v:.5f}' for v in means]}")

    return results, x


# ─── Fake model: returns a fixed noise pattern per timestep ──────────────────

class FakeModel:
    """Returns channel-biased noise to expose scheduler differences."""
    def __init__(self, seed: int):
        rng = np.random.default_rng(seed)
        self.base_noise = rng.standard_normal(LATENT_SHAPE).astype(np.float32)
        # Bias: ch3 prediction slightly positive to match real model behavior
        self.base_noise[0, 3] += 0.05

    def __call__(self, x: np.ndarray, t: int) -> np.ndarray:
        # Simple: model predicts noise proportional to the current latent + base noise
        # This is a rough approximation; real models are much more complex
        # Crucially: same noise is returned regardless of t, so any difference
        # between schedulers is purely from the scheduler math
        return self.base_noise


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("DPM-Solver Comparison: diffusers vs Swift port")
    print("=" * 70)

    # ── Verify timesteps match ──
    d_ts = diffusers_timesteps(TRAIN_TIMESTEPS, INFERENCE_STEPS)
    s_ts = swift_timesteps(TRAIN_TIMESTEPS, INFERENCE_STEPS)

    print(f"\nDiffusers timesteps: {d_ts}")
    print(f"Swift    timesteps: {s_ts}")
    if d_ts == s_ts:
        print("✓ Timesteps MATCH")
    else:
        print("✗ Timesteps DIFFER")
        for i, (d, s) in enumerate(zip(d_ts, s_ts)):
            if d != s:
                print(f"  step {i}: diffusers={d}, swift={s}")

    # ── Same initial latent for both ──
    rng = np.random.default_rng(SEED)
    initial_latent = rng.standard_normal(LATENT_SHAPE).astype(np.float32)
    print(f"\nInitial latent ch_means: {[float(initial_latent[0,c].mean()) for c in range(4)]}")

    # ── Run both with fake model ──
    model = FakeModel(SEED + 1)

    print("\n── Diffusers scheduler ──")
    d_results, d_final = run_diffusers_scheduler(initial_latent, model)

    print("\n── Swift scheduler port ──")
    s_results, s_final = run_swift_scheduler(initial_latent, model)

    # ── Compare per-step ──
    print("\n── Per-step channel mean difference (Swift - Diffusers) ──")
    print(f"{'step':>5} {'t':>5} {'Δch0':>10} {'Δch1':>10} {'Δch2':>10} {'Δch3':>10}")
    for i, (dr, sr) in enumerate(zip(d_results, s_results)):
        diffs = [sr['means'][c] - dr['means'][c] for c in range(4)]
        flag = " ←" if any(abs(d) > 0.001 for d in diffs) else ""
        print(f"{i+1:>5} {dr['t']:>5} {diffs[0]:>10.5f} {diffs[1]:>10.5f} {diffs[2]:>10.5f} {diffs[3]:>10.5f}{flag}")

    # ── Final latent comparison ──
    print("\n── Final latent channel means ──")
    for c in range(4):
        d_mean = float(d_final[0, c].mean())
        s_mean = float(s_final[0, c].mean())
        print(f"  ch{c}: diffusers={d_mean:.5f}  swift={s_mean:.5f}  Δ={s_mean-d_mean:.5f}")

    max_diff = max(abs(float(d_final[0,c].mean()) - float(s_final[0,c].mean())) for c in range(4))
    if max_diff < 0.001:
        print("✓ Final latents MATCH within tolerance")
    else:
        print(f"✗ Final latents DIFFER (max channel mean diff = {max_diff:.5f})")

    print("\n── Timestep comparison detail ──")
    print(f"  Diffusers last timestep: {d_ts[-1]}  (alpha_cumprod={ALPHAS_CUMPROD[d_ts[-1]]:.6f})")
    print(f"  Swift    last timestep: {s_ts[-1]}  (alpha_cumprod={ALPHAS_CUMPROD[s_ts[-1]]:.6f})")


if __name__ == "__main__":
    main()
