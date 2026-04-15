# Generation Failure Diagnosis: pixart-swift-mlx vs flux-2-swift-mlx

**Date**: 2026-04-15  
**Purpose**: Root-cause analysis of why the PixArt-Sigma port produces bad output when Flux-2 did not.  
**Scope**: Planning only — no code changes in this document.

---

## Executive Summary

The failures span all three dimensions — model, library, and approach — but they are not equally weighted. The **approach** is the primary failure: the conversion was done without step-by-step validation against a Python reference, which allowed multiple compounding model errors to remain invisible. The **model** is the secondary failure: PixArt-Sigma's AdaLN-single architecture with micro-conditioning is fundamentally more complex than Flux-2's conditioning system, and the int4 weight conversion omitted critical embedder weights. The **library** (MLX) is not the cause; one known bug was correctly worked around.

---

## Why Flux-2 Converted Easily

### 1. The Scheduler Was Simple

Flux-2 uses Flow Matching with Euler steps. Sigmas are in `[0, 1]`, the formula is one line (`dx = (x - pred) * dt`), and there are no beta schedules, variance schedules, or prediction-type flags to get wrong. The scheduler had exactly one tunable parameter (empirical mu for resolution-aware time shifting), which is documented in the original paper and straightforward to port.

PixArt-Sigma uses a DPM-Solver with a DDPM-derived beta schedule. This introduces `betaStart`, `betaEnd`, `beta_schedule` (linear vs. scaled_linear vs. cosine), `prediction_type` (epsilon vs. v-prediction vs. sample), and `solver_order`. All of these must match the trained model exactly.

### 2. The Conditioning Was Transparent

Flux-2's conditioning path is: `T5 embedding → linear projection → concatenated to image sequence`. The conditioning tensor flows in one continuous path through the double-stream blocks, with joint attention. There is no side path, no secondary embedding, no micro-conditioning.

PixArt-Sigma's conditioning path is: `T5 embedding → cross-attention` AND `timestep → sinusoidal → MLP → (+ size_emb + ar_emb) → t_block AdaLN → (scale, shift, gate) × 28 blocks`. The micro-conditioning is a side channel that modifies the ENTIRE adaptive layer normalization system. If any part of this side channel is wrong, all 28 blocks produce corrupted outputs.

### 3. The Architecture Matched Existing MLX Patterns

Flux-2's architecture (RoPE, double-stream, patchified latents) had clear analogues in existing MLX/LLM code. The tensor operations were straightforward to map. The patchification (`[B, 32, H/8, W/8]` → `[B, seq, 128]`) is a simple reshape/transpose that the team had done before.

PixArt-Sigma's DiT architecture with AdaLN-single had no pre-existing pattern in the team's Swift/MLX work. The modulation path required understanding exactly how `adaln_single`, `t_block`, and the per-block `(shift, scale, gate)` triples interact.

### 4. Flux-2 Was Validated Incrementally

The Flux-2 project has 2813 lines of tests, a mock pipeline, per-component unit tests (latent packing, RoPE, attention shapes, scheduler steps). By the time the full pipeline ran, each component had been individually verified.

The PixArt project has integration tests but minimal unit tests for individual architectural components. The validation strategy was "run the full pipeline and see what happens."

---

## Where PixArt-Sigma Fails

### FAILURE 1: Missing Micro-Conditioning Weights (Critical)

**What should happen**: The `adaln_single` module receives a combined embedding:
```
t_emb = timestep_embedder(timestep)         # [B, 1152]
s_emb = size_embedder(height, width)        # [B, 256]  (sinusoidal on pixel counts)
r_emb = ar_embedder(aspect_ratio)           # [B, 256]  (sinusoidal on ratio)
combined = t_block(cat([t_emb, s_emb, r_emb]))  # [B, 6*1152] → 6 modulation params per block
```

**What actually happens**: The `SizeEmbedder` and `AspectRatioEmbedder` instances are created in `PixArtDiT.__init__()` but **their weights were not included in the int4 safetensors** and **they are never called in the forward pass**. The `t_block` receives only `t_emb` (wrong size and wrong trained distribution).

**Evidence**: `Sources/PixArtBackbone/PixArtDiT.swift:151–154` — the comment explicitly states micro-conditions are skipped. `scripts/convert_pixart_weights.py:65–91` includes key mappings for these weights, but they are silently skipped as "unmapped_keys" at conversion time.

**Effect**: Every one of the 28 DiT blocks computes its `(shift, scale, gate)` modulation triple from a corrupted timestep embedding. The AdaLN outputs for ALL blocks are systematically wrong. This alone would produce incoherent generation.

**Fix path**: Either (a) include `sizeEmbedder` and `arEmbedder` weights in the conversion and call them in the forward pass with actual resolution/AR values, or (b) verify that the int4 checkpoint genuinely omits them (some distilled models drop micro-conditioning) and remove the dead code.

---

### FAILURE 2: Beta Schedule Is Wrong (Critical)

**What should happen**: The PixArt-Sigma DPM-Solver uses `beta_schedule: linear`, `beta_start: 0.0001`, `beta_end: 0.02`, `prediction_type: epsilon`. This is documented in the HuggingFace `scheduler_config.json`.

**What actually happens**: `Sources/PixArtBackbone/PixArtRecipe.swift:65–71` uses `.scaledLinear`. The comment in the file states that **both `linear` and `scaledLinear` produce severe color distortion at 1024×1024**, which means the bug is NOT simply "wrong schedule name" — there is a deeper parameter mismatch.

**ARCHITECTURE.md documents** the correct config as `.linear(0.0001, 0.02)` with `predictionType: .epsilon`. The live code contradicts this. This divergence was introduced and not reverted after it failed to fix the problem.

**Effect**: The noise schedule controls how much noise is removed at each of the 20 DPM-Solver steps. A wrong schedule means denoising steps are applied at incorrect noise levels — producing systematic color drift, over-saturation, or near-black output depending on which direction the error runs.

**Fix path**: Revert to `linear` as documented. Then validate the DPMSolverScheduler implementation's `alphas_cumprod` computation against PyTorch's `diffusers.schedulers.DPMSolverMultistepScheduler` with identical parameters using a Python script.

---

### FAILURE 3: Position Embedding Scale Is Uncertain (High)

**What happened**: The `baseSize` parameter was `512` (wrong), then corrected to `128`. The comment in `Sources/PixArtBackbone/PixArtDiTConfiguration.swift:26–37` correctly identifies that `512` multiplied position coordinates by 4× relative to the trained grid, producing out-of-distribution sinusoidal frequencies.

**What is uncertain**: After the fix, the grid coordinates for a 1024×1024 image are computed as:
```
arange(128) / (128 / 128) / 2.0  →  range [0, 64)
```

Whether `[0, 64)` is the correct range for the trained model is **not validated**. The trained model expected coordinates derived from a base resolution of 512px (the PixArt-Sigma training resolution), which would produce a specific frequency distribution. The `peInterpolation: 2.0` value encodes this, but whether `baseSize: 128` combined with `peInterpolation: 2.0` reproduces the original Python behavior is unconfirmed.

**Fix path**: Run the Python reference implementation (`PixArtAlphaPipeline` from diffusers) and extract the actual position embedding tensor for a 1024×1024 input. Compare against the Swift output.

---

### FAILURE 4: No Step-By-Step Intermediate Validation (Approach)

The Flux-2 project validated each component in isolation before running the pipeline. The PixArt project attempted to run the full pipeline and debug from the output image.

This matters because the three failures above are **compounding**: a bad micro-conditioning signal corrupts the AdaLN modulations, then a wrong beta schedule applies denoising at wrong noise levels, then wrong position embeddings corrupt spatial attention. None of these errors produces a clean failure mode (crash, NaN, assertion). They all produce "bad image" — which is impossible to attribute without intermediate validation.

Without tensor-level comparison to a Python reference at each stage (after text encoding, after timestep embedding, after block 0, after block 28, after VAE decode), there is no way to know which bug is dominant.

---

## Comparative Summary

| Dimension | Flux-2 | PixArt-Sigma |
|-----------|--------|--------------|
| **Scheduler complexity** | Flow Matching, 1 param | DPM-Solver, 6 params, all must match |
| **Conditioning complexity** | Single path (T5 → linear → cross-attn) | Dual path (T5 → cross-attn + timestep+micro → AdaLN×28) |
| **Architecture novelty** | Similar to existing MLX LLM patterns | AdaLN-single with micro-conditioning, no prior art in this codebase |
| **Weight conversion** | Complete and verified | Missing micro-conditioning embedder weights |
| **Scheduler validation** | Matches Python reference | Diverges from documented config; both "correct" configs produce bad output |
| **Position embedding** | RoPE, validated | Sinusoidal with interpolation, fix unverified |
| **Testing strategy** | Unit tests per component | Integration tests only |
| **Validation strategy** | Incremental, compare to Python | Run full pipeline, inspect output |

---

## Recommended Investigation Order

These are **planning steps only** — no code changes until the root cause is confirmed.

### Step 1: Set Up a Python Reference Script

Before changing any Swift code, create a Python script that:
1. Loads the original `PixArt-alpha/PixArt-Sigma-XL-2-1024-MS` model from HuggingFace
2. Runs a single forward pass with a known seed and prompt
3. Saves intermediate tensors at every stage: `t_emb`, `s_emb`, `r_emb`, `combined_emb`, output of block 0, output of block 28
4. Also saves the scheduler's `alphas_cumprod` array and the first timestep's denoised latent

This gives ground-truth values to compare against Swift.

### Step 2: Validate the Beta Schedule in Isolation

Run the DPMSolverScheduler in Swift with `linear`, `betaStart: 0.0001`, `betaEnd: 0.02`, 20 steps. Print the full `timesteps` and `alphas_cumprod` arrays. Compare to the Python reference output. If they don't match exactly, the scheduler implementation has a bug independent of the backbone.

### Step 3: Validate the Micro-Conditioning Decision

Check whether the `intrusive-memory/pixart-sigma-xl-dit-int4-mlx` safetensors actually contain `sizeEmbedder` and `arEmbedder` weight keys. If they do: the forward pass is missing calls to them (fix the forward pass). If they don't: determine whether this is a distilled model that genuinely doesn't use micro-conditioning, or a conversion error (re-run the conversion with these weights included).

### Step 4: Validate Position Embeddings

Extract the actual position embedding tensor from the Python pipeline for a 1024×1024 input. Compare the shape, min, max, and a sample of values against the Swift `get2dSinusoidal` output with `baseSize: 128`, `peInterpolation: 2.0`.

### Step 5: Run a Single Block Forward Pass

With correct micro-conditioning and a validated scheduler, run a single DiT block forward pass in both Python and Swift with identical inputs. Compare:
- Input to block 0: latents, conditioning, modulation params
- Output of block 0

If these match, the architecture is correct. If they don't, find the specific operation that diverges.

### Step 6: Validate the VAE Decoder Separately

Use a known latent tensor (from the Python pipeline's intermediate output) and run only the VAE decoder in Swift. Verify that it produces a recognizable image. This isolates the backbone problem from any potential VAE issue.

---

## Hypothesis: Most Likely Root Cause

If forced to rank, the **missing micro-conditioning weights + wrong forward pass** is the most probable dominant failure. Here is why:

The AdaLN-single architecture computes a single set of modulation parameters from the combined `(timestep, size, ar)` embedding, then broadcasts these across all 28 blocks. If this embedding is wrong, EVERY block receives wrong `(shift, scale, gate)` values — meaning the denoiser cannot remove noise correctly at any layer. This produces incoherent output regardless of whether the scheduler is correct.

The wrong beta schedule would produce color-distorted but potentially recognizable output (you'd see a blurry version of the correct image with wrong colors). What's described and observed is more fundamental than color distortion — suggesting the backbone itself is producing garbage, which points to the conditioning path.

The position embedding issue is real but secondary — it would produce spatially blurry or low-frequency output rather than completely incoherent noise.

**Predicted fix sequence**: Fix micro-conditioning first → observe whether output becomes blurry-but-recognizable → then fix beta schedule → observe whether colors normalize → then verify position embeddings → observe whether spatial detail improves.

---

## Files to Change (When Planning is Done)

| File | Change Needed |
|------|---------------|
| `scripts/convert_pixart_weights.py` | Verify `sizeEmbedder`/`arEmbedder` keys exist in source model and are exported |
| `Sources/PixArtBackbone/PixArtDiT.swift` | Call `sizeEmbedder` and `arEmbedder` in forward pass; pass actual resolution/AR to pipeline |
| `Sources/PixArtBackbone/PixArtRecipe.swift` | Revert `betaSchedule` to `.linear(0.0001, 0.02)` |
| `Sources/PixArtBackbone/PixArtDiTConfiguration.swift` | Confirm `baseSize: 128` + `peInterpolation: 2.0` produces correct coordinate range |
| New: `scripts/validate_pipeline.py` | Step-by-step Python reference for intermediate tensor comparison |
| New: `Tests/PixArtBackboneTests/SchedulerValidationTests.swift` | Unit test: compare `alphas_cumprod` against saved Python reference values |
| New: `Tests/PixArtBackboneTests/EmbeddingTests.swift` | Unit test: compare `t_emb`, `s_emb`, `r_emb` shapes and sample values |
