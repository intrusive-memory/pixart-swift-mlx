---
feature_name: "OPERATION SIGMA FOUNDRY"
iteration: 2
wave: 3
repository: pixart-swift-mlx
status: in_progress
source_requirements: REQUIREMENTS.md
starting_point_commit: 9ea71031e6f836089c54f64c26313fb969b2bd81
mission_branch: mission/sigma-foundry/2
---

# EXECUTION_PLAN.md — pixart-swift-mlx

## Terminology

> **Mission** — A definable, testable scope of work. Defines scope, acceptance criteria, and dependency structure.

> **Sortie** — An atomic, testable unit of work executed by a single autonomous AI agent in one dispatch. One aircraft, one mission, one return.

> **Work Unit** — A grouping of sorties (package, component, phase).

We deliberately avoid agile/waterfall terminology (sprint, iteration, phase) because those map to **time**. Missions and sorties map to **agentic work cycles**, which have no inherent time dimension.

---

**Goal**: Deliver the PixArt-Sigma DiT backbone as a SwiftTubería model plugin. This package provides the DiT transformer, weight key mapping, pipeline recipe, Acervo descriptors, and a CLI tool. All shared infrastructure (weight loading, scheduling, VAE decoding, image rendering, memory management) comes from SwiftTubería.

**⚠️ Execution context**: All source files, tests, scripts, and CI workflow described in this plan already exist (see Open Questions §1). Every sortie is scoped as **verify against spec; fix any gaps** — not build from scratch.

---

## Work Units

| Work Unit | Directory | Sorties | Layer | Dependencies |
|-----------|-----------|---------|-------|--------------|
| PixArt Swift Package | `.` | 7 | 1 | none |
| Weight Conversion Scripts | `scripts/` | 1 | 2 | WU1-Sortie 3 complete |

**iOS testing is out of scope** for this iteration. The `build-ios` CI job should be removed and no iOS tests added. The `platforms: [.macOS(.v26), .iOS(.v26)]` declaration in Package.swift is kept as-is.

**Note**: Work Unit 2 (Sortie 8) does not wait for all of Work Unit 1. It unlocks as soon as Sortie 3 (key mapping) completes — the key names defined there are the sole Swift-side dependency for the Python scripts.

---

## Work Unit 1: PixArt Swift Package

### Sortie 1: Package Structure (P1)

**Priority**: 24.75 — Highest dependency depth (blocks all remaining sorties); establishes build system and platform constraints reused by every subsequent sortie.

**Entry criteria**:
- [ ] First sortie — no prerequisites

**Tasks** (verify existing files meet spec; fix any discrepancies):
1. Verify `Package.swift`: two products (`PixArtBackbone` library, `PixArtCLI` executable); dependencies SwiftTubería (≥0.2.8) and swift-argument-parser (≥1.3.0); `PixArtBackbone` target links `Tuberia` + `TuberiaCatalog`; `PixArtCLI` links `PixArtBackbone` + `ArgumentParser`
2. Verify platforms: `[.macOS(.v26), .iOS(.v26)]` present and Swift language mode `.v6`
3. Verify `Sources/PixArtBackbone/` contains at least one `.swift` source file
4. Verify `Sources/PixArtCLI/PixArtCLI.swift` exists and contains `@main`
5. Verify `Tests/PixArtBackboneTests/` contains at least one test file using Swift Testing (`@Suite` / `@Test`)
6. Run `make build` and `make test` to confirm package compiles and placeholder tests pass

**Exit criteria**:
- [ ] `make build` exits 0
- [ ] `make test` exits 0
- [ ] `Sources/PixArtBackbone/` contains at least one `.swift` file
- [ ] `Sources/PixArtCLI/PixArtCLI.swift` exists and contains `@main`
- [ ] `Tests/PixArtBackboneTests/` contains at least one test file
- [ ] `Package.swift` declares `platforms: [.macOS(.v26), .iOS(.v26)]`
- [ ] `Package.swift` specifies Swift `.v6` language mode

---

### Sortie 2: PixArt DiT Backbone (P2)

**Priority**: 23.5 — Core types and protocols reused by all subsequent WU1 sorties; complex algorithm risk (attention, embeddings, AdaLN-Zero).

**Entry criteria**:
- [ ] Sortie 1 complete (`PixArtBackbone` target compiles)

**Tasks** (verify existing files meet spec; fix any discrepancies):
1. Verify `Sources/PixArtBackbone/PixArtDiTConfiguration.swift`: all architecture constants present — hiddenSize: 1152, numHeads: 16, headDim: 72, depth: 28, patchSize: 2, inChannels: 4, outChannels: 8, mlpRatio: 4.0, captionChannels: 4096, maxTextLength: 120; 64-bucket aspect ratio table present
2. Verify `Sources/PixArtBackbone/Embeddings.swift`: 2D sinusoidal position embeddings (recomputed per forward pass); timestep sinusoidal embedding (dim 256 → MLP 256→1152); micro-condition embedders for resolution (2×384=768 flattened) and aspect ratio (384-dim)
3. Verify `Sources/PixArtBackbone/Attention.swift`: self-attention with QK normalization (LayerNorm on q and k after reshape); cross-attention (no timestep modulation); both use `MLXFast.scaledDotProductAttention`; separate Q/K/V projections matching diffusers format
4. Verify `Sources/PixArtBackbone/DiTBlock.swift`: norm1 → AdaLN-Zero self-attention → cross-attention (no modulation) → norm2 → AdaLN-Zero GEGLU FFN; per-block `scale_shift_table` (6 × 1152 params); `t2i_modulate(x, shift, scale) = x * (1 + scale) + shift` helper defined once
5. Verify `Sources/PixArtBackbone/FinalLayer.swift`: AdaLN (2-param: shift+scale only, no gate) → Linear(1152, 32) → unpatchify to `[B, H/8, W/8, 8]`; last dim sliced `[..<4]` → outlet `[B, H/8, W/8, 4]`
6. Verify `Sources/PixArtBackbone/PixArtDiT.swift`: patch embedding `Conv2d(4, 1152, kernel=2, stride=2)` → flatten to `[B, T, 1152]`; 2D sinusoidal pos embeddings added; caption projection `Linear(4096, 1152) → GELU(tanh) → Linear(1152, 1152)`; timestep pipeline: `sinusoidal(256) → MLP → concat micro-conditions → t_block: SiLU → Linear(1152, 6*1152)`; 28 DiT blocks; final layer; `Backbone` + `WeightedSegment` protocol conformance

**Exit criteria**:
- [ ] `PixArtDiT` conforms to `Backbone` (inlet: `BackboneInput`, outlet: `MLXArray`)
- [ ] `PixArtDiT` conforms to `WeightedSegment`
- [ ] `PixArtDiTConfiguration` has all required properties matching: hiddenSize=1152, numHeads=16, headDim=72, depth=28, patchSize=2, inChannels=4, outChannels=8, mlpRatio=4.0, captionChannels=4096, maxTextLength=120
- [ ] `expectedConditioningDim` == 4096, `outputLatentChannels` == 4, `expectedMaxSequenceLength` == 120
- [ ] DiT block array count verifiable as exactly 28 via `blocks.count == 28`
- [ ] GEGLU FFN: fc1 projects to 2×4608 (`4 * mlpRatio * hiddenSize = 18432 / 2 = 4608` per half), splits for gate×value, uses GELU(tanh)
- [ ] `forward()` output sliced to `[B, H/8, W/8, 4]` (last 4 channels discarded)
- [ ] `t2i_modulate` defined once and reused across `DiTBlock` and `FinalLayer`
- [ ] Cross-attention in `DiTBlock` receives NO shift/scale/gate from timestep
- [ ] `make build` exits 0

---

### Sortie 3: Weight Key Mapping (P3)

**Priority**: 20.0 — Directly unblocks WU2 Sortie 8 (parallel) and all of WU1 Sorties 4–7 (sequential); defines the interface between PyTorch and MLX key spaces.

**Entry criteria**:
- [ ] Sortie 2 complete (MLX module paths are known from the implemented backbone)

**Tasks** (verify existing file meets spec; fix any discrepancies):
1. Verify `Sources/PixArtBackbone/WeightMapping.swift`: `keyMapping: KeyMapping` closure with ~14 global pairs — patch embedding (`pos_embed.proj.*`), caption projection (`caption_projection.linear_{1,2}.*`), timestep/resolution/AR embedders (`adaln_single.emb.*`), t_block (`adaln_single.linear.*`), final layer (`proj_out.*`, `scale_shift_table`)
2. Verify per-block mappings cover all 28 blocks (loop over index 0–27): `scale_shift_table` direct rename; self-attn `attn1.to_{q,k,v}.*` and `attn1.to_out.0.*`; QK norms `attn1.{q,k}_norm.*`; cross-attn `attn2.to_{q,k,v}.*` and `attn2.to_out.0.*`; FFN `ff.net.0.proj.*` and `ff.net.2.*`
3. Verify `tensorTransform: TensorTransform`: applies `transpose(0, 2, 3, 1)` to Conv2d patch embedding weight only
4. Verify `keyMapping` returns `nil` for discarded keys: `pos_embed` (recomputed at runtime) and `y_embedder.y_embedding` (unused in inference)
5. Verify code comment listing all 224 LoRA-eligible keys (self-attn Q/K/V/out + cross-attn Q/K/V/out × 28 blocks)

**Exit criteria**:
- [ ] `WeightMapping.swift` exists in `Sources/PixArtBackbone/`
- [ ] `keyMapping("adaln_single.linear.weight")` returns a non-nil MLX module path
- [ ] `keyMapping("transformer_blocks.0.attn1.to_q.weight")` returns a non-nil MLX module path
- [ ] `keyMapping("pos_embed")` returns `nil`
- [ ] `keyMapping("y_embedder.y_embedding")` returns `nil`
- [ ] `tensorTransform` for the patch embedding Conv2d weight returns a transposed array
- [ ] Code comment in `WeightMapping.swift` enumerates the 224 LoRA-eligible keys
- [ ] `make build` exits 0

---

### Sortie 4: Pipeline Recipe & Acervo Descriptors (P4 + P5)

**Priority**: 15.0 — Blocks CLI (Sortie 5) and tests (Sortie 6); external API risk from `Acervo.register()`.

**Entry criteria**:
- [ ] Sortie 2 complete (`PixArtDiTConfiguration` defined)
- [ ] SwiftTubería `PipelineRecipe` protocol available in the resolved dependency

**Tasks** (verify existing files meet spec; fix any discrepancies):
1. Verify `Sources/PixArtBackbone/PixArtRecipe.swift`: `PixArtRecipe` conforms to `PipelineRecipe`; type aliases `Encoder = T5XXLEncoder`, `Sched = DPMSolverScheduler`, `Back = PixArtDiT`, `Dec = SDXLVAEDecoder`, `Rend = ImageRenderer`
2. Verify configuration properties: `encoderConfig` (componentId: `"t5-xxl-encoder-int4"`, maxSequenceLength: 120, embeddingDim: 4096); `schedulerConfig` (linear beta schedule betaStart: 0.0001, betaEnd: 0.02, predictionType: .epsilon, solverOrder: 2, trainTimesteps: 1000); `backboneConfig` from `PixArtDiTConfiguration`; `decoderConfig` (componentId: `"sdxl-vae-decoder-fp16"`, latentChannels: 4, scalingFactor: 0.13025); `rendererConfig: Void = ()`
3. Verify recipe properties: `supportsImageToImage: false`, `unconditionalEmbeddingStrategy: .emptyPrompt`, `allComponentIds: ["t5-xxl-encoder-int4", "pixart-sigma-xl-dit-int4", "sdxl-vae-decoder-fp16"]`, `quantizationFor(_:) = .asStored`; default steps: 20, default guidance: 4.5
4. Verify `Sources/PixArtBackbone/PixArtComponents.swift`: `PixArtComponents` enum with `public static let registered: Bool` computed via Swift static initializer; registers 3 `ComponentDescriptor` entries via `Acervo.register()`
5. Verify descriptors: `pixart-sigma-xl-dit-int4` (type `.backbone`, HuggingFace: `intrusive-memory/pixart-sigma-xl-dit-int4-mlx`); `t5-xxl-encoder-int4` (type `.encoder`, `intrusive-memory/t5-xxl-int4-mlx`); `sdxl-vae-decoder-fp16` (type `.decoder`, `intrusive-memory/sdxl-vae-fp16-mlx`)

**Exit criteria**:
- [ ] `PixArtRecipe` conforms to `PipelineRecipe`
- [ ] `PixArtRecipe.encoderConfig.maxSequenceLength` == 120
- [ ] `PixArtRecipe.encoderConfig.embeddingDim` == 4096
- [ ] `PixArtRecipe.schedulerConfig` uses `.linear(betaStart: 0.0001, betaEnd: 0.02)`
- [ ] `PixArtRecipe.decoderConfig.scalingFactor` == 0.13025
- [ ] `PixArtRecipe.allComponentIds` == `["t5-xxl-encoder-int4", "pixart-sigma-xl-dit-int4", "sdxl-vae-decoder-fp16"]`
- [ ] `PixArtComponents.registered` evaluates to `true` without error
- [ ] After `_ = PixArtComponents.registered`: `Acervo.component(id: "pixart-sigma-xl-dit-int4")` returns non-nil
- [ ] After `_ = PixArtComponents.registered`: `Acervo.component(id: "t5-xxl-encoder-int4")` returns non-nil
- [ ] After `_ = PixArtComponents.registered`: `Acervo.component(id: "sdxl-vae-decoder-fp16")` returns non-nil
- [ ] `make build` exits 0

---

### Sortie 5: CLI Tool (P8)

**Priority**: 6.0 — Required by Sortie 7 (integration tests verify CLI help output); new-technology risk from async argument parser.

**Entry criteria**:
- [ ] Sortie 1 complete (`PixArtCLI` target in package)
- [ ] Sortie 4 complete (recipe and component descriptors available)

**Tasks** (verify existing files meet spec; fix any discrepancies):
1. Verify `Sources/PixArtCLI/PixArtCLI.swift`: `@main` struct using `AsyncParsableCommand`; declares `subcommands: [GenerateCommand.self, DownloadCommand.self, InfoCommand.self]`
2. Verify `Sources/PixArtCLI/GenerateCommand.swift`: `--prompt` (required), `--width` (default 1024), `--height` (default 1024), `--output` (default `image.png`), `--steps` (default 20), `--guidance` (default 4.5), `--seed` (optional); calls `_ = PixArtComponents.registered`, assembles `PixArtRecipe`, calls `pipeline.generate()`
3. Verify `Sources/PixArtCLI/DownloadCommand.swift`: fetches all model components via Acervo; reports download progress to stdout
4. Verify `Sources/PixArtCLI/InfoCommand.swift`: prints model configuration, component IDs, estimated sizes, and download status for each component
5. Verify `Sources/PixArtCLI/CLIUtilities.swift`: shared error formatting and progress display helpers used by multiple commands

**Exit criteria**:
- [ ] `make build` exits 0
- [ ] Running the built binary with `--help` prints `generate`, `download`, `info` as listed subcommands
- [ ] `pixart-cli generate --help` output contains: `--prompt`, `--width`, `--height`, `--output`, `--steps`, `--guidance`, `--seed`
- [ ] `pixart-cli download --help` exits 0
- [ ] `pixart-cli info --help` exits 0
- [ ] All 4 CLI source files exist under `Sources/PixArtCLI/`
- [ ] `GenerateCommand.swift` contains `_ = PixArtComponents.registered` before pipeline assembly

---

### Sortie 6: Unit Tests (P9.1 + P9.4)

**Priority**: 5.0 — Only Sortie 7 is blocked by this sortie; test code is low-risk.

**Entry criteria**:
- [ ] Sorties 2–4 complete (backbone, weight mapping, recipe, and descriptors all implemented)

**Tasks** (verify all 8 test files meet spec; fix any gaps):
1. Verify `Tests/PixArtBackboneTests/AttentionTests.swift`: self-attention output shape (synthetic input, no weights); QK norm layers present; cross-attention with separate Q/KV — Q from image tokens, KV from text
2. Verify `Tests/PixArtBackboneTests/DiTBlockTests.swift`: synthetic `[1, 16, 1152]` input + conditioning `[1, 120, 4096]` + timestep modulation tensor → output shape `[1, 16, 1152]`; `scale_shift_table` parameter count is 6×1152
3. Verify `Tests/PixArtBackboneTests/EmbeddingsTests.swift`: 2D sinusoidal embedding for `(H=8, W=8)` → shape `[1, 64, 1152]`; timestep sinusoidal for `B=2` → shape `[2, 256]`; micro-condition embedder output shape verified
4. Verify `Tests/PixArtBackboneTests/PatchEmbeddingTests.swift`: input `[1, 32, 32, 4]` → patch embed → output shape `[1, 256, 1152]` (32/2 × 32/2 = 256 patches)
5. Verify `Tests/PixArtBackboneTests/FinalLayerTests.swift`: input `[1, 256, 1152]` → final layer → unpatchify → output shape `[1, 16, 16, 4]`
6. Verify `Tests/PixArtBackboneTests/WeightMappingTests.swift`: `keyMapping("adaln_single.linear.weight")` non-nil; `keyMapping("transformer_blocks.0.attn1.to_q.weight")` non-nil; `keyMapping("transformer_blocks.27.attn2.to_out.0.weight")` non-nil; `keyMapping("pos_embed")` nil; `keyMapping("y_embedder.y_embedding")` nil; Conv2d transposition produces correct shape
7. Verify `Tests/PixArtBackboneTests/RecipeTests.swift`: `encoderConfig.maxSequenceLength == 120`, `encoderConfig.embeddingDim == 4096`, `decoderConfig.scalingFactor == 0.13025`, `allComponentIds == ["t5-xxl-encoder-int4", "pixart-sigma-xl-dit-int4", "sdxl-vae-decoder-fp16"]`, `supportsImageToImage == false`, `unconditionalEmbeddingStrategy == .emptyPrompt`
8. Verify `Tests/PixArtBackboneTests/ComponentRegistrationTests.swift`: `_ = PixArtComponents.registered`; `Acervo.component(id: "pixart-sigma-xl-dit-int4")` != nil; `Acervo.component(id: "t5-xxl-encoder-int4")` != nil; `Acervo.component(id: "sdxl-vae-decoder-fp16")` != nil

**Exit criteria**:
- [ ] All 8 test files exist in `Tests/PixArtBackboneTests/`
- [ ] `make test` exits 0 (all unit tests pass)
- [ ] Zero uses of `sleep()`, `Task.sleep()`, `Thread.sleep()`, or fixed-duration timeouts in any test file
- [ ] `WeightMappingTests.swift` covers ≥3 global keys, ≥3 per-block keys, ≥2 nil-return keys, ≥1 Conv2d transposition
- [ ] `RecipeTests.swift` asserts: maxSequenceLength==120, betaStart==0.0001, betaEnd==0.02, scalingFactor==0.13025, allComponentIds exact-equal
- [ ] `ComponentRegistrationTests.swift` asserts all 3 component IDs return non-nil
- [ ] `DiTBlockTests.swift` asserts output shape preserves hidden dim (1152)
- [ ] `AttentionTests.swift` asserts QK norm is applied in self-attention path

---

### Sortie 7: Integration Tests & CI Finalization (P9.2 + P9.4)

**Priority**: 4.0 — Terminal sortie for WU1; no downstream dependencies. CI configuration risk (external system).

**Entry criteria**:
- [ ] Sortie 6 complete (all unit tests passing)
- [ ] Sortie 5 complete (CLI tool builds)

**Tasks** (verify existing files meet spec; fix any gaps):
1. Verify `Tests/PixArtBackboneTests/BackboneForwardTests.swift`: instantiates `PixArtDiT` with config; provides synthetic weights via `apply(weights:)`; calls `forward()` with synthetic `BackboneInput` (latents `[1, 128, 128, 4]`, conditioning `[1, 120, 4096]`, conditioningMask `[1, 120]`, timestep `[1]`); asserts output shape is `[1, 128, 128, 4]`
2. Verify `Tests/PixArtBackboneTests/IntegrationTests.swift`: all test methods wrapped in `#if INTEGRATION_TESTS`; seed reproducibility test runs twice with seed 42 and asserts `computePSNR(image1:image2:) > 40.0`; two-phase loading test verifies `pipeline.encodePrompt()` → `pipeline.unloadEncoder()` → `pipeline.denoise()` sequence completes without error; `computePSNR` helper function present in the file
3. Update `.github/workflows/tests.yml`: remove the `build-ios` job (iOS is out of scope); macOS test job must use `runs-on: macos-26` and run `xcodebuild test -scheme pixart-swift-mlx-Package -destination 'platform=macOS,arch=arm64'`; no CI step sets `INTEGRATION_TESTS` compiler flag

**Exit criteria**:
- [ ] `Tests/PixArtBackboneTests/BackboneForwardTests.swift` exists
- [ ] `BackboneForwardTests` passes with output shape `[1, 128, 128, 4]` (1024×1024 input → 1024/8=128 spatial): `make test` exits 0
- [ ] `Tests/PixArtBackboneTests/IntegrationTests.swift` exists with `#if INTEGRATION_TESTS` guard on all test methods
- [ ] `make test` exits 0 (unit tests + BackboneForwardTests; integration tests gated out)
- [ ] `.github/workflows/tests.yml` specifies `runs-on: macos-26`; contains only the macOS test job (no iOS job)
- [ ] No CI step sets `INTEGRATION_TESTS` compiler flag

---

## Work Unit 2: Weight Conversion Scripts

### Sortie 8: Weight Conversion Scripts (P7)

**Priority**: 4.5 — Terminal sortie for WU2; no Swift build dependency. External API risk (HuggingFace + MLX quantization).

**Entry criteria**:
- [ ] Work Unit 1 Sortie 3 complete (key mapping implemented — defines target MLX key names)
- [ ] Python 3.10+ available in the execution environment
- [ ] HuggingFace model access verified: `python -c "from huggingface_hub import HfApi; HfApi().model_info('PixArt-alpha/PixArt-Sigma-XL-2-1024-MS')"` exits 0

**Tasks** (verify existing files meet spec; fix any gaps):
1. Verify `scripts/requirements.txt`: `diffusers` pinned to exact version (currently `diffusers==0.32.2`) to lock key names; `torch`, `transformers`, `safetensors`, `mlx`, `numpy` use minimum-version constraints; `pip install -r scripts/requirements.txt` exits 0
2. Verify `scripts/convert_pixart_weights.py`: loads PixArt-Sigma-XL-2-1024-MS state dict from HuggingFace; applies `keyMapping` renames matching `WeightMapping.swift`; applies Conv2d transposition `[O,I,kH,kW] → [O,kH,kW,I]`; int4 quantizes Linear weights (group_size=64: pack to `uint32`, save `scales` and `biases` in float16); saves MLX safetensors + `config.json` to `--output` path
3. Verify `scripts/convert_t5_weights.py`: loads T5-v1_1-xxl; int4 quantizes Linear weights (group_size=64); keeps `shared.weight` and `relative_attention_bias` in float16; saves MLX safetensors + `config.json`
4. Verify `scripts/convert_vae_weights.py`: loads SDXL VAE; casts all weights to float16 (no quantization — Conv2d layers); saves MLX safetensors + `config.json`
5. Verify `scripts/validate_conversion.py`: runs 5 deterministic prompts at fixed seeds on both PyTorch and MLX; computes per-output PSNR; logs `WARNING: Layer {name} PSNR={value:.1f}dB < 25dB threshold` to stderr for any layer below 25 dB (then continues); fails with non-zero exit code if any end-to-end PSNR < 30 dB

**Exit criteria**:
- [ ] `scripts/requirements.txt` exists; `pip install -r scripts/requirements.txt` exits 0
- [ ] `diffusers` is pinned to an exact version (e.g., `diffusers==0.32.2`) in requirements.txt; grep `diffusers==[0-9]` exits 0
- [ ] `python scripts/convert_pixart_weights.py --output /tmp/pixart-dit-int4` exits 0 and directory contains `config.json` + at least one `.safetensors` file
- [ ] `python scripts/convert_t5_weights.py --output /tmp/t5-xxl-int4` exits 0 and directory contains `config.json` + at least one `.safetensors` file
- [ ] `python scripts/convert_vae_weights.py --output /tmp/sdxl-vae-fp16` exits 0 and directory contains `config.json` + at least one `.safetensors` file
- [ ] `python scripts/validate_conversion.py` reports PSNR > 30 dB for all 5 validation prompts per converted component
- [ ] No unmapped PyTorch key appears in conversion output without an explicit discard entry

---

## Dependency Graph

```
Sortie 1: Package Structure
    │
    ▼
Sortie 2: DiT Backbone
    │
    ▼
Sortie 3: Weight Key Mapping ──────────────────► Sortie 8: Conversion Scripts (WU2, sub-agent)
    │
    ▼
Sortie 4: Recipe & Acervo Descriptors
    │
    ├──────────────────────────────────────────► Sortie 6: Unit Tests (independent of S5)
    ▼
Sortie 5: CLI Tool
    │         │
    └────┬────┘
         │ (both required)
         ▼
Sortie 7: Integration Tests & CI
```

**Critical path** (Work Unit 1): 1 → 2 → 3 → 4 → 5 → 7 (length: 6)

**Parallelism note**: After Sortie 4 completes, Sorties 5 and 6 are independent. Both require `xcodebuild` (supervising agent), so they run sequentially. Optimal order: Sortie 5 first (higher priority: 6.0 vs 5.0).

---

## Parallelism Structure

**Critical Path**: Sortie 1 → 2 → 3 → 4 → 5 → 7 (6 sorties)

**Parallel Execution Groups**:

- **Group 1** (sequential — all depend on previous sortie):
  - WU1 Sortie 1–4 — SUPERVISING AGENT (each has `make build` step)

- **Group 2** (parallel window — opens when Sortie 3 completes):
  - WU1 Sortie 4 → 5 → 6 → 7 — SUPERVISING AGENT (sequential; all require `make build` or `make test`)
  - WU2 Sortie 8 — **SUB-AGENT, NO BUILD** (Python scripts only; no `xcodebuild` or `make` Swift targets)

**Agent Constraints**:
- **Supervising agent**: All sorties with `make build`, `make test`, or `xcodebuild` steps (Sorties 1–7)
- **Sub-agent**: WU2 Sortie 8 only — Python script verification; no Swift build operations

**Maximum parallelism**: 2 agents simultaneously (1 supervising + 1 sub-agent)

---

## Open Questions & Missing Documentation

### Issues Requiring Manual Review Before Execution

| # | Sortie | Type | Description | Recommendation |
|---|--------|------|-------------|----------------|
| 1 | All | **Scope** | All described artifacts already exist in the repository (source files, tests, scripts, CI workflow). The plan was originally written for a greenfield build but the implementation is now complete. | ✓ **Auto-fixed**: All sorties reframed from "Create" to "Verify/Fix". Execution proceeds as a spec-compliance verification pass. |
| 2 | S7 | **CI gap** | `.github/workflows/tests.yml` has a `build-ios` job that only builds (not tests) on iOS. | ✓ **Resolved**: iOS is out of scope (REQUIREMENTS.md P1.3). Remove the `build-ios` job entirely. macOS-only CI. |
| 3 | All | **Build commands** | All exit criteria throughout the original plan used `-scheme PixArtBackbone` or `-scheme PixArtCLI`. The project's correct scheme is `pixart-swift-mlx-Package` (Makefile: `SCHEME = pixart-swift-mlx-Package`). | ✓ **Auto-fixed**: All exit criteria updated to `make build` / `make test` / `make test-python`. |
| 4 | S1 | **Test framework** | Original plan specified "XCTestCase subclass" in test placeholder. Existing implementation uses Swift Testing (`@Suite` / `@Test` / `#expect`). | ✓ **Auto-fixed**: Task description updated to Swift Testing terminology. |
| 5 | S8 | **requirements.txt pinning** | Plan task description said "pin exact versions" for all packages. Existing file uses exact pin only for `diffusers==0.32.2` (the one that needs key-name stability) and `>=` ranges for others. | ✓ **Auto-fixed**: Task description updated to specify diffusers exact pin + minimum-version constraints for others. Added `grep diffusers==` exit criterion to enforce the pin. |

---

## Summary

| Metric | Value |
|--------|-------|
| Work units | 2 |
| Total sorties | 8 |
| Dependency structure | layers |
| Requirements detected | 11 (P1–P11; P11 is reference-only, no sorties) |
| Atomic tasks | 42 |
| Parallel agents | 1 supervising + 1 sub-agent (WU2 Sortie 8) |
| Critical path | 6 sorties (1→2→3→4→5→7) |
| Blocking open questions | 0 |
