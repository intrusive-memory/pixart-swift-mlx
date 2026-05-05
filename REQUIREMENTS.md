# pixart-swift-mlx — Requirements

**Scope**: Active requirements for pixart-swift-mlx, derived from a 3-pass architectural evaluation conducted 2026-05-05. Supersedes the prior architecture-and-API spec, whose substantive content is already implemented in `Sources/PixArtBackbone/`. Open work below.
**Status**: Pass 1 complete (no work). Pass 2 and Pass 3 require implementation.
**Audience**: Maintainers and AI agents.

---

## Pass 1 — SwiftAcervo Metadata-Driven Conversion

**Status: COMPLETE. No action required.**

Audit confirmed `PixArtBackbone` and `PixArtCLI` consume SwiftAcervo by component ID only. No source file constructs paths, calls `appendingPathComponent` against an Acervo model directory, or hardcodes runtime filenames.

Reference points (good patterns, do not change):

- `Sources/PixArtBackbone/PixArtComponents.swift:6-30` — bare `ComponentDescriptor` initializer; manifest is runtime source of truth.
- `Sources/PixArtBackbone/PixArtRecipe.swift`, `PixArtFP16Recipe.swift` — components exposed as Acervo IDs only.
- `Sources/PixArtBackbone/PixArtDiT.swift:189-248` — `apply(weights:)` operates on pre-loaded `MLXArray` tensors, never opens a file.
- `Sources/PixArtCLI/DownloadCommand.swift` — uses `Acervo.ensureComponentReady`; `progress.fileName` is populated by Acervo from manifest.

The `safetensors` / `config.json` strings present in `WeightMapping.swift` and `PixArtDiT.swift` are doc comments describing tensor key suffixes, not filename references. Cosmetic only; out of scope.

---

## Pass 3 — CLI / Downloader Removal

**Status: Architecture pivot complete inside `PixArtBackbone`. User-visible CLI surface still in tree.**

The user has stated the CLI and the model downloader are removed in favor of SwiftAcervo. The library reflects that. Source files, Package.swift entries, Makefile targets, and documentation referencing the CLI do not. Removal is a single mechanical PR.

### R3.1 — Delete `PixArtCLI` source target

Delete the following files:

- `Sources/PixArtCLI/PixArtCLI.swift`
- `Sources/PixArtCLI/GenerateCommand.swift`
- `Sources/PixArtCLI/DownloadCommand.swift`
- `Sources/PixArtCLI/InfoCommand.swift`
- `Sources/PixArtCLI/CLIUtilities.swift`

Remove the now-empty `Sources/PixArtCLI/` directory.

Before deleting `CLIUtilities.swift`, verify no sibling repo imports `runAsync` (`grep -rn runAsync ../*/Sources` is sufficient).

### R3.2 — Strip CLI from `Package.swift`

In `/Users/stovak/Projects/pixart-swift-mlx/Package.swift`:

- Remove `.executable(name: "PixArtCLI", targets: ["PixArtCLI"])` product (lines 42–45).
- Remove `swift-argument-parser` dependency (line 56). No other target uses `ArgumentParser`.
- Remove `.executableTarget(name: "PixArtCLI", …)` target (lines 67–74).

### R3.3 — Strip CLI from `Makefile`

In `/Users/stovak/Projects/pixart-swift-mlx/Makefile`:

- Delete `BINARY = PixArtCLI` and `BIN_DIR = ./bin` (lines 4–5).
- Change default `all: install` → `all: build` (line 10).
- Delete `install` target (lines 20–36).
- Delete `release` target (lines 38–54).
- Drop `$(BIN_DIR)` from `clean` (around line 70–72).
- Strip `install` and `release` from the help text (lines 81–82) and from the `.PHONY` list (line 1).

### R3.4 — Update documentation

- `AGENTS.md:31` — remove `CLI tool (\`PixArtCLI\`)` from deliverables list.
- `AGENTS.md:54-55` — drop `make install` / `make release` from Makefile target list.
- `AGENTS.md:78` — retarget the `ACERVO_APP_GROUP_ID` guidance off "CLI tools" wording (still applies to tests/CI).
- `README.md:15` — strip the CLI feature bullet.
- `README.md:48` — remove `make install   # Build + install PixArtCLI to ./bin`.
- `ARCHITECTURE.md:15` — remove the `swift-argument-parser (CLI only)` row.
- `ARCHITECTURE.md:30` — drop the `PixArtCLI` row from the "What This Package Provides" table.
- `docs/missions/EXECUTION_PLAN.md` lines 55, 58, 66, 158-179, 219, 281, 324 — historical mission doc with stale CLI sortie. Either move to `docs/complete/` or strip CLI sections.
- `docs/incomplete/REQUIREMENTS_ARCHITECTURE_DRAFT.md`, `docs/incomplete/REQUIREMENTS_STANDALONE.md` — explicitly marked draft/superseded but reference `PixArtModelDownloader.swift` and `PixArtCLI`. Confirm with maintainer whether to leave the historical record or strip.

### R3.5 — Tests

No action required. `Tests/PixArtBackboneTests/` has zero references to `PixArtCLI`, `DownloadCommand`, or any custom downloader. CI workflow files (`.github/workflows/build-ios.yml`, `tests.yml`) build/test only the package scheme — no CLI step to remove.

### R3.6 — Out-of-scope but adjacent

- `converted/` directory — already in `.gitignore`. Local conversion output, not committed residue. ~6 GB on disk; maintainer can `rm -rf` to reclaim space. No commit-side action.
- `default.profraw` (5.2 MB) at repo root — stale coverage profile. Add to `.gitignore` and delete. Unrelated to CLI removal but worth fixing in the same PR.
- Version drift — `Sources/PixArtBackbone/PixArtBackbone.swift` says `0.5.3-dev`; `AGENTS.md:5` still says `0.5.2-dev`. Sync.

---

## Pass 2 — Test Suite Improvements

**Status: Suite is weak.** Most tests assert output shapes against zero-tensor inputs; the package's highest-risk paths (int4 dequant in `apply(weights:)`, `PixArtFP16Recipe`, `nearestBucket`, modulation slicing, `unpatchify` axis order) have **zero numerical coverage**. A known scheduler-mode divergence between `PixArtRecipe` and `PixArtFP16Recipe` ships unguarded.

This pass also absorbs three sets of unmet requirements lifted out of the prior spec's §P9: **Backbone Unit Tests**, **Integration Tests**, and the **≥90% line-coverage** target. Re-stated below as concrete, traceable items.

### Absorbed from prior spec §P9

#### R2.A — Backbone unit-test substance (was prior §P9.1)

The prior §P9.1 listed five behaviors that backbone unit tests must verify. All five exist as test files but assert only output *shape* against zero inputs. The numerical content of the spec has not been met.

| Prior §P9.1 item | Current state | Required action |
|---|---|---|
| DiT block forward pass | Shape-only (`DiTBlockTests.swift`) | Numerical test with non-zero `t` — see R2.5 |
| AdaLN-Single modulation: scale/shift/gate | Shape-only | Pin 6-way modulation slicing — see R2.5 |
| Cross-attention: Q from image, K/V from text | Shape-only (`AttentionTests.swift`) | Identity-weight test that sends distinguishable image vs text vectors and asserts which side appears in each role — see R2.6 |
| Patch embedding: spatial → sequence | Shape-only (`PatchEmbeddingTests.swift`) | Per-patch known-value test — see R2.6 |
| Position embedding: 2D sinusoidal | Shape-only (`EmbeddingsTests.swift`) | Reference-vector pin for ordering and denominator — see R2.4 |

#### R2.B — Integration tests must run in CI (was prior §P9.2)

Prior §P9.2 required: full pipeline recipe assembly validation; prompt → CGImage with correct dims and non-zero pixels; seed reproducibility (PSNR > 40 dB between runs on same device + seed); two-phase loading on macOS (iPad deferred). The first item is partially met by `RecipeTests.recipeValidates`. The remaining three live inside `Tests/PixArtBackboneTests/IntegrationTests.swift` gated on `#if INTEGRATION_TESTS`, but the Makefile never sets the flag and no CI job compiles them. The file is dead in CI.

Required action — pick one and execute:

- Add `make test-integration` that compiles with `-D INTEGRATION_TESTS`, plus a CI job that runs it (separate from the default `make test` job so unit-test latency is unaffected).
- Or delete `IntegrationTests.swift` entirely and drop the integration-test goal from the spec.

Current state — present but never compiled — is the worst of both options.

#### R2.C — Coverage measurement (was prior §P9.4 line-coverage target)

Prior §P9.4 required ≥90% line coverage measured per-target and enforced in CI. There is no coverage measurement anywhere: no `-enableCodeCoverage` in the Makefile, no `xccov`/`llvm-cov` step in `.github/workflows/build-ios.yml` or `tests.yml`, no Codecov hookup. The `default.profraw` at the repo root is a stray artifact, not part of a pipeline.

Required action:

- Add `make test-coverage` that invokes `xcodebuild` with `-enableCodeCoverage YES -resultBundlePath Build/test.xcresult`.
- Add a CI step that runs `xccov view --report --json Build/test.xcresult` and fails the job if line coverage on the `PixArtBackbone` target falls below an agreed threshold.
- Decide the threshold once R2.1–R2.13 land — a 90% gate is unrealistic until the matrix below is closed; a 70% interim gate may make sense.
- Add `default.profraw` to `.gitignore` and remove it from the working tree (also tracked under R3.6).

### New requirements from coverage gap audit

The Pass 2 audit produced a per-symbol coverage matrix. The items below capture every symbol marked **Uncovered** or **Tautological**, grouped by required new-or-extended test file.

#### R2.1 — `WeightApplyTests.swift` (new file): `PixArtDiT.apply(weights:)` and `unload()`

Source: `Sources/PixArtBackbone/PixArtDiT.swift:189-253`. Single highest-risk untested function in the package. Build a synthetic small `PixArtDiTConfiguration(hiddenSize: 16, depth: 1, …)` model and exercise:

- **Int4 dequant path** (lines 226-236): supply a `Tuberia.ModuleParameters` containing a uint32-packed `.weight` plus matching `.scales` + `.biases`. Assert post-`apply` dtype is float16 and dequantized values reconstruct the round-trip identity (pack → dequant → assert ε-equal to source float16).
- **Fp16 passthrough path** (lines 237-240): supply a fp16 `.weight` with no scales/biases sidecar. Assert tensor passes through unchanged into params.
- **`.scales`/`.biases` base-key stripping** (lines 207-216): `to_q.scales` → base `to_q`. Pin via test that constructs a parameter dict with the suffix and asserts the base key is what the dequant attaches to.
- **`.bias` vs `.biases` collision**: a key ending in `.bias` (singular) must always pass through the bias slot — assert it is *not* consumed by the scales/biases collection logic.
- **`MLXNN.ModuleParameters.unflattened`**: nested key `blocks.0.attn.to_q.weight` ends up under the right nested module slot.
- **`isLoaded` flag and `currentWeights` after apply**: both set/populated.
- **`unload()` (lines 250-253)**: after `apply` then `unload`, `isLoaded == false`, `currentWeights == nil`, and the model is still callable on its existing module params.

#### R2.2 — `PixArtFP16RecipeTests.swift` (new file)

Source: `Sources/PixArtBackbone/PixArtFP16Recipe.swift`. **Zero tests today.** Mirror `RecipeTests` for the fp16 variant:

- `defaultSteps == 20`, `defaultGuidanceScale == 4.5` (parity with int4).
- `encoderConfig.componentId == "t5-xxl-encoder-int4"`; sequence length 120; embedding dim 4096.
- **`schedulerConfig.betaSchedule`** — pin the divergence: int4 uses `.linear`; fp16 currently uses `.scaledLinear` (line 64). Either is wrong if not intended; this test pins whichever the maintainer chooses and forces a code-doc reconciliation.
- `decoderConfig.componentId == "sdxl-vae-decoder-fp16"`, scaling 0.13025, latent channels 4.
- `supportsImageToImage == false`; `unconditionalEmbeddingStrategy == .emptyPrompt`.
- **`allComponentIds`** must contain `"pixart-sigma-xl-dit-fp16"` and **must NOT contain `-int4`**. Silent regression risk: the entire fp16 diagnostic harness becomes invalid if it pulls int4 weights.
- **`componentIdFor[.backbone] == "pixart-sigma-xl-dit-fp16"`** explicitly.
- `quantizationFor(.backbone) == .asStored` for all five roles.
- `validate()` succeeds for default; throws `PixArtRecipeError.shapeMismatch` for at least one mutated config (caption channels, max text length, latent channels).

#### R2.3 — `BucketSelectionTests.swift` (new file)

Source: `Sources/PixArtBackbone/PixArtDiTConfiguration.swift:161-174`. 64-bucket aspect-ratio lookup, fully untested.

- `aspectRatioBuckets.count == 64`.
- Buckets sorted by aspect ratio (monotonic).
- All buckets near 1024² total pixels (invariant from PixArt-Sigma paper).
- `nearestBucket(width:height:)` cases: 1:1 (1024×1024 → 1024×1024); 16:9 (1920×1080 → 1344×768); 9:16 (1080×1920 → 768×1344); 4:3 (1152×864 → 1152×896); 3:4; ultrawide (2560×1080).
- Boundary case: aspect ratios at midpoints between adjacent buckets — pin which bucket wins.
- Degenerate input: `width == 0` or `height == 0` — pin behavior (graceful or trap).

#### R2.4 — Embeddings numerical correctness

Source: `Sources/PixArtBackbone/Embeddings.swift`. Extend `EmbeddingsTests.swift`:

- **`timestepSinusoidalEmbedding`** (lines 96-127): pin `[cos, sin]` ordering (`flip_sin_to_cos=True`) and the `denominator = halfDim` (not `halfDim - 1`) divisor against a known diffusers reference. Both are documented footguns; both are unpinned.
- **`get2DSinusoidalPositionEmbeddings`** (lines 53-55): pin W-first, H-second concatenation order against a fixed-input reference vector.
- **`sinusoidalEmbedding1D`**: pin `[sin, cos]` ordering. `pos = 0` → first half all zero, second half all one.
- **`SizeEmbedder` shared MLP**: assert that h and w go through the *same* embedder instance, not two distinct linears. Asymmetric input `[B, [10, 1000]]` should produce halves where the first 384 dims encode 10 and the second 384 dims encode 1000. A direct test that the linear's parameters are shared closes this.
- **`SelfAttention.scale`**: assert `scale == 1.0 / sqrt(headDim)` (Attention.swift). Shape tests cannot catch a wrong scale.

#### R2.5 — Modulation slicing in `DiTBlock` and `FinalLayer`

Source: `Sources/PixArtBackbone/DiTBlock.swift:93-98`, `Sources/PixArtBackbone/FinalLayer.swift`. Both currently use `t = 0` and `scaleShiftTable = 0`, which makes any swap of the 6-way (DiTBlock) or 2-way (FinalLayer) modulation parameters invisible. Replace with non-zero `t` of structured form (e.g. `t = MLXArray([1, 2, 3, 4, 5, 6])` after broadcast) and assert each named modulation slot — `shiftMsa`, `scaleMsa`, `gateMsa`, `shiftMlp`, `scaleMlp`, `gateMlp` — corresponds to the correct index. A swapped index in source must fail this test.

Direct unit tests for helpers:

- `t2iModulate(x, shift, scale)` — with `x=1, shift=2, scale=3` → expect `1 * (1+3) + 2 == 6`. Trivial direct test instead of the indirect path through `FinalLayer`.
- `GEGLUFFN.callAsFunction` — verify the constants `0.7978845608` and `0.044715` produce gelu_new output ε-matching `MLXNN.geluApproximate` on a random input.

#### R2.6 — `FinalLayer.unpatchify` axis order

Source: `Sources/PixArtBackbone/FinalLayer.swift`. The transpose order `(0, 1, 3, 2, 4, 5)` is load-bearing. Test: build a synthetic patch grid where patch `(i, j)` is filled with `i * gridW + j`. Run through `unpatchify`. Assert the unpatched grid pixel at `(i*P + r, j*P + c)` equals `i * gridW + j` for all `r, c < P`. A swapped axis silently corrupts spatial output; this test catches it.

Cross-attention identity test (R2.A item):

- `CrossAttention` — set Q-projection to identity, K and V to identity. Pass distinguishable image and text vectors. Assert image vector dominates the Q axis and text vector dominates K and V; mask=all-ones output ε-equal to mask=nil output.

Patch embedding spatial→sequence test (R2.A item):

- Build input where pixel at `(i, j)` is `i * W + j`. Assert the resulting sequence token at index `t = i_patch * (W/P) + j_patch` carries the expected pooled identity of patch `(i_patch, j_patch)`.

#### R2.7 — `PixArtRecipe.componentIdFor` regression pin

Source: `Sources/PixArtBackbone/PixArtRecipe.swift:136-142`. This override exists *because of* a historical bug fixed in PR #10 (commit 32ce4c3). It is currently untested. Pin it directly: `recipe.componentIdFor(.encoder) == "t5-xxl-encoder-int4"`, `.backbone → "pixart-sigma-xl-dit-int4"`, `.decoder → "sdxl-vae-decoder-fp16"`. Without this test, a future refactor that drops the override silently regresses the bug.

Also pin `validate()` throw paths: mutate `captionChannels`, `maxTextLength`, `latentChannels` one at a time and assert `validate()` throws `PixArtRecipeError.shapeMismatch`.

Pin remaining scheduler config fields not currently asserted: `predictionType == .epsilon`, `solverOrder == 2`, `trainTimesteps == 1000`.

#### R2.8 — Component descriptor metadata

Source: `Sources/PixArtBackbone/PixArtComponents.swift`. Extend `ComponentRegistrationTests.swift`:

- **FP16 descriptor**: assert `Acervo.descriptor(for: "pixart-sigma-xl-dit-fp16") != nil` after registration. Currently silently untested.
- `repoId == "intrusive-memory/pixart-sigma-xl-dit-int4-mlx"` (and the fp16 equivalent).
- `minimumMemoryBytes` thresholds (800 MB int4, 2.5 GB fp16) — these get passed to download infra; a silent regression breaks model loads.
- `metadata["component_role"] == "backbone"`, `metadata["quantization"]` ∈ {"int4", "fp16"}, `metadata["architecture"] == "DiT-XL"`.
- **Idempotency**: `_ = PixArtComponents.registered` called twice must not throw or duplicate-register.

#### R2.9 — Forward-pass micro-conditioning regression pin

Source: `Sources/PixArtBackbone/PixArtDiT.swift` `forward` (around lines 152-200). The current implementation deliberately skips micro-conditioning (`tBlock` is built from raw timestep without size/AR embeddings) because the int4 weight checkpoint was trained with `adaln_single.linear.weight = [6912, 1152]` — no micro-conditioning tower. Pin this:

- Assert that two forward passes at the same timestep but different `(width, height)` inputs produce *identical* outputs (modulo position embeddings) — proving size/AR embeddings are not folded into `tBlock`.
- Assert `tRaw` (passed to `FinalLayer`) is independent of `tBlock` (passed to per-block AdaLN). A swap silently corrupts output.

Without this pin, a future engineer who "fixes" what looks like an obvious omission breaks every existing int4 weight load.

### Cleanup of existing tests

#### R2.10 — Collapse redundant shape tests

`BackboneForwardTests.swift:31-193` — six near-identical tests (`outputShapeBatch1`, `outputShapeContractVerified`, `outputHasFourChannels`, `outputNdimIs4`, `spatialDimensionsPreserved`, `outputLatentChannelsMatchesForward`) all rebuild the full 28-block model and assert variants of the same output-shape contract. Collapse into one parameterized case. Net effect: ~5 fewer cold inits of the 600M-param model per CI run.

`ComponentRegistrationTests.swift:33-61` overlaps with `RecipeTests` (`componentIds`, `encoderComponentId`, `decoderComponentId`). Within `RecipeTests` itself: `componentIds` / `componentIdsExactOrder`, `encoderEmbeddingDim` / `encoderBackboneContract`, `encoderMaxSequenceLength` / `sequenceLengthContract` re-assert the same constants. Keep one canonical assertion per fact; delete duplicates.

#### R2.11 — Delete tautological and misleading tests

| File:line | Test | Action | Reason |
|---|---|---|---|
| `AttentionTests.swift:39-51` | `selfAttentionHasQKNorm` | **Delete or rename** | Source `Attention.swift:49-51` deliberately removes QK norm. Test name is a lie. |
| `AttentionTests.swift:53-63` | `selfAttentionNdim` | Delete | Pure tautology. |
| `AttentionTests.swift:126-137` | `crossAttentionNdim` | Delete | Tautology. |
| `DiTBlockTests.swift:65-79` | `outputNdim` | Delete | Tautology, covered by `outputShapeMatchesInput`. |
| `DiTBlockTests.swift:121-144` | `adaLNModulationShape` | Rewrite | Comment claims "any other shape would crash"; with `t=0` test only verifies output shape, not that 6 modulation params are sliced in the correct order. Replace with non-zero `t` (covered by R2.5). |
| `EmbeddingsTests.swift:47-58` | `positionEmbeddingNdim` | Delete | Tautology. |
| `EmbeddingsTests.swift:153-168` | `captionProjectionPreservesDims` | Delete | Duplicates `captionProjectionShape` (139-152). |
| `FinalLayerTests.swift:31-42` | `outputNdim` | Delete | Tautology. |
| `FinalLayerTests.swift:44-68` | `varianceChannels` | Rewrite | Asserts `dim(3) == outChannels` — only tests Swift slicing, not FinalLayer. Promote to verify forward()'s slice when `outChannels=8`. |
| `BackboneForwardTests.swift:111-126` | `outputNdimIs4` | Delete | Tautology (and covered by R2.10's collapse). |
| `ComponentRegistrationTests.swift:44-49` | `ditComponentIdMatchesRecipe` | Delete | Duplicates `allThreeComponentIds`. |
| `PatchEmbeddingTests.swift:116-135` | `patchEmbeddingDefaultConfig` | Delete | Builds 600M-param `PixArtDiT` to read four `Int` constants from a struct; never uses the model. |

#### R2.12 — Re-target `WeightMappingTests.swift`

`WeightMapping.keyMapping` is `{ key in key }` (`WeightMapping.swift:38-39`). Every test calls `mapping(x) == x`. Delete identity-passthrough cases. Replace with two pins:

- `tensorTransform == nil` (already covered by `WeightMappingTests:92-99`; keep).
- `noKeysDiscarded` (line 81-90) — keep but rename for clarity; the contract being pinned is "filtering is not the mapping's job," and the bogus-key example obscures intent.

Real key-transform coverage moves into `WeightApplyTests.swift` (R2.1) since runtime mapping is identity by design.

#### R2.13 — Share fixtures across full-model tests

Each test in `BackboneForwardTests` and `WeightMappingTests` reinitializes a full 28-block `PixArtDiT` (~13 cold inits per CI run). Use a shared lazy fixture or a class-level `setUp` to amortize. Performance hygiene only; do this after R2.10 collapses the redundant shape tests.

---

## Out of scope

- Changes to SwiftAcervo itself.
- Python weight-conversion scripts (`scripts/`). Conversion-time mapping (key remapping, Conv2d transposition) lives in those scripts and is validated by `scripts/validate_conversion.py` / `scripts/test_conversion.py` — not a Swift-test concern.
- `.github/workflows/` structural changes beyond adding a coverage step (R2.C) and an integration-test job (R2.B).

## Suggested ordering

1. **Pass 3 cleanup** (R3.1 → R3.6). Mechanical, low-risk, single PR.
2. **R2.7 + R2.2 + R2.8** — the three regression-pin items that cost almost nothing and protect against silent re-breakage of historical bugs (componentIdFor, FP16 recipe drift, FP16 descriptor registration).
3. **R2.1** — close the int4 dequant gap. Highest-risk untested path.
4. **R2.5 + R2.6 + R2.9** — modulation slicing, unpatchify axis order, micro-conditioning skip pin.
5. **R2.4 + R2.3** — embedding numerical correctness and bucket selection.
6. **R2.B** — wire `make test-integration` and CI job (or delete the dead file).
7. **R2.10 + R2.11 + R2.12** — cleanup of redundant/misleading tests.
8. **R2.13** — shared fixture (performance hygiene).
9. **R2.C** — coverage measurement and CI gate (do this last, with an interim threshold informed by what the prior steps achieve).

## Cross-references

- **Tests to write**: R2.1, R2.2, R2.3, R2.4, R2.5, R2.6, R2.7, R2.8, R2.9 → ~5 new test files (`WeightApplyTests.swift`, `PixArtFP16RecipeTests.swift`, `BucketSelectionTests.swift`) plus extensions to 5 existing files.
- **Tests to delete or rewrite**: R2.10, R2.11, R2.12 → ~12 cases deleted, ~3 rewritten.
- **Build/CI work**: R2.B (integration job), R2.C (coverage gate), R2.13 (fixture).
