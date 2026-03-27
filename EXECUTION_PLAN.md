---
feature_name: "pixart-swift-mlx — PixArt Model Plugin"
iteration: 1
wave: 3
repository: pixart-swift-mlx
status: refined
depends_on: ["SwiftAcervo v2", "SwiftTubería Waves 1-2"]
---

# Execution Plan: pixart-swift-mlx

> **Terminology**: A *mission* is the definable scope of work — the whole campaign. A *sortie* is an atomic, testable unit of work executed by a single autonomous AI agent in one dispatch. A sortie has a defined objective, machine-verifiable entry/exit criteria, and bounded scope (fits within a single agent context window). We avoid agile/waterfall terminology (sprint, iteration, phase) because those map to *time*. Missions and sorties map to *agentic work cycles*, which have no inherent time dimension.

**Goal**: Deliver the PixArt-Sigma DiT backbone as a SwiftTubería model plugin. This package provides ~400 lines of model-specific code — the DiT transformer, weight key mapping, pipeline recipe, Acervo descriptors, and a CLI tool. All shared infrastructure (weight loading, scheduling, VAE decoding, image rendering, memory management) comes from SwiftTubería.

**Upstream blockers**: SwiftTubería must ship its `Backbone`, `WeightedSegment`, `PipelineRecipe`, and catalog components (T5XXLEncoder, SDXLVAEDecoder, DPMSolverScheduler, ImageRenderer) before Sorties 2-5 can compile against real protocols. Sortie 1 (package structure) can proceed with placeholder targets once SwiftTubería's package URL is known.

---

## Sortie 0: Reconnaissance (COMPLETED)

### Objective
Map the current codebase state, understand the PyTorch reference implementation, and identify all architectural details needed for implementation.

### Current Codebase State
- **Package.swift**: Exists but defines `PixArtMLX` library with only `SwiftAcervo` dependency. Needs restructuring per REQUIREMENTS.md (rename to `PixArtBackbone`, add `PixArtCLI` executable, depend on SwiftTubería instead of SwiftAcervo directly).
- **Sources/PixArtMLX/PixArtMLX.swift**: Single placeholder file with one comment line. No implementation.
- **Tests/PixArtMLXTests/PixArtMLXTests.swift**: Single placeholder test. No real tests.
- **docs/incomplete/ARCHITECTURE_STANDALONE.md**: Comprehensive implementation reference covering all tensor shapes, weight key mappings (PixArt DiT, T5-XXL, SDXL VAE), MLX idioms, scheduler math, and quantization format. Still valid as the primary implementation guide even though the "standalone" library approach was superseded by the plugin architecture.
- **docs/incomplete/REQUIREMENTS_STANDALONE.md**: Superseded by top-level REQUIREMENTS.md. Retained for historical context.
- **No scripts/ directory**: Weight conversion scripts (P7) must be created from scratch.
- **CI**: GitHub Actions workflow exists at `.github/workflows/tests.yml` — builds and tests on macOS 26 and iOS Simulator (iPhone 17, OS 26.1). Currently runs against the placeholder test.

### Key Reference Documents
- `/Users/stovak/Projects/pixart-swift-mlx/REQUIREMENTS.md` — P1-P11 specification (source of truth)
- `/Users/stovak/Projects/pixart-swift-mlx/ARCHITECTURE.md` — Ecosystem interface reference (protocol contracts, recipe, data flow)
- `/Users/stovak/Projects/pixart-swift-mlx/docs/incomplete/ARCHITECTURE_STANDALONE.md` — Internal architecture detail (tensor shapes, weight mappings, MLX idioms, scheduler math)
- PixArt-Sigma paper: arXiv:2403.04692
- PyTorch reference: `PixArt-alpha/PixArt-sigma` on GitHub
- HuggingFace diffusers: `PixArtSigmaPipeline`

### Architecture Summary (from reconnaissance)
- **28 DiT blocks**, each: Self-Attention -> Cross-Attention -> FFN
- **AdaLN-Single**: One global `t_block` MLP + per-block learned `scale_shift_table` (6 x 1152)
- **Hidden dim 1152**, 16 heads, head dim 72, patch size 2
- **8-channel output** (4 noise + 4 variance; variance discarded at inference)
- **2D sinusoidal position embeddings** recomputed per forward pass (variable resolution)
- **Micro-conditioning**: resolution + aspect ratio embeddings concatenated with timestep
- **Cross-attention receives NO AdaLN modulation**
- **QK normalization enabled** in self-attention
- **Caption projection**: Linear(4096, 1152) -> GELU(tanh) -> Linear(1152, 1152) before blocks
- **Conv2d weights**: PyTorch [O,I,kH,kW] -> MLX [O,kH,kW,I] transposition required

### Exit Criteria
- [x] All architecture docs read and understood
- [x] Current codebase state documented
- [x] Gap analysis complete (placeholder code vs. REQUIREMENTS.md)
- [x] Execution plan written

### Notes
The `docs/incomplete/ARCHITECTURE_STANDALONE.md` errata section (A7) documents five corrections to the standalone REQUIREMENTS that are already reflected in the top-level REQUIREMENTS.md. Confirm these during implementation: linear beta schedule (not shifted cosine), max 120 tokens (not 512), head dim 72 (not 64), 8-channel output, GELU(tanh) activation.

---

## Sortie 1: Package Structure (P1)

**Priority**: 17.0 — Foundation sortie; all subsequent sorties depend on this. Blocks 8 downstream sorties transitively.

### Objective
Restructure Package.swift to define the correct products, targets, and dependencies per REQUIREMENTS.md P1. Rename the library from `PixArtMLX` to `PixArtBackbone`, add `PixArtCLI` executable target, and switch the primary dependency from SwiftAcervo to SwiftTubería.

### Entry Criteria
- Sortie 0 complete (reconnaissance documented)
- SwiftTubería package URL known (GitHub repo under intrusive-memory org)

### Files
| File | Action | Description |
|------|--------|-------------|
| `Package.swift` | **Rewrite** | Two products: `PixArtBackbone` library, `PixArtCLI` executable. Dependencies: SwiftTubería (from: "0.1.0"), swift-argument-parser (from: "1.3.0"). Platforms: `.macOS(.v26), .iOS(.v26)`. Swift language mode: `.v6`. |
| `Sources/PixArtMLX/` | **Rename** to `Sources/PixArtBackbone/` | Source target directory rename. |
| `Sources/PixArtCLI/` | **Create** | New directory with `PixArtCLI.swift` entry point (placeholder `@main` struct). |
| `Tests/PixArtMLXTests/` | **Rename** to `Tests/PixArtBackboneTests/` | Test target directory rename. |
| `.github/workflows/tests.yml` | **Update** | Scheme name changes from `pixart-swift-mlx` to match new package structure. Verify xcodebuild discovers the right scheme. |

### Dependencies
- SwiftTubería package URL must be known (GitHub repo under intrusive-memory org).
- SwiftAcervo becomes a transitive dependency via SwiftTubería — no longer listed directly.

### Target Layout
```swift
products: [
    .library(name: "PixArtBackbone", targets: ["PixArtBackbone"]),
    .executable(name: "PixArtCLI", targets: ["PixArtCLI"]),
]
targets: [
    .target(name: "PixArtBackbone",
            dependencies: [.product(name: "Tubería", package: "SwiftTubería"),
                           .product(name: "TuberíaCatalog", package: "SwiftTubería")]),
    .executableTarget(name: "PixArtCLI",
                      dependencies: ["PixArtBackbone",
                                     .product(name: "ArgumentParser", package: "swift-argument-parser")]),
    .testTarget(name: "PixArtBackboneTests",
                dependencies: ["PixArtBackbone"]),
]
```

### Exit Criteria
- [ ] `xcodebuild build -scheme PixArtBackbone -destination 'platform=macOS'` succeeds (with placeholder source)
- [ ] `xcodebuild build -scheme PixArtCLI -destination 'platform=macOS'` succeeds (with placeholder `@main`)
- [ ] `xcodebuild test -scheme PixArtBackbone -destination 'platform=macOS'` runs placeholder test
- [ ] CI workflow updated: `.github/workflows/tests.yml` references correct scheme names
- [ ] `Sources/PixArtBackbone/` directory exists and contains at least one `.swift` file
- [ ] `Sources/PixArtCLI/PixArtCLI.swift` exists with `@main` struct
- [ ] `Tests/PixArtBackboneTests/` directory exists with at least one test file

### Notes
- The product name in REQUIREMENTS.md is `PixArtBackbone` (not `PixArtMLX` or `PixArtCore`). SwiftVinetas will `import PixArtBackbone`.
- SwiftTubería may not yet exist at time of writing — if so, use a branch dependency until a tagged release is available.
- The `PixArtCLI` executable target uses `.executableTarget` (not `.target`), and requires `@main` attribute on the entry point struct.

---

## Sortie 2: PixArt DiT Backbone (P2)

**Priority**: 15.5 — Core implementation; blocks weight mapping, recipe, descriptors, CLI, and tests. Highest-complexity sortie. Assign to opus.

### Objective
Implement the PixArt-Sigma DiT transformer backbone conforming to SwiftTubería's `Backbone` and `WeightedSegment` protocols. This is the single substantial piece of new code (~400 lines).

### Entry Criteria
- Sortie 1 complete (`PixArtBackbone` target compiles with placeholder source)
- SwiftTubería `Backbone` and `WeightedSegment` protocols available (from SwiftTubería dependency)

### Files
| File | Action | Description |
|------|--------|-------------|
| `Sources/PixArtBackbone/PixArtDiTConfiguration.swift` | **Create** | Configuration struct: hiddenSize (1152), numHeads (16), headDim (72), depth (28), patchSize (2), inChannels (4), outChannels (8), mlpRatio (4.0), captionChannels (4096), maxTextLength (120). Aspect ratio bucket table. |
| `Sources/PixArtBackbone/PixArtDiT.swift` | **Create** | Top-level backbone: patch embedding, caption projection, timestep conditioning pipeline (sinusoidal + MLP + micro-conditions), t_block, 28 DiT blocks, final layer, unpatchify. Conforms to `Backbone` and `WeightedSegment`. |
| `Sources/PixArtBackbone/DiTBlock.swift` | **Create** | Single DiT block: norm1, self-attention, cross-attention, norm2, FFN, scale_shift_table. AdaLN-Zero modulation. |
| `Sources/PixArtBackbone/Attention.swift` | **Create** | Self-attention (with QK norm) and cross-attention modules. Multi-head reshape, `MLXFast.scaledDotProductAttention`. |
| `Sources/PixArtBackbone/Embeddings.swift` | **Create** | 2D sinusoidal position embeddings (dynamic recomputation), timestep sinusoidal embedding, micro-condition embedders (resolution, aspect ratio). |
| `Sources/PixArtBackbone/FinalLayer.swift` | **Create** | Final AdaLN (2-param: shift+scale, no gate), linear projection, unpatchify to spatial. |

### Dependencies
- SwiftTubería `Backbone` protocol (inlet: `BackboneInput`, outlet: `MLXArray`)
- SwiftTubería `WeightedSegment` protocol (keyMapping, tensorTransform, estimatedMemoryBytes)
- MLX via SwiftTubería (transitive: mlx-swift)

### Shape Contracts
```
inlet:  BackboneInput {
            latents:          MLXArray [B, H/8, W/8, 4]
            conditioning:     MLXArray [B, 120, 4096]
            conditioningMask: MLXArray [B, 120]
            timestep:         MLXArray [B]
        }
outlet: MLXArray [B, H/8, W/8, 4]  (variance channels discarded)
```

Shape contract properties:
- `expectedConditioningDim: 4096`
- `outputLatentChannels: 4`
- `expectedMaxSequenceLength: 120`

### Implementation Details (from ARCHITECTURE_STANDALONE.md)
1. **Patch embedding**: Conv2d(4, 1152, kernel=2, stride=2) -> flatten to [B, T, 1152]. Add 2D sinusoidal pos embeddings.
2. **Caption projection**: Linear(4096, 1152) -> GELU(tanh) -> Linear(1152, 1152). Applied once before blocks.
3. **Timestep pipeline**: sinusoidal(256) -> MLP(256->1152) -> add micro-conditions(resolution 2x384 + AR 384 = 1152) -> t_block: SiLU -> Linear(1152, 6*1152) = [B, 6912].
4. **Per block**: Unpack 6 modulation params from (scale_shift_table + t). Self-attn with AdaLN. Cross-attn (no modulation). FFN with AdaLN.
5. **Final layer**: AdaLN(2-param) + Linear(1152, 32) + unpatchify to [B, H/8, W/8, 8]. Discard last 4 channels.
6. **All modules**: `@unchecked Sendable`, `@ModuleInfo` on quantizable Linear layers.

### Exit Criteria
- [ ] `PixArtDiT` type exists and conforms to `Backbone` protocol
- [ ] `PixArtDiT` type conforms to `WeightedSegment` protocol
- [ ] `xcodebuild build -scheme PixArtBackbone -destination 'platform=macOS'` succeeds with all 6 source files
- [ ] `PixArtDiTConfiguration` struct has all required properties: hiddenSize=1152, numHeads=16, headDim=72, depth=28, patchSize=2, inChannels=4, outChannels=8, mlpRatio=4.0, captionChannels=4096, maxTextLength=120
- [ ] `expectedConditioningDim` returns 4096, `outputLatentChannels` returns 4, `expectedMaxSequenceLength` returns 120
- [ ] `DiTBlock` count is exactly 28 (verifiable via `blocks.count == 28` in init)
- [ ] `t2i_modulate` helper function defined and used by both `DiTBlock` and `FinalLayer`
- [ ] GEGLU FFN implementation: fc1 projects to 2*4608, splits for gate*value, uses GELU(tanh)

### Notes
- The `t2i_modulate(x, shift, scale) = x * (1 + scale) + shift` function appears in multiple places — define once and reuse.
- Cross-attention receives NO timestep modulation (no shift/scale/gate). This is a key difference from self-attention.
- The FFN uses GELU(tanh) approximation, specifically GEGLU gating: fc1 projects to 2*4608 and splits for gate*value.
- QK normalization is enabled in self-attention (LayerNorm on q and k after reshape).
- `pe_interpolation = 2` for PixArt-Sigma XL.

---

## Sortie 3: Weight Key Mapping (P3)

**Priority**: 13.0 — Blocks conversion scripts and LoRA verification. Foundation for weight loading.

### Objective
Implement the ~200 key-pair mapping from HuggingFace diffusers format to MLX module paths, plus Conv2d weight transposition.

### Entry Criteria
- Sortie 2 complete (MLX module paths are known from the implemented backbone)
- SwiftTubería `KeyMapping` and `TensorTransform` types available

### Files
| File | Action | Description |
|------|--------|-------------|
| `Sources/PixArtBackbone/WeightMapping.swift` | **Create** | `keyMapping: KeyMapping` closure implementation. ~14 global mappings + 28 blocks x ~7 per-block mapping groups. Conv2d tensor transform. |

### Dependencies
- SwiftTubería `KeyMapping` type (closure: `(String) -> String?`)
- SwiftTubería `TensorTransform` type (closure: `(String, MLXArray) -> MLXArray`)
- Sortie 2 complete (module paths must match the actual MLX module structure)

### Key Mapping Categories (from ARCHITECTURE_STANDALONE.md A6.1)

**Global keys (~14 pairs)**:
- Patch embedding: `pos_embed.proj.{weight,bias}`
- Caption projection: `caption_projection.linear_{1,2}.{weight,bias}`
- Timestep embedder: `adaln_single.emb.timestep_embedder.linear_{1,2}.{weight,bias}`
- Resolution embedder: `adaln_single.emb.resolution_embedder.linear_{1,2}.{weight,bias}`
- Aspect ratio embedder: `adaln_single.emb.aspect_ratio_embedder.linear_{1,2}.{weight,bias}`
- t_block: `adaln_single.linear.{weight,bias}`
- Final layer: `proj_out.{weight,bias}`, `scale_shift_table`
- Discarded keys: `pos_embed` (recomputed), `y_embedder.y_embedding`

**Per-block keys (28 blocks, ~7 groups each)**:
- `scale_shift_table`: direct rename
- Self-attention Q/K/V: diffusers stores separate `attn1.to_{q,k,v}` — map to module paths
- Self-attention output: `attn1.to_out.0` -> output projection
- QK norms: `attn1.q_norm`, `attn1.k_norm`
- Cross-attention Q: `attn2.to_q`
- Cross-attention K/V: `attn2.to_{k,v}`
- Cross-attention output: `attn2.to_out.0`
- FFN: `ff.net.0.proj` (GEGLU), `ff.net.2` (output)

**Tensor transform**: Conv2d weights from `[O, I, kH, kW]` -> `[O, kH, kW, I]` via `transpose(0, 2, 3, 1)`. Applied to patch embedding conv weight.

### Exit Criteria
- [ ] `WeightMapping.swift` file exists in `Sources/PixArtBackbone/`
- [ ] `keyMapping` closure handles all ~14 global key pairs (patch embed, caption projection, timestep/resolution/AR embedders, t_block, final layer)
- [ ] `keyMapping` closure handles all 28 blocks with ~7 mapping groups each (scale_shift_table, self-attn Q/K/V/out, QK norms, cross-attn Q/K/V/out, FFN)
- [ ] `tensorTransform` applies `transpose(0, 2, 3, 1)` to Conv2d patch embedding weight key
- [ ] Discarded keys (`pos_embed`, `y_embedder.y_embedding`) return `nil` from `keyMapping` (silently skipped)
- [ ] `xcodebuild build -scheme PixArtBackbone -destination 'platform=macOS'` succeeds
- [ ] Unit test: `keyMapping("adaln_single.linear.weight")` returns expected MLX module path
- [ ] Unit test: `keyMapping("transformer_blocks.0.attn1.to_q.weight")` returns expected MLX module path
- [ ] Unit test: `keyMapping("pos_embed")` returns `nil`

### Notes
- The diffusers format stores self-attention as separate Q/K/V projections. The original PixArt format fuses them as `qkv`. Since we load from diffusers format (HuggingFace), we map the separate projections.
- Decide in Sortie 2 whether the backbone uses fused QKV internally (requires concatenation in mapping) or separate Q/K/V (simpler 1:1 renames). Adjust mapping accordingly.
- Cross-attention K/V are also stored separately in diffusers format (original fuses them as `kv_linear`).

---

## Sortie 4: Pipeline Recipe & Acervo Descriptors (P4 + P5)

**Priority**: 11.5 — Blocks CLI tool and integration tests. Recipe + descriptors are tightly coupled (recipe references component IDs that descriptors register).

### Objective
Implement `PixArtRecipe` conforming to SwiftTubería's `PipelineRecipe` protocol, and register all Acervo component descriptors. These two concerns are merged because the recipe's `allComponentIds` and the descriptor registrations share the same component ID strings and must be consistent.

### Entry Criteria
- Sortie 2 complete (`PixArtDiTConfiguration` defined)
- SwiftTubería `PipelineRecipe` protocol available
- SwiftAcervo `ComponentDescriptor` and `Acervo.register()` API available (transitively)

### Files
| File | Action | Description |
|------|--------|-------------|
| `Sources/PixArtBackbone/PixArtRecipe.swift` | **Create** | `PipelineRecipe` conformance. Type aliases for encoder/scheduler/backbone/decoder/renderer. Configuration values for each. Memory profiles. Default generation parameters. |
| `Sources/PixArtBackbone/PixArtComponents.swift` | **Create** | `PixArtComponents` enum with static `registered` property. Registers 3 component descriptors. Thread-safe one-time initialization via Swift static `let`. |

### Dependencies
- SwiftTubería `PipelineRecipe` protocol
- SwiftTubería catalog configuration types: `T5XXLEncoderConfiguration`, `DPMSolverSchedulerConfiguration`, `SDXLVAEDecoderConfiguration`
- SwiftAcervo `ComponentDescriptor` and `Acervo.register()` API

### Configuration Values (Recipe)
| Config Property | Key Values |
|-----------------|------------|
| `encoderConfig` | componentId: `"t5-xxl-encoder-int4"`, maxSequenceLength: 120, embeddingDim: 4096 |
| `schedulerConfig` | betaSchedule: `.linear(betaStart: 0.0001, betaEnd: 0.02)`, predictionType: `.epsilon`, solverOrder: 2, trainTimesteps: 1000 |
| `backboneConfig` | hiddenSize: 1152, numHeads: 16, depth: 28, patchSize: 2, captionChannels: 4096 |
| `decoderConfig` | componentId: `"sdxl-vae-decoder-fp16"`, latentChannels: 4, scalingFactor: 0.13025 |
| `rendererConfig` | `Void` |

### Additional Properties (Recipe)
| Property | Value |
|----------|-------|
| `supportsImageToImage` | `false` |
| `unconditionalEmbeddingStrategy` | `.emptyPrompt` |
| `allComponentIds` | `["t5-xxl-encoder-int4", "pixart-sigma-xl-dit-int4", "sdxl-vae-decoder-fp16"]` |
| `quantizationFor(.backbone)` | `.asStored` |
| Default steps | 20 |
| Default guidance | 4.5 |

### Component Descriptors (Acervo)
| Component | Acervo ID | Type | HuggingFace Repo | Size (int4) |
|-----------|-----------|------|-------------------|-------------|
| PixArt-Sigma XL DiT | `pixart-sigma-xl-dit-int4` | `.backbone` | `intrusive-memory/pixart-sigma-xl-dit-int4-mlx` | ~300 MB |
| T5-XXL | `t5-xxl-encoder-int4` | `.encoder` | `intrusive-memory/t5-xxl-int4-mlx` | ~1.2 GB |
| SDXL VAE | `sdxl-vae-decoder-fp16` | `.decoder` | `intrusive-memory/sdxl-vae-fp16-mlx` | ~160 MB |

### Registration Pattern
```swift
public enum PixArtComponents {
    public static let registered: Bool = {
        Acervo.register([...])
        return true
    }()
}
```
Pipeline assembly triggers: `_ = PixArtComponents.registered`

### Exit Criteria
- [ ] `PixArtRecipe` type exists and conforms to `PipelineRecipe`
- [ ] `PixArtRecipe.encoderConfig.maxSequenceLength` == 120
- [ ] `PixArtRecipe.schedulerConfig.betaSchedule` is `.linear(betaStart: 0.0001, betaEnd: 0.02)`
- [ ] `PixArtRecipe.decoderConfig.scalingFactor` == 0.13025
- [ ] `PixArtRecipe.allComponentIds` contains exactly `["t5-xxl-encoder-int4", "pixart-sigma-xl-dit-int4", "sdxl-vae-decoder-fp16"]`
- [ ] `PixArtComponents.registered` evaluates to `true` without error
- [ ] After `_ = PixArtComponents.registered`, `Acervo.component(id: "pixart-sigma-xl-dit-int4")` returns non-nil
- [ ] After `_ = PixArtComponents.registered`, `Acervo.component(id: "t5-xxl-encoder-int4")` returns non-nil
- [ ] After `_ = PixArtComponents.registered`, `Acervo.component(id: "sdxl-vae-decoder-fp16")` returns non-nil
- [ ] `xcodebuild build -scheme PixArtBackbone -destination 'platform=macOS'` succeeds

### Notes
- PixArt uses a standard linear beta schedule, NOT shifted cosine. This is called out in ARCHITECTURE_STANDALONE.md A4.1 and errata A7.
- The VAE stays in float16 — Conv2d layers do not benefit from weight-only quantization.
- `unconditionalEmbeddingStrategy: .emptyPrompt` means CFG uses empty string encoding (not zero tensor).
- T5-XXL and SDXL VAE are catalog components — their authoritative definitions live in SwiftTubería. This package re-registers them for safety. Acervo deduplicates by component ID (same ID + same repo = no-op).
- The PixArt DiT HuggingFace repo is owned by this package and created during weight conversion (Sortie 6).

---

## Sortie 5: LoRA Support Verification (P6)

**Priority**: 7.5 — Low dependency depth (nothing depends on this). Verification-only sortie. Assign to haiku.

### Objective
Verify that the backbone's `keyMapping` covers all attention projection keys that LoRA adapters target. No new code unless gaps are found.

### Entry Criteria
- Sortie 3 complete (weight key mapping implemented)
- SwiftTubería LoRA infrastructure available

### Files
| File | Action | Description |
|------|--------|-------------|
| `Sources/PixArtBackbone/WeightMapping.swift` | **Verify** | Confirm that `keyMapping` covers all attention projection keys that LoRA adapters target. No new code unless gaps found. |

### LoRA Target Layers (per block, 28 blocks)
- Self-attention: Q, K, V, output projections
- Cross-attention: Q, K, V, output projections

### Exit Criteria
- [ ] For each of the 28 blocks, verify `keyMapping` handles LoRA keys for: self-attn Q, K, V, out; cross-attn Q, K, V, out (8 projections x 28 blocks = 224 keys)
- [ ] Document the full list of LoRA-eligible keys in a code comment in `WeightMapping.swift`
- [ ] If any gaps found: fix the mapping and rebuild (`xcodebuild build -scheme PixArtBackbone -destination 'platform=macOS'`)

### Notes
- REQUIREMENTS.md P6 states: "The backbone's `keyMapping` is reused for LoRA key translation — no separate LoRA target declaration is needed."
- Constraint: single active LoRA per generation (same as FLUX). Multiple LoRAs require sequential load/unload.
- This sortie is intentionally lightweight — the infrastructure lives in SwiftTubería. The work here is verification, not implementation.

---

## Sortie 6: Weight Conversion Scripts (P7)

**Priority**: 10.0 — External dependency (PyTorch, HuggingFace). High risk (PSNR validation). Parallel with Sorties 4-5. Assign to opus.

### Objective
Create Python scripts to convert PyTorch weights to MLX safetensors format. Validate conversions via forward pass comparison.

### Entry Criteria
- Sortie 3 complete (key mapping defines the target key names)
- Python 3.10+ with torch, transformers, diffusers, safetensors, mlx, numpy available
- Access to HuggingFace models: `PixArt-alpha/PixArt-Sigma-XL-2-1024-MS`, `google/t5-v1_1-xxl`, `stabilityai/sdxl-vae`

### Files
| File | Action | Description |
|------|--------|-------------|
| `scripts/convert_pixart_weights.py` | **Create** | PixArt-Sigma PyTorch -> int4 MLX safetensors. Key remapping, Conv2d transposition, int4 quantization (group_size=64). |
| `scripts/convert_t5_weights.py` | **Create** | T5-XXL PyTorch -> int4 MLX safetensors. Shared with any future T5 consumer. |
| `scripts/convert_vae_weights.py` | **Create** | SDXL VAE PyTorch -> float16 MLX safetensors. No quantization (Conv2d layers). |
| `scripts/requirements.txt` | **Create** | Python dependencies: torch, transformers, diffusers, safetensors, mlx, numpy. |
| `scripts/validate_conversion.py` | **Create** | Validation harness: run 5 deterministic prompts at known seeds on both PyTorch and MLX, compare PSNR. |

### Dependencies
- PyTorch reference models (HuggingFace: `PixArt-alpha/PixArt-Sigma-XL-2-1024-MS`)
- T5-XXL model (HuggingFace: `google/t5-v1_1-xxl`)
- SDXL VAE (HuggingFace: `stabilityai/sdxl-vae`)
- Python 3.10+, mlx Python package

### Conversion Steps (per script)
1. Load PyTorch state dict from HuggingFace
2. Apply key remappings (matching ARCHITECTURE_STANDALONE.md A6 tables)
3. Apply tensor transformations (Conv2d transposition for MLX NHWC)
4. Quantize (int4 group_size=64 for transformer/T5, float16 for VAE)
5. Save as MLX safetensors with `config.json`
6. Upload to `intrusive-memory` HuggingFace organization

### Quantization Format (int4, group_size=64)
For each `Linear` weight `[M, N]`:
- `weight` -> `[M, N/8]` uint32 (packed 4-bit)
- `scales` -> `[M, N/64]` float16
- `biases` -> `[M, N/64]` float16 (quantization zero-points)
- `bias` (singular, if present) -> unchanged float16

### Validation Protocol
1. Convert weights (PyTorch -> MLX safetensors)
2. Run forward pass with 5 deterministic prompts at known seeds on both PyTorch and MLX
3. Compare per-layer activations (where feasible) and final output images
4. **All outputs must achieve PSNR > 30 dB vs PyTorch reference**
5. Per-layer validation: investigate if any single layer drops below 25 dB (even if end-to-end passes)

### Exit Criteria
- [ ] `scripts/convert_pixart_weights.py` runs without error: `python scripts/convert_pixart_weights.py --output /tmp/pixart-dit-int4`
- [ ] `scripts/convert_t5_weights.py` runs without error: `python scripts/convert_t5_weights.py --output /tmp/t5-xxl-int4`
- [ ] `scripts/convert_vae_weights.py` runs without error: `python scripts/convert_vae_weights.py --output /tmp/sdxl-vae-fp16`
- [ ] `scripts/requirements.txt` exists and `pip install -r scripts/requirements.txt` succeeds
- [ ] Each output directory contains `config.json` and at least one `.safetensors` file
- [ ] `scripts/validate_conversion.py` reports PSNR > 30 dB for all 5 validation prompts per component
- [ ] No unmapped keys in conversion output (all PyTorch keys either mapped or explicitly discarded)
- [ ] HuggingFace repos created under `intrusive-memory` org: `pixart-sigma-xl-dit-int4-mlx`, `t5-xxl-int4-mlx`, `sdxl-vae-fp16-mlx`

### Notes
- 30 dB PSNR threshold is conservative and accounts for int4 quantization fidelity loss.
- The VAE is NOT quantized — stays in float16.
- `scale_shift_table`, LayerNorm weights, Conv2d patch embed weights, and Embedding weights all stay in float16 (not quantized).
- T5 `shared.weight` (Embedding) and `relative_attention_bias` (Embedding) stay in float16.
- Pin the diffusers version in `requirements.txt` to prevent key name changes from breaking the mapping.

---

## Sortie 7: CLI Tool (P8)

**Priority**: 9.0 — End-user deliverable. Depends on Sorties 1-4. Assign to sonnet.

### Objective
Implement the `PixArtCLI` standalone executable for testing image generation outside of SwiftVinetas.

### Entry Criteria
- Sortie 1 complete (package structure with PixArtCLI target)
- Sortie 4 complete (recipe and component descriptors available)

### Files
| File | Action | Description |
|------|--------|-------------|
| `Sources/PixArtCLI/PixArtCLI.swift` | **Rewrite** | `@main` struct using swift-argument-parser. Root command with subcommands. |
| `Sources/PixArtCLI/GenerateCommand.swift` | **Create** | `generate` subcommand: --prompt, --width, --height, --output, --steps (default 20), --guidance (default 4.5), --seed. Assembles PixArt pipeline recipe, calls `pipeline.generate()`. |
| `Sources/PixArtCLI/DownloadCommand.swift` | **Create** | `download` subcommand: fetch all model components via Acervo. Progress reporting. |
| `Sources/PixArtCLI/InfoCommand.swift` | **Create** | `info` subcommand: show model details, component sizes, download status. |

### Dependencies
- Sortie 1 (package structure with PixArtCLI target)
- Sortie 4 (recipe and component descriptors)
- swift-argument-parser
- SwiftTubería pipeline assembly API

### CLI Interface
```bash
pixart-cli generate --prompt "a cat sitting on a windowsill" --width 1024 --height 1024 --output image.png
pixart-cli generate --prompt "..." --steps 30 --guidance 5.0 --seed 42
pixart-cli download              # fetch all model components
pixart-cli info                  # show model details and download status
```

### Exit Criteria
- [ ] `xcodebuild build -scheme PixArtCLI -destination 'platform=macOS'` succeeds
- [ ] `pixart-cli --help` prints usage information with `generate`, `download`, and `info` subcommands listed
- [ ] `pixart-cli generate --help` lists all flags: --prompt, --width, --height, --output, --steps, --guidance, --seed
- [ ] `pixart-cli download --help` prints usage for the download subcommand
- [ ] `pixart-cli info --help` prints usage for the info subcommand
- [ ] `GenerateCommand.swift` calls `_ = PixArtComponents.registered` before pipeline assembly
- [ ] All 4 CLI source files exist in `Sources/PixArtCLI/`

### Notes
- The CLI is a thin wrapper (~50 lines per command) around the pipeline API. No model logic lives here.
- Error handling: surface Acervo download errors, pipeline assembly errors, and generation errors with human-readable messages.
- The `generate` command triggers `_ = PixArtComponents.registered` to ensure Acervo registration.
- End-to-end integration testing of the CLI (actual image generation) is deferred to Sortie 9 (integration tests).

---

## Sortie 8: Unit Tests (P9.1)

**Priority**: 8.5 — Validates all backbone code. Must pass before integration tests.

### Objective
Implement unit tests for all backbone components using synthetic inputs (no real weights or GPU required).

### Entry Criteria
- Sorties 2-4 complete (all backbone code, weight mapping, recipe, and descriptors)

### Files
| File | Action | Description |
|------|--------|-------------|
| `Tests/PixArtBackboneTests/DiTBlockTests.swift` | **Create** | DiT block forward pass: synthetic input -> expected output shape. AdaLN modulation verification. |
| `Tests/PixArtBackboneTests/AttentionTests.swift` | **Create** | Self-attention shape, QK norm, cross-attention Q/KV split. |
| `Tests/PixArtBackboneTests/EmbeddingsTests.swift` | **Create** | 2D sinusoidal position embedding shapes, timestep embedding, micro-condition embedders. |
| `Tests/PixArtBackboneTests/PatchEmbeddingTests.swift` | **Create** | Spatial -> sequence conversion, correct output dimensions. |
| `Tests/PixArtBackboneTests/FinalLayerTests.swift` | **Create** | Unpatchify output shape, variance channel handling. |
| `Tests/PixArtBackboneTests/WeightMappingTests.swift` | **Create** | Key mapping produces expected output for representative sample keys. Coverage of all ~200 keys. |
| `Tests/PixArtBackboneTests/RecipeTests.swift` | **Create** | Recipe configuration values match spec. Shape contracts validated. |
| `Tests/PixArtBackboneTests/ComponentRegistrationTests.swift` | **Create** | Acervo registration produces expected component IDs. |

### Dependencies
- Sorties 2-4 complete (all backbone code, weight mapping, recipe, descriptors)

### Unit Test Strategy
- All tests use synthetic `MLXArray` inputs with known shapes
- No real model weights required
- No GPU compute required (MLX evaluates lazily but shapes are validated)
- Deterministic — no randomness, no timing dependencies
- No `sleep()`, `Task.sleep()`, or wall-clock assertions

### Exit Criteria
- [ ] All 8 test files exist in `Tests/PixArtBackboneTests/`
- [ ] `xcodebuild test -scheme PixArtBackbone -destination 'platform=macOS'` passes all unit tests
- [ ] Placeholder test from original `PixArtMLXTests.swift` removed
- [ ] `WeightMappingTests.swift` tests at least: 3 global keys, 3 per-block keys, 2 discarded keys, 1 Conv2d transposition
- [ ] `RecipeTests.swift` verifies all configuration values from REQUIREMENTS.md P4
- [ ] `ComponentRegistrationTests.swift` verifies all 3 component IDs are registered
- [ ] `DiTBlockTests.swift` verifies output shape matches input shape (hidden dim preserved)
- [ ] `AttentionTests.swift` verifies QK norm is applied in self-attention
- [ ] `EmbeddingsTests.swift` verifies 2D sinusoidal embedding output shape
- [ ] No test uses `sleep()`, `Task.sleep()`, or wall-clock assertions

### Notes
- What is NOT tested here (tested in SwiftTubería): T5XXLEncoder, SDXLVAEDecoder, DPMSolverScheduler, ImageRenderer, weight loading mechanics.
- Original placeholder test in `PixArtMLXTests.swift` must be removed/replaced.

---

## Sortie 9: Integration Tests & CI Finalization (P9.2)

**Priority**: 6.0 — Final validation. Depends on everything else.

### Objective
Implement gated integration tests for end-to-end generation and finalize CI configuration.

### Entry Criteria
- Sortie 8 complete (all unit tests passing)
- Sortie 7 complete (CLI tool builds)
- Model weights available (either locally or via Acervo download)

### Files
| File | Action | Description |
|------|--------|-------------|
| `Tests/PixArtBackboneTests/BackboneForwardTests.swift` | **Create** | Full backbone forward pass with synthetic weights. Output shape [B, H/8, W/8, 4]. |
| `Tests/PixArtBackboneTests/IntegrationTests.swift` | **Create** | `#if INTEGRATION_TESTS` gated. Full pipeline: prompt -> CGImage. Seed reproducibility. Two-phase loading. |
| `.github/workflows/tests.yml` | **Verify** | CI runs unit tests on macOS 26 and iOS Simulator (iPhone 17, OS 26.1). Integration tests NOT run in CI (require GPU + weights). |

### Integration Test Strategy
- `#if INTEGRATION_TESTS` compilation flag
- Requires downloaded model weights
- Full pipeline assembly and generation
- Seed reproducibility: same seed -> PSNR > 40 dB between runs
- Two-phase loading on simulated memory budget

### Coverage Requirements
- >= 90% line coverage on all new code (unit + integration tests combined)
- No `sleep()`, `Task.sleep()`, or wall-clock assertions
- No environment-dependent unit tests
- Flaky tests treated as failures

### Exit Criteria
- [ ] `Tests/PixArtBackboneTests/BackboneForwardTests.swift` exists and tests full forward pass with synthetic data
- [ ] `Tests/PixArtBackboneTests/IntegrationTests.swift` exists with `#if INTEGRATION_TESTS` gate
- [ ] `BackboneForwardTests` verifies output shape is `[1, H/8, W/8, 4]` for 1024x1024 input (H/8=128, W/8=128)
- [ ] `xcodebuild test -scheme PixArtBackbone -destination 'platform=macOS'` passes (unit tests + backbone forward)
- [ ] CI workflow (`.github/workflows/tests.yml`) runs unit tests on both macOS and iOS Simulator targets
- [ ] Integration tests pass locally when `INTEGRATION_TESTS` flag is set (gated, not in CI)
- [ ] No flaky tests in 3 consecutive CI runs

### Notes
- Cross-platform reproducibility (macOS vs iPadOS) targets PSNR > 30 dB (not byte-identical — MLX makes no such promise).
- Integration tests are NOT run in CI — they require downloaded model weights and GPU compute.

---

## Parallelism Structure

**Critical Path**: Sortie 1 -> Sortie 2 -> Sortie 3 -> Sortie 4 -> Sortie 7 -> Sortie 8 -> Sortie 9 (7 sorties)

**Parallel Execution Groups**:

- **Group 1** (sequential, single agent — SUPERVISING AGENT):
  - Sortie 1: Package Structure (has build step)

- **Group 2** (sequential, single agent — SUPERVISING AGENT):
  - Sortie 2: DiT Backbone (has build step — opus model)

- **Group 3** (can run in parallel after Sortie 2 completes):
  - Sortie 3: Weight Key Mapping (Agent 1 — SUPERVISING AGENT, has build step)
  - No parallel work available here — Sorties 4-9 all depend on Sortie 3

- **Group 4** (can run in parallel after Sortie 3 completes):
  - Sortie 4: Recipe + Descriptors (Agent 1 — SUPERVISING AGENT, has build step)
  - Sortie 5: LoRA Verification (Agent 2 — sub-agent, NO BUILD, haiku model)
  - Sortie 6: Weight Conversion Scripts (Agent 3 — sub-agent, NO BUILD, opus model)

- **Group 5** (after Sortie 4 completes):
  - Sortie 7: CLI Tool (Agent 1 — SUPERVISING AGENT, has build step)

- **Group 6** (sequential after Sortie 7):
  - Sortie 8: Unit Tests (Agent 1 — SUPERVISING AGENT, has build/test step)

- **Group 7** (final):
  - Sortie 9: Integration Tests & CI (Agent 1 — SUPERVISING AGENT, has build/test step)

**Agent Constraints**:
- **Supervising agent**: Handles Sorties 1, 2, 3, 4, 7, 8, 9 (all have build/compile steps)
- **Sub-agent 1** (haiku): Sortie 5 (LoRA verification — read-only analysis)
- **Sub-agent 2** (opus): Sortie 6 (Python weight conversion scripts — file creation, no Swift build)

**Maximum parallelism**: 3 agents in Group 4 (Sorties 4 + 5 + 6 simultaneously)

---

## Dependency Graph

```
Sortie 0: Reconnaissance (COMPLETED)
    |
    v
Sortie 1: Package Structure
    |
    v
Sortie 2: DiT Backbone
    |
    v
Sortie 3: Weight Key Mapping ─────────────────────────────┐
    |                          \                            |
    v                           v                           v
Sortie 4: Recipe + Descriptors  Sortie 5: LoRA (parallel)  Sortie 6: Conversion Scripts (parallel)
    |
    v
Sortie 7: CLI Tool
    |
    v
Sortie 8: Unit Tests
    |
    v
Sortie 9: Integration Tests & CI
```

**Critical path**: 1 -> 2 -> 3 -> 4 -> 7 -> 8 -> 9

**Parallel work**:
- Sortie 6 (conversion scripts) can proceed in parallel with Sorties 4-5 once Sortie 3 defines the key mapping
- Sortie 5 (LoRA verification) can proceed in parallel with Sortie 4 once Sortie 3 is complete

---

## Open Questions & Missing Documentation

### Unresolved Items (must address before execution)

| Sortie | Issue Type | Description | Recommendation |
|--------|-----------|-------------|----------------|
| Sortie 1 | External dependency | SwiftTubería package URL not yet confirmed | Use `https://github.com/intrusive-memory/SwiftTubería` as assumed URL. If not yet published, use branch dependency: `.package(url: "...", branch: "main")`. Sortie agent should check if the repo exists before proceeding. |
| Sortie 2 | Open question | Fused QKV vs separate Q/K/V in backbone self-attention: "Decide in Sortie 2 and adjust [key mapping] here" | **Decision**: Use separate Q/K/V projections (matching diffusers format) for simpler 1:1 key mapping. This avoids concatenation logic in the key mapping closure. Document this decision in `Attention.swift`. |
| Sortie 2 | Open question | Cross-attention: fused KV linear vs separate K/V | **Decision**: Use separate K/V projections (matching diffusers format). Same rationale as self-attention. |
| Sortie 6 | External dependency | Validation requires access to HuggingFace models (~11 GB total download) and GPU for forward pass comparison | Ensure the execution environment has sufficient disk space and network access. Pin exact model revision hashes in scripts. |
| Sortie 6 | Vague criterion (auto-fixed) | Original: "HuggingFace repos created and populated" | Fixed to: specific repo names with verification commands |
| Sortie 9 | External dependency | Integration tests require downloaded model weights (~1.7 GB) and GPU | Integration tests are gated behind `#if INTEGRATION_TESTS`. Unit tests (Sortie 8) provide primary coverage. Integration tests run manually, not in CI. |

### Resolved Items (auto-fixed during refinement)

| Sortie | Original Issue | Fix Applied |
|--------|---------------|-------------|
| All | Exit criteria used `- item` format (not machine-verifiable checkboxes) | Converted to `- [ ]` checklist format |
| S1 | "CI workflow updated and passing" (vague) | Replaced with: "CI workflow updated: `.github/workflows/tests.yml` references correct scheme names" |
| S2 | "Forward pass compiles with synthetic input" (vague) | Replaced with specific `xcodebuild build` command |
| S4 (old) | "Pipeline assembly with PixArtRecipe passes validation (shape contracts match)" (vague) | Replaced with specific property value checks |
| S4 (old) | "Memory profile values are documented and accessible" (vague) | Removed — memory profiles are documentation, not code artifacts |
| S5 (old) | "T5-XXL and SDXL VAE descriptors match SwiftTubería requirements/CATALOG.md values exactly" (not machine-verifiable without the reference file) | Replaced with specific component ID lookup assertions |
| S7 (old, now conversion scripts) | "All 3 conversion scripts run successfully" (no specific commands) | Replaced with explicit `python scripts/convert_*.py --output /tmp/...` commands |
| S8 (old, now CLI) | "pixart-cli generate produces a valid PNG" (requires weights) | Deferred to integration tests; CLI sortie exit criteria focus on build and --help verification |
| S9 (old, now unit tests + integration split) | Single oversized sortie (10 test files, ~500 LoC, 90% budget) | Split into Sortie 8 (unit tests, 8 files) and Sortie 9 (integration tests, 2 files + CI) |

---

## Risk Register

| Risk | Impact | Mitigation |
|------|--------|------------|
| SwiftTubería protocols not yet finalized | Blocks Sorties 2-4 | Use branch dependency; mock protocols locally if needed |
| Weight conversion PSNR below 30 dB | Blocks Sortie 6 validation | Per-layer debugging; check dtype handling (float32 for norms); adjust quantization group_size |
| MLX attention kernel numerics differ from PyTorch | Subtle output differences | Use MLXFast.scaledDotProductAttention with explicit scale; validate per-layer |
| HuggingFace diffusers format changes key names | Key mapping breaks | Pin diffusers version in scripts; document exact model revision hash |
| iPad memory pressure during two-phase loading | OOM on 8 GB devices | Validate phase transitions with `Memory.clearCache()`; test on simulator with memory budget |
| GEGLU gating implementation variance | Incorrect FFN output | Match exact PyTorch implementation: fc1 -> split -> GELU(tanh) * linear; test against reference |
