# pixart-swift-mlx — Requirements

**Status**: DRAFT — debate and refine before implementation.
**Parent project**: [`PROJECT_PIPELINE.md`](../PROJECT_PIPELINE.md) — Unified MLX Inference Architecture (§3. pixart-swift-mlx, Wave 3)
**Scope**: PixArt-Sigma model plugin for SwiftTubería. Provides the PixArt DiT backbone and model-specific components. All shared infrastructure (weight loading, memory management, scheduling, VAE decoding, image rendering) comes from SwiftTubería.
**Supersedes**: `docs/incomplete/REQUIREMENTS_STANDALONE.md` (standalone library approach)

---

## Motivation

PixArt-Sigma is a ~600M parameter Diffusion Transformer — small enough for M-series iPads, Apache 2.0 licensed, and capable of up to 4K resolution. Under the SwiftTubería architecture, this package provides only what is unique to PixArt. Everything else is pipe segments from the shared catalog.

### What This Package Provides vs What SwiftTubería Provides

| Concern | This Package | SwiftTubería |
|---|---|---|
| DiT backbone (28-block transformer) | **Yes** — the only substantial new code | — |
| Weight key mapping (HF → MLX) | **Yes** | — |
| Model configuration | **Yes** | — |
| Pipeline recipe | **Yes** | — |
| T5-XXL text encoder | — | **Yes** (catalog: `T5XXLEncoder`) |
| SDXL VAE decoder | — | **Yes** (catalog: `SDXLVAEDecoder`) |
| DPM-Solver++ scheduler | — | **Yes** (catalog: `DPMSolverScheduler`) |
| Image rendering (MLXArray → CGImage) | — | **Yes** (catalog: `ImageRenderer`) |
| Weight loading + quantization | — | **Yes** (infrastructure: `WeightLoader`) |
| Model downloading + caching | — | **Yes** (via SwiftAcervo Component Registry) |
| Memory management | — | **Yes** (infrastructure: `MemoryManager`) |
| Progress reporting | — | **Yes** (infrastructure: `PipelineProgress`) |

---

## P1. Package Structure

### P1.1 Products

```swift
.library(name: "PixArtBackbone", targets: ["PixArtBackbone"]),
.executable(name: "PixArtCLI", targets: ["PixArtCLI"]),
```

- **`PixArtBackbone`** — The DiT transformer, configuration, weight key mapping, and pipeline recipe. This is what SwiftVinetas imports (as `PixArtCore` or via SwiftTubería's pipeline builder).
- **`PixArtCLI`** — Standalone command-line tool for testing and debugging.

### P1.2 Dependencies

```swift
.package(url: "<SwiftTubería>", from: "0.1.0"),  // protocols + catalog
.package(url: "https://github.com/apple/swift-argument-parser", from: "1.3.0"),
```

This package depends **only** on SwiftTubería (which transitively provides mlx-swift, swift-transformers). No direct MLX dependency needed — the pipeline protocols abstract it.

### P1.3 Platforms

```swift
platforms: [.macOS(.v26), .iOS(.v26)]
```

PixArt's ~2 GB footprint (int4, all components) makes it viable on M-series iPads. This is a key differentiator from FLUX.2.

---

## P2. PixArt DiT Backbone

The backbone is the single substantial piece of new code. It conforms to SwiftTubería's `Backbone` protocol.

### P2.1 Architecture Summary

| Parameter | Value |
|---|---|
| hidden_size | 1152 |
| num_heads | 16 |
| head_dim | 72 |
| depth | 28 blocks |
| patch_size | 2 |
| in_channels | 4 (VAE latent) |
| out_channels | 8 (4 noise + 4 variance; last 4 discarded at inference) |

**Variance channel handling**: At inference time, the backbone's `forward()` discards the last 4 channels: `rawOutput[.all, .all, .all, 0..<4]`. The Backbone protocol outlet is `[B, H/8, W/8, 4]` — the pipeline never sees 8 channels. Variance channels are only used during training (learned sigma for loss weighting).
| mlp_ratio | 4.0 (FFN hidden = 4608) |
| caption_channels | 4096 (T5-XXL embedding dim) |
| max_text_length | 120 tokens |

### P2.2 Key Architectural Features

1. **AdaLN-Single** — One global `t_block` MLP produces timestep conditioning. Each of the 28 blocks adds its own learned `scale_shift_table` (6 × 1152 parameters) to produce per-block modulation. This is cheaper than per-block MLPs.

2. **Cross-attention to T5 embeddings** — Each block has self-attention (image tokens attend to each other) then cross-attention (image tokens attend to projected T5 text embeddings). Cross-attention receives NO timestep modulation.

3. **2D sinusoidal position embeddings** — Recomputed dynamically per forward pass based on actual spatial dimensions. Enables variable resolution natively via aspect ratio binning.

4. **Micro-conditioning** — Resolution and aspect ratio are embedded and concatenated with the timestep embedding, giving the model explicit awareness of the target dimensions. Uses a 64-bucket aspect ratio scheme from the PixArt-Sigma paper at 1024px base resolution. Common buckets: 1:1 (1024×1024), 4:3 (1152×896), 3:4 (896×1152), 16:9 (1344×768), 9:16 (768×1344), etc. The backbone rounds user-requested resolution to the nearest bucket for conditioning but generates at the requested resolution. Bucket scheme is documented in the backbone configuration struct.

### P2.3 Backbone Protocol Conformance

```
inlet:  BackboneInput {
            latents:          MLXArray [B, H/8, W/8, 4]
            conditioning:     MLXArray [B, 120, 4096]    ← from T5XXLEncoder outlet
            conditioningMask: MLXArray [B, 120]           ← from T5XXLEncoder outlet
            timestep:         MLXArray [B]
        }
outlet: MLXArray [B, H/8, W/8, 4]                        ← noise prediction (variance channels discarded)
```

**Shape contract properties**:
- `expectedConditioningDim: 4096` — matches T5XXLEncoder's `outputEmbeddingDim`
- `outputLatentChannels: 4` — matches SDXLVAEDecoder's `expectedInputChannels`
- `expectedMaxSequenceLength: 120` — matches T5XXLEncoderConfiguration's `maxSequenceLength`

The backbone expects T5-XXL embeddings (dim 4096) as conditioning. This is validated at pipeline assembly time — connecting a CLIP encoder (dim 768) would fail.

**Lifecycle**: Conforms to `WeightedSegment` (see SwiftTubería `requirements/PROTOCOLS.md`). The pipeline loads weights via `WeightLoader` using the backbone's `keyMapping` and `tensorTransform`, then calls `apply(weights:)`.

---

## P3. Weight Key Mapping

PixArt weights from HuggingFace (diffusers format) require key remapping to match the MLX module structure. The backbone's `keyMapping: KeyMapping` property provides a closure that maps each safetensors key to the corresponding MLX module path. WeightLoader calls this for every key during loading.

**Scope**: ~14 global mappings + per-block mappings for 28 blocks. Total: ~200 key pairs.

**Key remapping categories**:
- Patch embedding: `pos_embed.proj` → module path
- Timestep conditioning: `adaln_single.*` → `t_block.*`, `t_embedder.*`
- Per-block attention: diffusers 3-way Q/K/V split → combined or separate projections
- Per-block cross-attention: 2-way K/V split
- Per-block FFN: `ff.net` → `mlp.*`
- Caption projection: `caption_projection` → module path
- Final layer: `proj_out`, `scale_shift_table`

**Conv2d weight transposition**: The backbone's `tensorTransform: TensorTransform` transposes Conv2d weights from PyTorch [O,I,kH,kW] → MLX [O,kH,kW,I] for all convolutional layers. WeightLoader applies this per-tensor after key remapping.

---

## P4. Pipeline Recipe

The PixArt recipe connects catalog components with the custom backbone:

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ T5XXLEncoder │───▶│  PixArtDiT   │───▶│SDXLVAEDecoder│───▶│ImageRenderer │
│   (catalog)  │    │ (THIS REPO)  │    │   (catalog)  │    │  (catalog)   │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
                          ▲
                    ┌─────┴──────┐
                    │DPMSolver++ │
                    │  (catalog) │
                    └────────────┘
```

**Pipe compatibility**:
- T5XXLEncoder outlet: embeddings [B, 120, 4096] → PixArtDiT inlet: conditioning expects 4096-dim
- PixArtDiT outlet: latents [B, H/8, W/8, 4] → SDXLVAEDecoder inlet: expects 4 latent channels, scale 0.13025
- SDXLVAEDecoder outlet: pixels [B, H, W, 3] → ImageRenderer inlet: expects 3-channel float data

### P4.1 Recipe Configuration Values

The `PixArtRecipe` provides configuration values for each catalog component. Configuration types are defined in SwiftTubería `requirements/CATALOG.md`.

```swift
struct PixArtRecipe: PipelineRecipe {
    typealias Encoder = T5XXLEncoder
    typealias Sched = DPMSolverScheduler
    typealias Back = PixArtDiT
    typealias Dec = SDXLVAEDecoder
    typealias Rend = ImageRenderer

    var encoderConfig: T5XXLEncoderConfiguration {
        .init(componentId: "t5-xxl-encoder-int4", maxSequenceLength: 120, embeddingDim: 4096)
    }
    var schedulerConfig: DPMSolverSchedulerConfiguration {
        .init(betaSchedule: .linear(betaStart: 0.0001, betaEnd: 0.02),
              predictionType: .epsilon, solverOrder: 2, trainTimesteps: 1000)
    }
    var backboneConfig: PixArtDiTConfiguration { ... }  // defined in this package
    var decoderConfig: SDXLVAEDecoderConfiguration {
        .init(componentId: "sdxl-vae-decoder-fp16", latentChannels: 4, scalingFactor: 0.13025)
    }
    var rendererConfig: Void { () }

    var supportsImageToImage: Bool { false }
    var unconditionalEmbeddingStrategy: UnconditionalEmbeddingStrategy { .emptyPrompt }
    var allComponentIds: [String] {
        ["t5-xxl-encoder-int4", "pixart-sigma-xl-dit-int4", "sdxl-vae-decoder-fp16"]
    }
    func quantizationFor(_ role: PipelineRole) -> QuantizationConfig { .asStored }
}
```

**Default generation parameters** (used by `PixArtModelDescriptor`):

| Parameter | Value |
|---|---|
| default_steps | 20 |
| default_guidance | 4.5 |

**Important**: PixArt uses standard linear beta schedule, NOT shifted cosine.

### P4.2 Memory Profiles

| Configuration | Peak Memory | Strategy |
|---|---|---|
| All components loaded (int4) | ~2 GB | Mac with 8+ GB |
| Two-phase: T5 phase | ~1.4 GB | iPad (8 GB) |
| Two-phase: DiT + VAE phase | ~500 MB | iPad (8 GB) |

---

## P5. Acervo Component Descriptors

Components registered into SwiftAcervo's Component Registry at import time:

| Component | Acervo ID | Type | Size (int4) | HuggingFace Source |
|---|---|---|---|---|
| PixArt-Sigma XL DiT | `pixart-sigma-xl-dit-int4` | backbone | ~300 MB | intrusive-memory CDN |
| T5-XXL | `t5-xxl-encoder-int4` | encoder | ~1.2 GB | intrusive-memory CDN |
| SDXL VAE | `sdxl-vae-decoder-fp16` | decoder | ~160 MB | existing SDXL VAE |

T5-XXL and SDXL VAE are **catalog components** — their Acervo descriptors are authoritatively defined in SwiftTubería `requirements/CATALOG.md` § Catalog Component Acervo Descriptors. TuberíaCatalog registers them at import time. This package re-registers them for safety (Acervo deduplicates by component ID; same ID + same repo = no-op). The values below MUST match SwiftTubería `requirements/CATALOG.md` § Catalog Component Acervo Descriptors.

Pipeline code accesses these components exclusively through `AcervoManager.shared.withComponentAccess(id)` — never through file paths.

**Registration timing**: Components are registered at import time via Swift static `let` initialization:
```swift
public enum PixArtComponents {
    public static let registered: Bool = {
        Acervo.register([
            // Model-specific — owned by this package
            ComponentDescriptor(id: "pixart-sigma-xl-dit-int4",
                                type: .backbone,
                                huggingFaceRepo: "intrusive-memory/pixart-sigma-xl-dit-int4-mlx",
                                ...),
            // Catalog components — values from SwiftTubería `requirements/CATALOG.md` § Catalog Component Acervo Descriptors (re-registered for safety)
            ComponentDescriptor(id: "t5-xxl-encoder-int4",
                                type: .encoder,
                                huggingFaceRepo: "intrusive-memory/t5-xxl-int4-mlx",
                                ...),
            ComponentDescriptor(id: "sdxl-vae-decoder-fp16",
                                type: .decoder,
                                huggingFaceRepo: "intrusive-memory/sdxl-vae-fp16-mlx",
                                ...),
        ])
        return true
    }()
}
```
Swift guarantees this initializer is thread-safe and runs exactly once. Pipeline assembly can call `_ = PixArtComponents.registered` as a defensive trigger.

**HuggingFace repo conventions**: Converted model weights are hosted under the `intrusive-memory` HuggingFace organization. Repo naming: `intrusive-memory/{model}-{quantization}-mlx`.

| Component | HuggingFace Repo | Notes |
|---|---|---|
| PixArt-Sigma XL DiT (int4) | `intrusive-memory/pixart-sigma-xl-dit-int4-mlx` | Owned by this package — created during weight conversion (P7) |
| T5-XXL (int4) | `intrusive-memory/t5-xxl-int4-mlx` | Catalog component — authoritative definition in SwiftTubería `requirements/CATALOG.md` § Catalog Component Acervo Descriptors |
| SDXL VAE (fp16) | `intrusive-memory/sdxl-vae-fp16-mlx` | Catalog component — authoritative definition in SwiftTubería `requirements/CATALOG.md` § Catalog Component Acervo Descriptors |

The T5-XXL and SDXL VAE repos are shared with all future consumers of these catalog components. The PixArt DiT repo is owned by this package. All repos are created during weight conversion (P7) and populated with MLX safetensors plus `config.json`.

---

## P6. LoRA Support

PixArt LoRA adapters target the DiT transformer's attention layers:
- Self-attention: Q, K, V, output projections (per block)
- Cross-attention: Q, K, V, output projections (per block)

SwiftTubería's LoRA infrastructure applies adapters to all keys in the LoRA safetensors file that match the loaded model's keys (see SwiftTubería `requirements/PIPELINE.md` § LoRA System). The backbone's `keyMapping` is reused for LoRA key translation — no separate LoRA target declaration is needed.

**Constraint**: Single active LoRA per generation (same constraint as FLUX). Multiple LoRAs require sequential load/unload. SwiftTubería's LoRA infrastructure can lift this constraint in a future version by supporting `[LoRAConfig]` with per-adapter scaling.

---

## P7. Weight Conversion Scripts

- `scripts/convert_pixart_weights.py` — PixArt-Sigma PyTorch → int4 MLX safetensors
- `scripts/convert_t5_weights.py` — T5-XXL PyTorch → int4 MLX safetensors (shared with any future T5 consumer)
- `scripts/convert_vae_weights.py` — SDXL VAE PyTorch → float16 MLX safetensors (shared)

Each script validates output via forward pass comparison to PyTorch reference. Test protocol:
1. Convert weights (PyTorch → MLX safetensors)
2. Run forward pass with 5 deterministic prompts at known seeds on both PyTorch and MLX
3. Compare per-layer activations (where feasible) and final output images
4. All outputs must achieve PSNR > 30 dB vs PyTorch reference
5. Per-layer validation: investigate if any single layer drops below 25 dB (even if end-to-end passes)

30 dB is conservative and accounts for int4 quantization fidelity loss.

---

## P8. CLI Tool

Standalone executable for testing outside of SwiftVinetas:

```bash
pixart-cli generate --prompt "..." --width 1024 --height 1024 --output image.png
pixart-cli download              # fetch all model components
pixart-cli info                  # show model details and download status
```

Internally, the CLI assembles the PixArt pipeline recipe and calls `pipeline.generate()`.

---

## P9. Testing Strategy

### P9.1 Backbone Unit Tests
- DiT block forward pass: synthetic input → expected output shape
- AdaLN-Single modulation: verify scale/shift/gate application
- Cross-attention: verify Q from image, K/V from text
- Patch embedding: verify spatial → sequence conversion
- Position embedding: verify 2D sinusoidal computation

### P9.2 Integration Tests
- Full pipeline recipe assembly → validation passes
- Prompt → CGImage (correct dimensions, non-zero pixels)
- Seed reproducibility:
  - Same device, same seed → PSNR > 40 dB between runs ("visually identical")
  - Cross-platform (macOS vs iPadOS, different M-series) → PSNR > 30 dB
  - Byte-for-byte reproduction is NOT guaranteed (MLX makes no such promise)
- Two-phase loading on memory-constrained simulated budget

### P9.3 What Is NOT Tested Here
- T5XXLEncoder correctness (tested in SwiftTubería catalog tests)
- SDXLVAEDecoder correctness (tested in SwiftTubería catalog tests)
- DPMSolverScheduler correctness (tested in SwiftTubería catalog tests)
- ImageRenderer correctness (tested in SwiftTubería catalog tests)
- Weight loading mechanics (tested in SwiftTubería infrastructure tests)

This is the core benefit of the pipeline architecture: component tests are written once and validated for every model that uses them.

### P9.4 Coverage and CI Stability Requirements

- All new code must achieve **≥90% line coverage** in unit tests. Coverage is measured per-target and enforced in CI.
- **No timed tests**: Tests must not use `sleep()`, `Task.sleep()`, `Thread.sleep()`, fixed-duration `XCTestExpectation` timeouts, or any wall-clock assertions. All asynchronous behavior must be validated via deterministic synchronization (`async`/`await`, `AsyncStream`, fulfilled expectations with immediate triggers).
- **No environment-dependent tests**: Backbone unit tests (P9.1) must use synthetic inputs and run without real model weights or GPU. Integration tests (P9.2) that require downloaded models and GPU compute must be clearly separated (separate test target or `#if INTEGRATION_TESTS` gate).
- **Flaky tests are test failures**: A test that passes intermittently is treated as a failing test until fixed. CI must not use retry-on-failure to mask flakiness.

---

## P10. SwiftVinetas Integration

SwiftVinetas's `PixArtEngine` stub currently gates on `#if canImport(PixArtCore)`. With the pipeline architecture:

1. `PixArtEngine` imports `PixArtBackbone` and `Tubería`
2. Constructs a `DiffusionPipeline` using the PixArt recipe
3. Delegates `generate()`, `loadModel()`, `download()` to the pipeline
4. The engine becomes a thin adapter (~50 lines) between Vinetas's `ImageGenerationEngine` protocol and the assembled pipeline

---

## P11. Reference Materials

- PixArt-Sigma paper: arXiv:2403.04692
- PixArt-Alpha paper: arXiv:2310.00426
- PyTorch reference: `PixArt-alpha/PixArt-sigma` on GitHub
- HuggingFace diffusers: `PixArtSigmaPipeline`
- Detailed internal architecture: `docs/incomplete/ARCHITECTURE_STANDALONE.md` (tensor shapes, weight mappings, MLX idioms — still valid as implementation reference)
