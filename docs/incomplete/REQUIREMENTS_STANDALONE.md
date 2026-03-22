# pixart-swift-mlx — Requirements

**Status**: DRAFT — debate and refine before implementation.
**Scope**: Standalone Swift package providing PixArt-Sigma inference on Apple Silicon via MLX. Consumed by SwiftVinetas as `PixArtCore`.

---

## Motivation

SwiftVinetas has a fully-wired engine abstraction (`ImageGenerationEngine` protocol, `EngineRouter`, `PixArtEngine` stub) waiting for a real PixArt backend. No MLX implementation of PixArt exists in any language. This package fills that gap.

PixArt-Sigma is attractive for on-device generation because:
- **~600M parameters** (vs FLUX.2's 4B–32B) — fits comfortably in 8 GB unified memory
- **Apache 2.0 license** — no commercial restrictions (vs FLUX.2's non-commercial license)
- **Up to 4K resolution** — native high-res support
- **Fast inference** — fewer parameters means faster generation per step
- **iPad-viable** — FLUX.2 is effectively macOS-only due to its 16+ GB memory floor; PixArt's ~2 GB footprint (int4, all components loaded) opens the door to on-device image generation on M-series iPads

### iPad Feasibility: PixArt vs FLUX.2

FLUX.2 Klein 4B requires 16 GB minimum RAM — no iPad meets this. FLUX.2 is a Mac-only engine in practice. PixArt-Sigma changes the equation for M-series iPads:

| | FLUX.2 Klein 4B | PixArt-Sigma XL (int4) |
|---|---|---|
| Minimum RAM | 16 GB | ~3 GB (all-at-once) |
| Total download | ~11 GB | ~1.7 GB |
| iPad Pro (M-series, 8–16 GB)? | No | Yes |
| iPad Air (M-series, 8 GB)? | No | Yes |
| iPhone? | No | Not targeted (see below) |

**iPhone is explicitly out of scope.** While PixArt's memory footprint could theoretically fit on 8 GB iPhones, the constrained app memory budget, thermal throttling, and user experience concerns make iPhone a poor target for diffusion model inference today. This package targets **macOS and M-series iPads only**.

This makes PixArt the **primary engine for iPad targets** in SwiftVinetas. The engine abstraction already supports this — `EngineRouter` can register only `PixArtEngine` on iPad while offering both engines on macOS. The practical implication is that SwiftVinetas becomes a macOS + iPad library rather than a macOS-only one.

---

## Architecture Overview

PixArt-Sigma is a Diffusion Transformer (DiT) model with three main components:

```
┌─────────────────────────────────────────────────┐
│                  PixArtPipeline                  │
│                                                  │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐    │
│  │ T5-XXL   │──▶│ PixArt   │──▶│ SDXL     │    │
│  │ Encoder  │   │ DiT      │   │ VAE      │    │
│  │ (int4)   │   │ (int4)   │   │ Decoder  │    │
│  └──────────┘   └──────────┘   └──────────┘    │
│                                                  │
│  prompt → tokens → embeddings → latents → image │
└─────────────────────────────────────────────────┘
```

### Key Architectural Differences from FLUX.2

| Aspect | FLUX.2 (flux-2-swift-mlx) | PixArt-Sigma (this package) |
|--------|---------------------------|----------------------------|
| Text Encoder | Mistral/Qwen (LLM-based) | T5-XXL (encoder-only) |
| Text Encoder Product | `FluxTextEncoders` (separate lib) | `PixArtTextEncoder` (separate lib) |
| Transformer | Double-stream + single-stream DiT | Single-stream DiT with cross-attention |
| Parameters | 4B–32B | ~600M |
| VAE | FLUX-specific VAE | SDXL VAE (shared with Stable Diffusion XL) |
| Scheduler | Flow matching | DPM-Solver / DDPM |
| Conditioning | Concatenated text + image tokens | Cross-attention from T5 embeddings |
| Resolution Handling | Aspect ratio tokens | Aspect ratio binning (predefined resolution buckets) |

---

## P1. Package Structure

### P1.1 Products

```swift
.library(name: "PixArtCore", targets: ["PixArtCore"]),
.library(name: "PixArtTextEncoder", targets: ["PixArtTextEncoder"]),
.executable(name: "PixArtCLI", targets: ["PixArtCLI"]),
```

- **`PixArtTextEncoder`**: T5-XXL encoder in MLX Swift (int4 quantized). Separate library so SwiftVinetas can manage two-phase loading (encode text → unload encoder → load transformer).
- **`PixArtCore`**: Pipeline, transformer, VAE, scheduler, model management. Depends on `PixArtTextEncoder`.
- **`PixArtCLI`**: Standalone command-line tool for testing and debugging.

### P1.2 Platforms

```swift
platforms: [.macOS(.v26), .iOS(.v26)]
```

macOS and iPadOS are first-class targets. iPhone is explicitly out of scope. This is a key differentiator from `flux-2-swift-mlx`, which is macOS-only in practice due to FLUX.2's 16+ GB memory floor. PixArt's lightweight footprint brings M-series iPads into play.

The CLI product (`PixArtCLI`) is macOS-only. The libraries (`PixArtCore`, `PixArtTextEncoder`) target macOS and iPadOS (M-series iPads only — A-series iPads lack the unified memory and GPU compute for viable inference).

### P1.3 Dependencies

```swift
.package(url: "https://github.com/ml-explore/mlx-swift", from: "0.30.2"),
.package(url: "https://github.com/apple/swift-argument-parser", from: "1.3.0"),
.package(url: "https://github.com/huggingface/swift-transformers", from: "1.1.6"),
```

Same MLX stack as `flux-2-swift-mlx`. No additional dependencies. MLX Swift supports both macOS and iOS on Apple Silicon.

### P1.4 Source Layout

```
Sources/
├── PixArtTextEncoder/
│   ├── PixArtTextEncoder.swift         # Public API
│   ├── T5/
│   │   ├── T5Model.swift              # T5 encoder-only model in MLX
│   │   ├── T5Config.swift             # Model configuration
│   │   ├── T5Attention.swift          # Multi-head attention
│   │   ├── T5Block.swift              # Encoder block
│   │   └── T5RelativePositionBias.swift # T5's relative position encoding
│   ├── Tokenizer/
│   │   └── T5Tokenizer.swift          # SentencePiece tokenizer via swift-transformers
│   └── Loading/
│       └── T5ModelLoader.swift        # Weight loading from safetensors
│
├── PixArtCore/
│   ├── PixArtCore.swift               # Public API re-exports
│   ├── Pipeline/
│   │   ├── PixArtPipeline.swift       # Main generation pipeline
│   │   └── PixArtConfig.swift         # Pipeline configuration
│   ├── Transformer/
│   │   ├── PixArtTransformer.swift    # DiT backbone
│   │   ├── DiTBlock.swift             # Transformer block with cross-attention
│   │   ├── AdaLayerNorm.swift         # Adaptive layer norm (timestep conditioning)
│   │   └── PatchEmbed.swift           # Image patch embedding
│   ├── VAE/
│   │   ├── AutoencoderKL.swift        # SDXL VAE decoder
│   │   ├── VAEDecoder.swift           # Decoder implementation
│   │   └── VAEConfig.swift            # VAE configuration
│   ├── Scheduler/
│   │   ├── DPMSolverScheduler.swift   # DPM-Solver++ multistep
│   │   └── SchedulerProtocol.swift    # Scheduler interface
│   ├── LoRA/
│   │   ├── PixArtLoRA.swift           # LoRA adapter injection
│   │   └── LoRAConfig.swift           # LoRA configuration
│   ├── Loading/
│   │   ├── PixArtModelLoader.swift    # Transformer weight loading
│   │   ├── PixArtModelDownloader.swift # HuggingFace Hub download
│   │   └── ModelRegistry.swift        # Known model variants and paths
│   └── Utils/
│       ├── MemoryOptimization.swift   # Memory management for constrained devices
│       └── AspectRatioBins.swift      # Resolution bucket mapping
│
├── PixArtCLI/
│   └── PixArtCLI.swift                # CLI entry point
│
Tests/
├── PixArtTextEncoderTests/
│   └── T5ModelTests.swift
└── PixArtCoreTests/
    ├── PipelineTests.swift
    ├── TransformerTests.swift
    ├── SchedulerTests.swift
    └── VAETests.swift
```

---

## P2. T5-XXL Text Encoder (`PixArtTextEncoder`)

PixArt uses T5-XXL (4.7B parameters full precision) as its text encoder. For on-device use, int4 quantization brings this to ~1.2 GB.

### P2.1 Requirements

- P2.1.1: Implement T5 encoder-only model (no decoder) in MLX Swift using `MLXNN`.
- P2.1.2: Support loading int4 quantized weights from safetensors format.
- P2.1.3: Use T5's relative position bias (not absolute positional embeddings).
- P2.1.4: Support sequence lengths up to 512 tokens (PixArt-Sigma's maximum).
- P2.1.5: Tokenization via `swift-transformers` `Hub` for SentencePiece T5 tokenizer.
- P2.1.6: Output shape: `[batch, seq_len, 4096]` (T5-XXL hidden dimension).
- P2.1.7: Support unloading from memory independently of the transformer (two-phase loading).

### P2.2 Reference Model

- HuggingFace: `PixArt-alpha/PixArt-Sigma-XL-2-1024-MS` (includes T5 encoder weights)
- Alternative: `google/t5-v1_1-xxl` (standalone T5-XXL)
- Quantized: Will need to produce int4 quantized weights (no pre-quantized MLX version exists)

### P2.3 Weight Conversion

- P2.3.1: Provide a Python script (`scripts/convert_t5_weights.py`) that takes PyTorch T5-XXL weights and produces int4 quantized MLX safetensors.
- P2.3.2: Document the quantization method (symmetric int4, group size 64 or 128).
- P2.3.3: Host converted weights on HuggingFace under `intrusive-memory/` org.

---

## P3. PixArt DiT Transformer (`PixArtCore`)

The core denoising model. A Diffusion Transformer with cross-attention to T5 text embeddings.

### P3.1 Requirements

- P3.1.1: Implement PixArt-Sigma XL-2 transformer architecture in MLX Swift.
  - 28 DiT blocks
  - Hidden dimension: 1152
  - 16 attention heads
  - Patch size: 2
  - Cross-attention to T5 text embeddings (4096 → 1152 projection)
- P3.1.2: Adaptive layer norm (AdaLN-Zero) for timestep conditioning.
- P3.1.3: Support int4 quantization for the transformer weights (~600M params → ~300 MB int4).
- P3.1.4: Patch embedding layer that converts latent patches to transformer tokens.
- P3.1.5: Unpatchify layer that converts transformer output back to latent dimensions.
- P3.1.6: Support variable resolution via aspect ratio binning (predefined resolution buckets from 256x256 to 2048x2048+).

### P3.2 Reference Implementation

- PyTorch: `PixArt-alpha/PixArt-sigma` → `diffusion/model/nets/PixArtMS.py`
- HuggingFace diffusers: `PixArtTransformer2DModel`

### P3.3 Weight Conversion

- P3.3.1: Python script (`scripts/convert_pixart_weights.py`) to convert PyTorch weights to int4 MLX safetensors.
- P3.3.2: Handle the key mapping differences between diffusers format and the original PixArt format.
- P3.3.3: Host converted weights on HuggingFace under `intrusive-memory/` org.

---

## P4. SDXL VAE

PixArt-Sigma uses the same VAE as Stable Diffusion XL for encoding/decoding between pixel and latent space.

### P4.1 Requirements

- P4.1.1: Implement the SDXL AutoencoderKL decoder in MLX Swift.
- P4.1.2: Latent channels: 4 (standard SDXL latent space).
- P4.1.3: Scaling factor: 0.13025 (SDXL standard).
- P4.1.4: Decode-only for inference (encoder needed only for img2img, which is a stretch goal).
- P4.1.5: Support fp16/bf16 precision (VAE is small enough to not need int4).

### P4.2 Reference

- The `ml-explore/mlx-examples/stable_diffusion` Python implementation has an SDXL VAE in MLX Python. This is the closest reference for porting.
- HuggingFace: `stabilityai/sdxl-vae` or `madebyollin/sdxl-vae-fp16-fix`

### P4.3 Weight Source

- P4.3.1: Use existing SDXL VAE weights from HuggingFace (already widely available).
- P4.3.2: Convert to MLX safetensors format if needed, host under `intrusive-memory/`.

---

## P5. Scheduler

PixArt-Sigma uses DPM-Solver++ as its primary scheduler.

### P5.1 Requirements

- P5.1.1: Implement DPM-Solver++ multistep scheduler in MLX Swift.
- P5.1.2: Support configurable step counts (default 20, minimum 10, maximum 50).
- P5.1.3: Support classifier-free guidance (default scale 4.5).
- P5.1.4: Implement the noise schedule used by PixArt-Sigma (shifted cosine schedule).
- P5.1.5: Scheduler must be stateless between generations (no shared mutable state).

### P5.2 Reference

- HuggingFace diffusers: `DPMSolverMultistepScheduler`
- PixArt paper: Section 3.2 on noise schedule modifications for high-resolution

---

## P6. Pipeline (`PixArtPipeline`)

The main orchestrator that ties all components together.

### P6.1 Requirements

- P6.1.1: `PixArtPipeline` is the primary public API of `PixArtCore`.
- P6.1.2: Text-to-image generation flow:
  1. Tokenize prompt with T5 tokenizer
  2. Encode tokens with T5-XXL → text embeddings `[1, seq_len, 4096]`
  3. (Optional) Unload T5 to free memory
  4. Initialize random latents at target resolution
  5. Run denoising loop: for each timestep, run DiT transformer with text cross-attention
  6. Apply classifier-free guidance
  7. Decode final latents with SDXL VAE → pixel image
  8. Return `CGImage`
- P6.1.3: Support two-phase loading for memory-constrained devices:
  - Phase 1: Load T5 encoder → encode prompt → unload T5
  - Phase 2: Load DiT transformer + VAE → denoise → decode → unload
- P6.1.4: Step progress callback: `(currentStep: Int, totalSteps: Int) -> Void`
- P6.1.5: Seed support for reproducible generation.
- P6.1.6: Return type includes image, seed used, and generation duration.

### P6.2 Public API Shape

```swift
public actor PixArtPipeline {

    public init(config: PixArtConfig)

    /// Load all model components into memory.
    public func loadModels(
        progressCallback: @Sendable (Double, String) -> Void
    ) async throws

    /// Unload all models from memory.
    public func unloadModels()

    /// Generate an image from a text prompt.
    public func generateTextToImage(
        prompt: String,
        negativePrompt: String?,
        height: Int,
        width: Int,
        steps: Int,
        guidance: Float,
        seed: UInt64,
        onProgress: @Sendable (Int, Int) -> Void
    ) async throws -> PixArtGenerationResult
}

public struct PixArtGenerationResult: Sendable {
    public let image: CGImage
    public let seed: UInt64
    public let steps: Int
    public let guidanceScale: Float
    public let durationSeconds: Double
}

public struct PixArtConfig: Sendable {
    public var model: PixArtModel
    public var quantization: PixArtQuantization
    public var memoryOptimization: PixArtMemoryOptimization

    public static let `default` = PixArtConfig(
        model: .sigmaXL,
        quantization: .int4,
        memoryOptimization: .auto
    )
}

public enum PixArtModel: Sendable {
    case sigmaXL      // PixArt-Sigma XL-2-1024-MS (~600M params)
}

public enum PixArtQuantization: Sendable {
    case none         // Full precision (bf16) — ~1.2 GB transformer
    case int4         // int4 quantized — ~300 MB transformer
    case int8         // int8 quantized — ~600 MB transformer
}

public enum PixArtMemoryOptimization: Sendable {
    case none                    // Load everything at once
    case twoPhaseLoading         // Unload T5 before loading transformer
    case aggressive              // Two-phase + aggressive cache clearing
    case auto                    // Choose based on available memory
}
```

---

## P7. LoRA Support

PixArt supports LoRA fine-tuning for style and subject adaptation.

### P7.1 Requirements

- P7.1.1: Load LoRA adapters from safetensors format.
- P7.1.2: Apply LoRA to DiT transformer attention layers (Q, K, V, out projections).
- P7.1.3: Support configurable LoRA scale (0.0–1.0).
- P7.1.4: Support unloading LoRA to restore base model weights.
- P7.1.5: LoRA targets differ from FLUX.2 — document which layers are targeted.

### P7.2 LoRA Target Layers

PixArt DiT LoRA typically targets:
- `attn.to_q`, `attn.to_k`, `attn.to_v`, `attn.to_out` (self-attention)
- `cross_attn.to_q`, `cross_attn.to_k`, `cross_attn.to_v`, `cross_attn.to_out` (cross-attention)
- Optionally: `ff.net.0.proj` (feed-forward)

This is architecturally different from FLUX.2, where LoRA targets the double-stream and single-stream transformer blocks.

### P7.3 Public API

```swift
public struct PixArtLoRAConfig: Sendable {
    public var filePath: String
    public var scale: Float
    public var activationKeyword: String?
}

extension PixArtPipeline {
    public func loadLoRA(_ config: PixArtLoRAConfig) throws
    public func unloadAllLoRAs()
}
```

---

## P8. Model Registry and SwiftAcervo Integration

Model downloads are handled by SwiftAcervo, not a package-internal downloader. `PixArtCore` declares what it needs; Acervo fetches it from our CDN.

### P8.1 Requirements

- P8.1.1: `ModelRegistry` declares three downloadable components as Acervo-compatible assets:
  - T5-XXL encoder (int4): ~1.2 GB
  - PixArt-Sigma transformer (int4): ~300 MB
  - SDXL VAE: ~160 MB
  - **Total: ~1.7 GB** (vs FLUX.2's ~11 GB for Klein 4B)
- P8.1.2: Each component entry includes: CDN path, expected file size, SHA-256 checksum, and a list of safetensors filenames.
- P8.1.3: `ModelRegistry.isDownloaded(_:)` checks Acervo's cache for component availability.
- P8.1.4: Weight loading reads safetensors from the local path provided by Acervo after download.
- P8.1.5: `PixArtCore` does **not** own download orchestration, progress UI, retry logic, cache eviction, or CDN base URL configuration — all delegated to Acervo and the consuming app.

### P8.2 CDN Hosting

Converted MLX weights are hosted on the intrusive-memory CDN:
- `pixart-sigma-xl-int4/` — transformer weights
- `t5-xxl-encoder-int4/` — T5 text encoder weights
- `sdxl-vae-mlx/` — SDXL VAE weights

Source-of-truth for original PyTorch weights remains HuggingFace (`PixArt-alpha/PixArt-Sigma-XL-2-1024-MS`). Converted weights are produced by the conversion scripts (P11) and mirrored to our CDN.

---

## P9. Memory Management

### P9.1 Requirements

- P9.1.1: Full pipeline (all components loaded simultaneously): ~2 GB with int4 quantization.
- P9.1.2: Two-phase loading peak memory: ~1.4 GB (T5 encoder phase) or ~500 MB (transformer + VAE phase).
- P9.1.3: `PixArtMemoryOptimization.auto` selects the appropriate strategy based on platform and available memory (see P9.2).
- P9.1.4: Expose memory requirements so SwiftVinetas `PixArtEngine.validateMemory()` can check before loading.
- P9.1.5: On iPadOS, respect system memory pressure notifications (`os_proc_available_memory()`) and abort generation gracefully if memory becomes critical, rather than being jettisoned by the OS.

### P9.2 Platform-Aware Memory Strategy

The `.auto` strategy must account for the fact that iPadOS apps receive a fraction of total device RAM (the OS and background processes claim the rest).

| Device Class | Total RAM | App Budget (approx) | Strategy | Max Default Resolution |
|---|---|---|---|---|
| Mac (M1/M2/M3/M4) | 8–192 GB | Nearly all | `.none` (>= 16 GB) or `.twoPhaseLoading` (8–16 GB) | 2048x2048 |
| iPad Pro (M-series) | 8–16 GB | ~5–6 GB | `.twoPhaseLoading` | 1024x1024 |
| iPad Air (M-series) | 8 GB | ~4–5 GB | `.twoPhaseLoading` | 1024x1024 |
| iPhone | Any | N/A | **Not supported** | N/A |
| A-series iPad | Any | N/A | **Not supported** | N/A |

Notes:
- iPadOS enforces hard memory limits per app. Exceeding the limit causes immediate termination (no warning, no swap). Two-phase loading is not optional on iPad — it is the minimum viable strategy.
- `PixArtMemoryOptimization.auto` should query both `ProcessInfo.processInfo.physicalMemory` and `os_proc_available_memory()` at runtime, using the latter on iPadOS to get the real budget.
- A-series iPads and all iPhones are out of scope. The package should fail gracefully at model load time on unsupported hardware (e.g., return `.insufficient` from `validateMemory`) rather than attempting inference that will be killed by the OS.

---

## P10. CLI Tool (`PixArtCLI`)

### P10.1 Requirements

- P10.1.1: Standalone executable for generating images from the command line.
- P10.1.2: Arguments: `--prompt`, `--negative-prompt`, `--width`, `--height`, `--steps`, `--guidance`, `--seed`, `--output`, `--quantization`.
- P10.1.3: Progress output showing current step and elapsed time.
- P10.1.4: `download` subcommand to fetch model weights.
- P10.1.5: `info` subcommand to show model details and download status.

---

## P11. Weight Conversion Scripts

### P11.1 Requirements

- P11.1.1: `scripts/convert_pixart_weights.py` — Convert PixArt-Sigma PyTorch weights to MLX safetensors.
- P11.1.2: `scripts/convert_t5_weights.py` — Convert T5-XXL PyTorch weights to int4 MLX safetensors.
- P11.1.3: `scripts/convert_vae_weights.py` — Convert SDXL VAE PyTorch weights to MLX safetensors.
- P11.1.4: All scripts depend on `torch`, `safetensors`, `mlx`, `transformers` (Python).
- P11.1.5: Scripts validate output by running a forward pass and comparing to PyTorch reference.
- P11.1.6: Document exact reproduction commands in `scripts/README.md`.

---

## P12. Testing Strategy

### P12.1 Unit Tests (No GPU Required)

- P12.1.1: T5 tokenizer produces correct token IDs for known inputs.
- P12.1.2: T5 model config loads correctly from JSON.
- P12.1.3: DiT block forward pass produces correct output shape.
- P12.1.4: Scheduler computes correct timesteps and noise levels.
- P12.1.5: VAE config loads correctly.
- P12.1.6: Aspect ratio binning maps arbitrary dimensions to valid buckets.
- P12.1.7: LoRA weight shapes are validated before injection.
- P12.1.8: Model registry reports correct download status.

### P12.2 Integration Tests (GPU Required)

- P12.2.1: Full pipeline generates a 512x512 image from a prompt (smoke test).
- P12.2.2: Two-phase loading produces the same result as full loading.
- P12.2.3: LoRA loading changes generation output.
- P12.2.4: Seed reproducibility: same seed produces identical output.
- P12.2.5: Different aspect ratios produce correctly-sized images.

### P12.3 iPad Tests

- P12.3.1: Unit tests run on iPad Simulator (no GPU — shape and config tests only).
- P12.3.2: Integration test on a physical M-series iPad confirms end-to-end generation completes without being jettisoned.
- P12.3.3: Two-phase loading is exercised on an M-series iPad (8 GB minimum).
- P12.3.4: Memory pressure handling: generation aborts gracefully under simulated memory pressure rather than crashing.
- P12.3.5: `validateMemory` returns `.insufficient` on A-series iPads and iPhones — no attempt to generate.

### P12.4 Validation Tests (Accuracy)

- P12.4.1: Compare MLX Swift output to PyTorch reference output for the same seed/prompt, within a tolerance (PSNR > 30 dB).
- P12.4.2: T5 encoder output matches PyTorch T5 output within tolerance.

---

## P13. SwiftVinetas Integration Contract

This is the interface `PixArtCore` must satisfy for SwiftVinetas's `PixArtEngine` to replace its stub.

### P13.1 Required from `PixArtCore`

| What SwiftVinetas needs | What `PixArtCore` provides |
|------------------------|---------------------------|
| Create pipeline | `PixArtPipeline(config:)` |
| Load models with progress | `pipeline.loadModels(progressCallback:)` |
| Generate image | `pipeline.generateTextToImage(prompt:negativePrompt:height:width:steps:guidance:seed:onProgress:)` |
| Load LoRA | `pipeline.loadLoRA(_:)` |
| Unload LoRA | `pipeline.unloadAllLoRAs()` |
| Download models | `PixArtModelDownloader` + `ModelRegistry` |
| Check availability | `ModelRegistry.isDownloaded(_:)` |
| Delete models | `PixArtModelDownloader.delete(_:)` |
| Memory requirements | `PixArtConfig` + `PixArtMemoryOptimization` |

### P13.2 Required from `PixArtTextEncoder`

SwiftVinetas does not import `PixArtTextEncoder` directly. It's an internal dependency of `PixArtCore`. However, SwiftVinetas benefits from the separation because:
- Two-phase loading is handled inside `PixArtPipeline`
- The T5 encoder can be shared if future models also use T5

### P13.3 Platform Strategy

With `PixArtCore` available, SwiftVinetas becomes a macOS + iPad library. The engine registration in `VinetasClient` should reflect platform capabilities:

```swift
public init() {
    var engines: [any ImageGenerationEngine] = []

    // PixArt is available on macOS and iPadOS (M-series only)
    #if canImport(PixArtCore)
    engines.append(PixArtEngine())
    #endif

    // FLUX.2 is macOS-only (16+ GB memory requirement excludes all iPads)
    #if os(macOS)
    engines.append(Flux2Engine())
    #endif

    self.router = EngineRouter(engines: engines)
}
```

On iPadOS, `PixArtEngine` is the only registered engine. On macOS, both engines are available and the user or app can choose. This means:
- `VinetasClient.defaultModel` should be platform-aware: `PixArtModelDescriptor.sigmaXL` on iPadOS, `Flux2ModelDescriptor.klein4B` on macOS
- `EngineRouter.allModels` naturally returns only iPad-viable models when only `PixArtEngine` is registered
- No `#if os(iOS)` scattered through downstream app code — the router handles it
- `PixArtEngine.validateMemory()` rejects A-series iPads and iPhones at runtime, so even if the library is linked on an unsupported device, it fails gracefully

---

## P14. Implementation Order

Recommended build order, each step producing a testable artifact:

1. **Weight conversion scripts** — Convert PyTorch weights to MLX format, validate numerically against PyTorch reference. This unblocks everything else.
2. **T5-XXL encoder** (`PixArtTextEncoder`) — Port T5 encoder-only model to MLX Swift. Test: tokenize + encode a prompt, compare embeddings to PyTorch.
3. **SDXL VAE decoder** — Port AutoencoderKL to MLX Swift. Test: decode known latents, compare to PyTorch.
4. **DPM-Solver++ scheduler** — Implement scheduler. Test: noise schedule and step computation match reference.
5. **PixArt DiT transformer** — Port PixArtMS transformer to MLX Swift. Test: single forward pass with known inputs matches PyTorch.
6. **Pipeline assembly** — Wire everything together in `PixArtPipeline`. Test: end-to-end generation produces recognizable images.
7. **LoRA support** — Implement adapter injection. Test: LoRA changes output.
8. **Model download + registry** — HuggingFace Hub integration. Test: download, verify, delete cycle.
9. **CLI tool** — Wire up command-line interface.
10. **Memory optimization** — Implement two-phase loading and aggressive mode.
11. **SwiftVinetas integration** — Replace `PixArtEngine` stub with real implementation.

---

## P15. Resolved Questions

| # | Question | Decision |
|---|----------|----------|
| Q1 | T5 encoder: separate repo or same package? | **Same package, two library products.** Standalone enough for two-phase loading, no reason to split into a separate repo. |
| Q2 | Support PixArt-Alpha? | **No. Sigma only.** Alpha is superseded, not worth the complexity. |
| Q3 | Image-to-image at launch? | **Deferred.** Text-to-image only for v1. img2img requires VAE encoder — add later. |
| Q4 | int4 quantization group size? | **Benchmark and decide.** See P15.1 below. |
| Q5 | Download management? | **SwiftAcervo.** Models hosted on our CDN, downloaded in-app via Acervo. See P15.2 below. |
| Q6 | Bundle weights or download? | **Download on first launch.** Weights are not bundled in the app binary. |
| Q7 | Hardware gating? | **No restrictive gating.** Err on the side of allowing behavior. No M-series chip checks — if the hardware can run it, let it run. `validateMemory` provides guidance, not hard blocks. |
| Q8 | Background task / thread management? | **Consuming app's responsibility.** The library provides an async pipeline. Thread scheduling, `BGProcessingTask`, cancellation policies — all owned by the app, not the library. |

### P15.1 Quantization Benchmark Task

**This must not be skipped.** Before finalizing weight conversion, run a structured benchmark comparing int4 group size 64 vs group size 128:

1. Convert PixArt-Sigma XL weights at both group sizes.
2. Generate the same 10 prompts at the same seeds with both variants on:
   - Mac (M2 Max or similar)
   - M-series iPad (if available)
3. Measure and record:
   - File size difference
   - Generation speed (seconds per image)
   - Peak memory usage
   - Visual quality (side-by-side comparison, PSNR vs PyTorch reference)
4. Choose the group size that gives the best quality-to-size tradeoff for the iPad memory budget.
5. Document the results and decision in this repo.

### P15.2 SwiftAcervo Integration for Model Downloads

Model downloads use SwiftAcervo instead of a package-internal downloader. This replaces the self-contained `PixArtModelDownloader` described in P8.

**Decision**: Host converted PixArt weights on the intrusive-memory CDN. `PixArtCore` declares its model components (T5 encoder, DiT transformer, SDXL VAE) as Acervo-compatible assets. The consuming app (or SwiftVinetas) uses Acervo to download, cache, and manage these assets.

**Implications for P8**:
- P8.1.1: ~~`PixArtModelDownloader` manages downloading~~ → `PixArtCore` provides `ModelRegistry` entries compatible with SwiftAcervo's asset protocol. Actual download orchestration is delegated to Acervo.
- P8.1.7: ~~Cache in `~/Library/Caches/pixart-swift-mlx/`~~ → Cache location managed by Acervo (shared across all intrusive-memory packages).
- P8.2: ~~HuggingFace repos~~ → CDN URLs under intrusive-memory infrastructure. HuggingFace remains the source-of-truth for original weights; converted MLX weights are mirrored to our CDN.

**What `PixArtCore` still owns**:
- `ModelRegistry` — declares known model components, their CDN paths, expected file sizes, and checksums
- Weight loading from local safetensors files (once Acervo has downloaded them)
- `isDownloaded` checks (delegates to Acervo's cache)

**What `PixArtCore` does NOT own**:
- Download orchestration, progress UI, retry logic — that's Acervo
- Cache eviction policy — that's Acervo
- CDN configuration — that's the consuming app, passed through at init

---

## P16. Reference Materials

- **PixArt-Sigma paper**: [arXiv:2403.04692](https://arxiv.org/abs/2403.04692)
- **PixArt-Alpha paper**: [arXiv:2310.00426](https://arxiv.org/abs/2310.00426)
- **PyTorch reference**: `PixArt-alpha/PixArt-sigma` on GitHub
- **HuggingFace diffusers**: `PixArtSigmaPipeline` in `diffusers` library
- **HuggingFace weights**: `PixArt-alpha/PixArt-Sigma-XL-2-1024-MS`
- **MLX Swift**: `ml-explore/mlx-swift` (MLX, MLXNN, MLXRandom, MLXFast)
- **flux-2-swift-mlx**: `VincentGourbin/flux-2-swift-mlx` — architectural reference for MLX Swift image generation
- **mlx-examples stable_diffusion**: `ml-explore/mlx-examples/stable_diffusion` — Python MLX reference for SDXL VAE
- **T5 paper**: [arXiv:1910.10683](https://arxiv.org/abs/1910.10683)

---

## P17. Implementation Architecture

See **[ARCHITECTURE.md](ARCHITECTURE.md)** for internal implementation detail including:

- **A1**: DiT transformer internals (block structure, AdaLN-Zero, cross-attention, timestep conditioning, unpatchify)
- **A2**: T5-XXL encoder internals (RMSNorm, unscaled attention, relative position bias bucketing, GeGLU FFN, tokenizer)
- **A3**: SDXL VAE decoder internals (ResNet blocks, mid-block attention, upsampling, channel progression)
- **A4**: DPM-Solver++ scheduler math (noise schedule, first/second-order updates, CFG, state management)
- **A5**: MLX Swift idioms (module patterns, weight loading, quantization, attention, lazy eval, memory management)
- **A6**: Complete weight key mappings (transformer, T5, VAE, int4 quantization format)
- **A7**: Errata correcting items in this document based on reference implementation research

