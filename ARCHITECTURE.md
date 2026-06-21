# pixart-swift-mlx — Architecture (Ecosystem Interface Reference)

**Companion to**: [`docs/complete/REQUIREMENTS.md`](docs/complete/REQUIREMENTS.md)
**Role in ecosystem**: First model plugin. Provides PixArt-Sigma DiT backbone + recipe. Validates the entire pipeline architecture end-to-end.

---

## Dependency Position

```
pixart-swift-mlx
├──▶ SwiftTubería/Tubería          (protocols: Backbone, WeightedSegment)
├──▶ SwiftTubería/TuberíaCatalog   (components: T5XXLEncoder, SDXLVAEDecoder, DPMSolverScheduler, ImageRenderer)
└──▶ SwiftAcervo                   (direct: component registration)
```

---

## What This Package Provides

| Component | Lines | Protocol Conformance |
|---|---|---|
| `PixArtDiT` backbone | ~400 | `Backbone`, `WeightedSegment` |
| `PixArtDiTConfiguration` | ~30 | — |
| Weight key mapping | ~200 keys | via `WeightedSegment.keyMapping` |
| Conv2d tensor transform | ~10 | via `WeightedSegment.tensorTransform` |
| `PixArtRecipe` | ~20 | `PipelineRecipe` |
| Acervo descriptors | ~30 | `ComponentDescriptor` registration |

**Everything else comes from SwiftTubería.**

---

## Backbone Contract

### PixArtDiT : Backbone, WeightedSegment

```
inlet:  BackboneInput {
            latents:          MLXArray [B, H/8, W/8, 4]
            conditioning:     MLXArray [B, 120, 4096]      ← T5XXLEncoder output
            conditioningMask: MLXArray [B, 120]             ← T5XXLEncoder output
            timestep:         MLXArray [B] or scalar
        }
outlet: MLXArray [B, H/8, W/8, 4]                          ← noise prediction
```

### Shape Contract Properties

| Property | Value | Validated Against |
|---|---|---|
| `expectedConditioningDim` | **4096** | `T5XXLEncoder.outputEmbeddingDim` (4096) |
| `outputLatentChannels` | **4** | `SDXLVAEDecoder.expectedInputChannels` (4) |
| `expectedMaxSequenceLength` | **120** | `T5XXLEncoderConfiguration.maxSequenceLength` (120) |

### WeightedSegment Properties

| Property | Value |
|---|---|
| `keyMapping` | Closure: ~200 key remappings (diffusers → MLX module paths) |
| `tensorTransform` | Conv2d transposition: [O,I,kH,kW] → [O,kH,kW,I] |
| `estimatedMemoryBytes` | ~300 MB (int4) |

---

## Pipeline Recipe

```swift
struct PixArtRecipe: PipelineRecipe {
    typealias Encoder = T5XXLEncoder            // catalog
    typealias Sched   = DPMSolverScheduler      // catalog
    typealias Back    = PixArtDiT               // THIS PACKAGE
    typealias Dec     = SDXLVAEDecoder          // catalog
    typealias Rend    = ImageRenderer           // catalog
}
```

### Recipe Configuration Values

| Config | Key Values |
|---|---|
| `encoderConfig` | componentId: `"t5-xxl-encoder-int4"`, maxSequenceLength: 120, embeddingDim: 4096 |
| `schedulerConfig` | betaSchedule: `.linear(0.0001, 0.02)`, predictionType: `.epsilon`, solverOrder: 2, trainTimesteps: 1000 |
| `backboneConfig` | hiddenSize: 1152, numHeads: 16, depth: 28, patchSize: 2, captionChannels: 4096 |
| `decoderConfig` | componentId: `"sdxl-vae-decoder-fp16"`, latentChannels: 4, scalingFactor: 0.13025 |
| `rendererConfig` | `Void` |

### Additional Recipe Properties

| Property | Value |
|---|---|
| `supportsImageToImage` | `false` |
| `unconditionalEmbeddingStrategy` | `.emptyPrompt` |
| `allComponentIds` | `["t5-xxl-encoder-int4", "pixart-sigma-xl-dit-int4", "sdxl-vae-decoder-fp16"]` |
| `quantizationFor(.backbone)` | `.asStored` (pre-quantized int4) |
| Default steps | 20 |
| Default guidance | 4.5 |

---

## Acervo Registration

Requires **SwiftAcervo ≥ 0.16**. This package owns the two PixArt-Sigma backbone descriptors only — T5-XXL encoder and SDXL VAE decoder are registered automatically by `TuberiaCatalog` on module load, so do **not** re-register them here.

```swift
import SwiftAcervo
import TuberiaCatalog

public enum PixArtComponents {
    public static let registered: Bool = {
        CatalogRegistration.shared.ensureRegistered()  // T5 + SDXL VAE
        Acervo.register([
            ComponentDescriptor(
                id: "pixart-sigma-xl-dit-int4",
                type: .backbone,
                displayName: "PixArt-Sigma XL DiT (int4)",
                repoId: "intrusive-memory/pixart-sigma-xl-dit-int4-mlx",
                minimumMemoryBytes: 800_000_000,
                metadata: [
                    "component_role": "backbone",
                    "quantization": "int4",
                    "architecture": "DiT-XL",
                ]
            ),
            ComponentDescriptor(
                id: "pixart-sigma-xl-dit-fp16",
                type: .backbone,
                displayName: "PixArt-Sigma XL DiT (fp16)",
                repoId: "intrusive-memory/pixart-sigma-xl-dit-fp16-mlx",
                minimumMemoryBytes: 2_500_000_000,
                metadata: [
                    "component_role": "backbone",
                    "quantization": "fp16",
                    "architecture": "DiT-XL",
                ]
            ),
        ])
        return true
    }()
}
```

### Slug-vs-`id` warning

Acervo looks up CDN manifests by component `id`, **not** by HuggingFace repo name. The registered `id`s here drop the `-mlx` suffix (`pixart-sigma-xl-dit-int4`, not `pixart-sigma-xl-dit-int4-mlx`). When publishing these models to the CDN, the shipping invocation MUST pin `--slug` explicitly to match the `id`:

```bash
acervo ship intrusive-memory/pixart-sigma-xl-dit-int4-mlx --slug pixart-sigma-xl-dit-int4
acervo ship intrusive-memory/pixart-sigma-xl-dit-fp16-mlx --slug pixart-sigma-xl-dit-fp16
```

Letting `Acervo.slugify(...)` derive the slug from the HF repo would publish under a path the consumer can never find.

---

## Pipeline Data Flow (End-to-End)

```
"a cat sitting on a windowsill"
    │
    ▼
T5XXLEncoder.encode(text, maxLength=120)
    │ → TextEncoderOutput { embeddings: [1,120,4096], mask: [1,120] }
    │
    ▼
DPMSolverScheduler.configure(steps=20)
    │ → SchedulerPlan { timesteps: [999,949,...], sigmas: [...] }
    │
    ▼
[20 iterations]
    PixArtDiT.forward(BackboneInput {
        latents:          [1, 128, 128, 4],    // for 1024x1024
        conditioning:     [1, 120, 4096],
        conditioningMask: [1, 120],
        timestep:         scalar
    }) → [1, 128, 128, 4]
    │
    DPMSolverScheduler.step(output, timestep, sample) → updated latents
    │
    ▼
SDXLVAEDecoder.decode([1, 128, 128, 4])
    │ → DecodedOutput { data: [1, 1024, 1024, 3], metadata: ImageDecoderMetadata }
    │
    ▼
ImageRenderer.render(decoded)
    │ → RenderedOutput.image(CGImage 1024x1024)
```

---

## Memory Profiles

| Configuration | Peak Memory | Components |
|---|---|---|
| All loaded (int4) | ~2 GB | T5 (1.2GB) + DiT (300MB) + VAE (160MB) |
| Phase 1: Conditioning | ~1.4 GB | T5 only |
| Phase 2: Generation | ~500 MB | DiT + VAE |
