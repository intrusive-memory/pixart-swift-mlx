# Graph Report - .  (2026-11-22)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 808 nodes · 1178 edges · 40 communities (28 shown, 12 thin omitted)
- Extraction: 88% EXTRACTED · 12% INFERRED · 0% AMBIGUOUS · INFERRED: 138 edges (avg confidence: 0.81)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `cdbfc449`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- [[_COMMUNITY_Process-Wide Telemetry Tests|Process-Wide Telemetry Tests]]
- [[_COMMUNITY_DiT Embedding Layers|DiT Embedding Layers]]
- [[_COMMUNITY_Int4 Quantization Validation|Int4 Quantization Validation]]
- [[_COMMUNITY_FP16 Recipe Config|FP16 Recipe Config]]
- [[_COMMUNITY_PixArt DiT Backbone|PixArt DiT Backbone]]
- [[_COMMUNITY_DiT Block & Modulation|DiT Block & Modulation]]
- [[_COMMUNITY_Pipeline Recipe Config|Pipeline Recipe Config]]
- [[_COMMUNITY_VAE Weight Conversion|VAE Weight Conversion]]
- [[_COMMUNITY_Project Documentation|Project Documentation]]
- [[_COMMUNITY_PixArt Weight Conversion|PixArt Weight Conversion]]
- [[_COMMUNITY_Telemetry Event Types|Telemetry Event Types]]
- [[_COMMUNITY_DPM-Solver Comparison|DPM-Solver Comparison]]
- [[_COMMUNITY_T5 Weight Conversion|T5 Weight Conversion]]
- [[_COMMUNITY_Weight Key Mapping Tests|Weight Key Mapping Tests]]
- [[_COMMUNITY_Pipeline Integration Tests|Pipeline Integration Tests]]
- [[_COMMUNITY_Aspect Ratio Bucket Selection|Aspect Ratio Bucket Selection]]
- [[_COMMUNITY_Int4 Dequantization Script|Int4 Dequantization Script]]
- [[_COMMUNITY_Attention Layers|Attention Layers]]
- [[_COMMUNITY_Component Registration Tests|Component Registration Tests]]
- [[_COMMUNITY_Weight Apply Tests|Weight Apply Tests]]
- [[_COMMUNITY_Beta Schedule Validation|Beta Schedule Validation]]
- [[_COMMUNITY_Reference Generation Stats|Reference Generation Stats]]
- [[_COMMUNITY_Backbone Configuration Tests|Backbone Configuration Tests]]
- [[_COMMUNITY_Attention Tests|Attention Tests]]
- [[_COMMUNITY_Backbone Forward Pass Tests|Backbone Forward Pass Tests]]
- [[_COMMUNITY_DiT Block Tests|DiT Block Tests]]
- [[_COMMUNITY_Final Layer Tests|Final Layer Tests]]
- [[_COMMUNITY_Pixel Range Regression Tests|Pixel Range Regression Tests]]
- [[_COMMUNITY_Weight Mapping Tests|Weight Mapping Tests]]
- [[_COMMUNITY_Recipe Validation Errors|Recipe Validation Errors]]
- [[_COMMUNITY_DiT Configuration|DiT Configuration]]
- [[_COMMUNITY_Micro-Conditioning Skip Tests|Micro-Conditioning Skip Tests]]
- [[_COMMUNITY_Telemetry Reporter|Telemetry Reporter]]
- [[_COMMUNITY_Weight Mapping Definitions|Weight Mapping Definitions]]
- [[_COMMUNITY_Telemetry Singleton|Telemetry Singleton]]
- [[_COMMUNITY_Pipeline Components|Pipeline Components]]
- [[_COMMUNITY_Quantization Config|Quantization Config]]
- [[_COMMUNITY_CI Workflows|CI Workflows]]

## God Nodes (most connected - your core abstractions)
1. `PixArtRecipe` - 48 edges
2. `PixArtFP16Recipe` - 39 edges
3. `PixArtDiT` - 24 edges
4. `RecipeTests` - 23 edges
5. `PixArtFP16RecipeTests` - 22 edges
6. `MockReporter` - 19 edges
7. `EmbeddingsTests` - 18 edges
8. `PixArtDiT` - 14 edges
9. `compute_psnr()` - 13 edges
10. `BucketSelectionTests` - 12 edges

## Surprising Connections (you probably didn't know these)
- `PixArtFP16Recipe` --semantically_similar_to--> `PixArtRecipe`  [INFERRED] [semantically similar]
  docs/REQUIREMENTS-instrumentation.md → ARCHITECTURE.md
- `Warm color-cast attribution rationale` --conceptually_related_to--> `Failure 2: wrong beta schedule`  [AMBIGUOUS]
  Sources/PixArtBackbone/Telemetry/README.md → docs/complete/sigma-foundry-diagnosis.md
- `PixArtBackbone library product` --references--> `PixArtDiT`  [INFERRED]
  README.md → ARCHITECTURE.md
- `MockReporter (test helper)` --implements--> `PixArtTelemetryReporter protocol`  [INFERRED]
  docs/REQUIREMENTS-instrumentation.md → Sources/PixArtBackbone/Telemetry/README.md
- `Instrumentation Requirements (docs/)` --conceptually_related_to--> `backboneForwardComplete event`  [AMBIGUOUS]
  docs/REQUIREMENTS-instrumentation.md → Sources/PixArtBackbone/Telemetry/README.md

## Import Cycles
- None detected.

## Communities (40 total, 12 thin omitted)

### Community 0 - "Process-Wide Telemetry Tests"
Cohesion: 0.05
Nodes (21): Error, PixArtRecipe, PixArtRecipeError, shapeMismatch, PipelineAssemblyTests, TwoPhaseLoadingTests, PixelRangeRegressionTests, RecipeTests (+13 more)

### Community 1 - "DiT Embedding Layers"
Cohesion: 0.08
Nodes (24): Module, CrossAttention, SelfAttention, AspectRatioEmbedder, CaptionProjection, get2DSinusoidalPositionEmbeddings(), MicroConditionEmbedder, sinusoidalEmbedding1D() (+16 more)

### Community 2 - "Int4 Quantization Validation"
Cohesion: 0.06
Nodes (50): AdaLN-Zero / AdaLN-Single Conditioning, Cross-Attention + Caption Projection, pixart-swift-mlx Implementation Architecture, DPM-Solver++ Scheduler (linear beta schedule), Final Layer / Unpatchify (variance channels), Patch Embedding + 2D Sinusoidal Position Embeddings, PixArt DiT Transformer (28 blocks, 1152 hidden), SDXL VAE Decoder (+42 more)

### Community 3 - "FP16 Recipe Config"
Cohesion: 0.07
Nodes (17): MockReporter, PixArtProcessWideTelemetryTests, PixArtTelemetryAnomalyTests, PixArtTelemetryLockContentionTests, PixArtTelemetryWeightApplyFP16Tests, PixArtTelemetryWeightApplyINT4Tests, PixArtTelemetryReporter, PixArtTelemetryEvent (+9 more)

### Community 4 - "PixArt DiT Backbone"
Cohesion: 0.05
Nodes (47): BackboneInput, Conv2d tensor transform [O,I,kH,kW]->[O,kH,kW,I], DPMSolverScheduler, ImageRenderer, Memory profiles (~2 GB int4), Pipeline data flow (end-to-end), PipelineRecipe protocol, PixArtDiT (+39 more)

### Community 5 - "DiT Block & Modulation"
Cohesion: 0.06
Nodes (40): App Group configuration (ACERVO_APP_GROUP_ID), SwiftAcervo 0.16 'ask the library' contract, CDNManifest, Critical rules for AI agents, graphify codemap (graphify-out/), Makefile targets, ModelAvailability (4-case enum), pixart-swift-mlx (model plugin) (+32 more)

### Community 6 - "Pipeline Recipe Config"
Cohesion: 0.07
Nodes (29): ndarray, Tests for int4 quantization and dequantization roundtrip., Quantize then dequantize and return PSNR vs original., Random weights should survive int4 roundtrip with PSNR > 25 dB., Constant weights should survive roundtrip with very high PSNR.          Not exac, All-zero weights should roundtrip exactly., Verify packed weight, scales, biases have correct shapes., PixArt quantizer requires N divisible by GROUP_SIZE. (+21 more)

### Community 7 - "VAE Weight Conversion"
Cohesion: 0.07
Nodes (16): PipelineRecipe, PixArtFP16Recipe, PixArtFP16RecipeTests, Bool, DPMSolverSchedulerConfiguration, Float, Int, PipelineRole (+8 more)

### Community 8 - "Project Documentation"
Cohesion: 0.08
Nodes (21): AspectRatioEmbedder, Backbone, CaptionProjection, Configuration, Conv2d, FinalLayer, PixArtDiT, PatchEmbeddingTests (+13 more)

### Community 9 - "PixArt Weight Conversion"
Cohesion: 0.07
Nodes (35): CLITelemetryBootstrap, Dual-seam telemetry pattern, effectiveReporter, PixArtEventEncoding.swift, PixArtTelemetry (process-wide seam), PixArtTelemetryCLIAdapter, PixArtProcessWideTelemetryTests, PixArtDiT.setTelemetry (instance seam) (+27 more)

### Community 10 - "Telemetry Event Types"
Cohesion: 0.09
Nodes (16): CrossAttention, DiTBlock, GEGLUFFN, FinalLayer, t2iModulate(), ModulationHelperTests, SelfAttention, Float (+8 more)

### Community 11 - "DPM-Solver Comparison"
Cohesion: 0.07
Nodes (19): build_config(), convert_weights(), load_vae_state_dict(), main(), ndarray, Tensor, Transpose Conv2d weight from PyTorch [O,I,kH,kW] to MLX [O,kH,kW,I]., Load SDXL VAE state dict and config from HuggingFace. (+11 more)

### Community 12 - "T5 Weight Conversion"
Cohesion: 0.09
Nodes (23): build_config(), build_key_mapping(), convert_weights(), load_pytorch_state_dict(), main(), ndarray, Tensor, quantize_int4() (+15 more)

### Community 13 - "Weight Key Mapping Tests"
Cohesion: 0.09
Nodes (25): Codable, String, AnomalyKind, inf, nan, outOfRange, zeroLatent, AnomalyPhase (+17 more)

### Community 14 - "Pipeline Integration Tests"
Cohesion: 0.15
Nodes (19): diffusers_timesteps(), epsilon_to_x0(), FakeModel, main(), ndarray, Swift epsilon prediction conversion., Run diffusers DPMSolverMultistepScheduler and return per-step latent means., Run Swift DPM-Solver port and return per-step latent means. (+11 more)

### Community 15 - "Aspect Ratio Bucket Selection"
Cohesion: 0.18
Nodes (10): CGImage, Double, computePSNR(), SeedReproducibilityTests, PixArtInt4InMemoryTests, Set, DiTBlock, Float (+2 more)

### Community 16 - "Int4 Dequantization Script"
Cohesion: 0.22
Nodes (14): convert_weights(), load_t5_state_dict(), main(), ndarray, Path, Tensor, quantize_int4(), Load T5-XXL encoder state dict and config from HuggingFace. (+6 more)

### Community 17 - "Attention Layers"
Cohesion: 0.13
Nodes (8): Tests for convert_pixart_weights.build_key_mapping()., 23 global keys + 28 blocks * 23 per-block keys = 667 total., All MLX key values should be unique (no two HF keys map to same MLX key)., Every block index 0..27 should appear in both keys and values., Spot-check that all global key groups exist., Discarded keys should not appear in the mapping table., Block 0 should have all expected sub-keys., TestBuildKeyMapping

### Community 19 - "Weight Apply Tests"
Cohesion: 0.29
Nodes (10): array, copy_config(), dequantize_safetensors(), dequantize_tensor(), main(), ndarray, Path, Copy config.json from input_dir to output_dir if present. (+2 more)

### Community 22 - "Backbone Configuration Tests"
Cohesion: 0.31
Nodes (9): compute_dpm_timesteps(), compute_linear_schedule(), compute_scaled_linear_schedule(), compute_sigmas(), main(), ndarray, Reference implementation matching diffusers get_timestep_embedding for PixArt-Si, Matches DPMSolverScheduler.configure() in Swift. (+1 more)

### Community 23 - "Attention Tests"
Cohesion: 0.28
Nodes (8): Image, compute_channel_stats(), compute_image_stats(), main(), Tensor, Compute per-channel mean/std of a latent tensor [B, C, H, W]., Compute channel means for a PIL RGB image., run_reference()

### Community 26 - "Final Layer Tests"
Cohesion: 0.29
Nodes (3): BackboneFixture, BackboneForwardTests, PixArtDiT

### Community 30 - "DiT Configuration"
Cohesion: 0.60
Nodes (3): PixArtDiTConfiguration, Float, Int

### Community 32 - "Telemetry Reporter"
Cohesion: 0.40
Nodes (3): PixArtTelemetryEvent, NoopPixArtTelemetryReporter, PixArtTelemetryReporter

### Community 33 - "Weight Mapping Definitions"
Cohesion: 0.50
Nodes (3): KeyMapping, PixArtDiT, TensorTransform

### Community 36 - "Quantization Config"
Cohesion: 0.67
Nodes (3): PixArtComponents (Acervo registration), SwiftAcervo Component Registry, ModelRegistry (Acervo assets)

## Ambiguous Edges - Review These
- `AGENTS.md` → `PixArtBackbone Telemetry README`  [AMBIGUOUS]
  Sources/PixArtBackbone/Telemetry/README.md · relation: conceptually_related_to
- `backboneForwardComplete event` → `Instrumentation Requirements (docs/)`  [AMBIGUOUS]
  docs/REQUIREMENTS-instrumentation.md · relation: conceptually_related_to
- `Warm color-cast attribution rationale` → `Failure 2: wrong beta schedule`  [AMBIGUOUS]
  Sources/PixArtBackbone/Telemetry/README.md · relation: conceptually_related_to
- `ditForwardComplete Step-Boundary Emission (Sortie 6)` → `Final Layer / Unpatchify (variance channels)`  [AMBIGUOUS]
  docs/complete/stethoscope-furnace-01/SUPERVISOR_STATE.md · relation: conceptually_related_to

## Knowledge Gaps
- **102 isolated node(s):** `Float`, `Float`, `Bool`, `Bool`, `Int` (+97 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **12 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **What is the exact relationship between `AGENTS.md` and `PixArtBackbone Telemetry README`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **What is the exact relationship between `backboneForwardComplete event` and `Instrumentation Requirements (docs/)`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **What is the exact relationship between `Warm color-cast attribution rationale` and `Failure 2: wrong beta schedule`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **What is the exact relationship between `ditForwardComplete Step-Boundary Emission (Sortie 6)` and `Final Layer / Unpatchify (variance channels)`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **Why does `PixArtRecipe` connect `Process-Wide Telemetry Tests` to `DiT Embedding Layers`, `Beta Schedule Validation`, `Aspect Ratio Bucket Selection`, `VAE Weight Conversion`?**
  _High betweenness centrality (0.062) - this node is a cross-community bridge._
- **Why does `PixArtFP16Recipe` connect `VAE Weight Conversion` to `Process-Wide Telemetry Tests`, `DiT Embedding Layers`?**
  _High betweenness centrality (0.036) - this node is a cross-community bridge._
- **Why does `PixArtDiT` connect `Project Documentation` to `DiT Embedding Layers`?**
  _High betweenness centrality (0.033) - this node is a cross-community bridge._