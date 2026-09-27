# Graph Report - .  (2026-09-26)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 714 nodes · 1061 edges · 47 communities (33 shown, 14 thin omitted)
- Extraction: 88% EXTRACTED · 12% INFERRED · 0% AMBIGUOUS · INFERRED: 128 edges (avg confidence: 0.8)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `49290d5e`
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
- [[_COMMUNITY_Contributing Guide|Contributing Guide]]
- [[_COMMUNITY_Package Manifest|Package Manifest]]
- [[_COMMUNITY_PixArt Backbone Module|PixArt Backbone Module]]
- [[_COMMUNITY_CI Testing TODO|CI Testing TODO]]
- [[_COMMUNITY_Community 42|Community 42]]
- [[_COMMUNITY_Community 46|Community 46]]

## God Nodes (most connected - your core abstractions)
1. `PixArtRecipe` - 48 edges
2. `PixArtFP16Recipe` - 39 edges
3. `PixArtDiT` - 30 edges
4. `RecipeTests` - 23 edges
5. `PixArtFP16RecipeTests` - 22 edges
6. `MockReporter` - 19 edges
7. `EmbeddingsTests` - 18 edges
8. `PixArtTelemetryEvent` - 15 edges
9. `compute_psnr()` - 13 edges
10. `pixart-swift-mlx` - 13 edges

## Surprising Connections (you probably didn't know these)
- `effectiveReporter` --references--> `PixArtTelemetry`  [EXTRACTED]
  AGENTS.md → Sources/PixArtBackbone/Telemetry/PixArtTelemetry.swift
- `PixArtEventEncoding (SwiftVinetas)` --references--> `PixArtTelemetryEvent`  [EXTRACTED]
  AGENTS.md → Sources/PixArtBackbone/Telemetry/PixArtTelemetryEvent.swift
- `PixArtTelemetry.setReporter` --references--> `PixArtTelemetry`  [EXTRACTED]
  AGENTS.md → Sources/PixArtBackbone/Telemetry/PixArtTelemetry.swift
- `Dual-Seam Telemetry Pattern` --references--> `PixArtTelemetryEvent`  [EXTRACTED]
  AGENTS.md → Sources/PixArtBackbone/Telemetry/PixArtTelemetryEvent.swift
- `Telemetry event guidelines (boundaries not internals, no runID, Sendable)` --references--> `PixArtTelemetryEvent`  [EXTRACTED]
  AGENTS.md → Sources/PixArtBackbone/Telemetry/PixArtTelemetryEvent.swift

## Import Cycles
- None detected.

## Communities (47 total, 14 thin omitted)

### Community 0 - "Process-Wide Telemetry Tests"
Cohesion: 0.05
Nodes (27): build_config(), convert_weights(), load_vae_state_dict(), main(), ndarray, Tensor, Transpose Conv2d weight from PyTorch [O,I,kH,kW] to MLX [O,kH,kW,I]., Load SDXL VAE state dict and config from HuggingFace. (+19 more)

### Community 1 - "DiT Embedding Layers"
Cohesion: 0.07
Nodes (29): AspectRatioEmbedder, Backbone, BackboneInput, Bool, CaptionProjection, Configuration, Conv2d, DiTBlock (+21 more)

### Community 2 - "Int4 Quantization Validation"
Cohesion: 0.10
Nodes (18): Module, AspectRatioEmbedder, CaptionProjection, get2DSinusoidalPositionEmbeddings(), MicroConditionEmbedder, sinusoidalEmbedding1D(), SizeEmbedder, TimestepEmbedder (+10 more)

### Community 3 - "FP16 Recipe Config"
Cohesion: 0.07
Nodes (16): MockReporter, PixArtProcessWideTelemetryTests, PixArtTelemetryAnomalyTests, PixArtTelemetryLockContentionTests, PixArtTelemetryWeightApplyFP16Tests, PixArtTelemetryWeightApplyINT4Tests, PixArtTelemetryEvent, PixArtDiT (+8 more)

### Community 4 - "PixArt DiT Backbone"
Cohesion: 0.07
Nodes (29): ndarray, Tests for int4 quantization and dequantization roundtrip., Quantize then dequantize and return PSNR vs original., Random weights should survive int4 roundtrip with PSNR > 25 dB., Constant weights should survive roundtrip with very high PSNR.          Not exac, All-zero weights should roundtrip exactly., Verify packed weight, scales, biases have correct shapes., PixArt quantizer requires N divisible by GROUP_SIZE. (+21 more)

### Community 5 - "DiT Block & Modulation"
Cohesion: 0.07
Nodes (16): PipelineRecipe, PixArtFP16Recipe, PixArtFP16RecipeTests, Bool, DPMSolverSchedulerConfiguration, Float, Int, PipelineRole (+8 more)

### Community 6 - "Pipeline Recipe Config"
Cohesion: 0.09
Nodes (16): CrossAttention, DiTBlock, GEGLUFFN, FinalLayer, t2iModulate(), ModulationHelperTests, SelfAttention, Float (+8 more)

### Community 7 - "VAE Weight Conversion"
Cohesion: 0.09
Nodes (12): PixArtRecipe, RecipeTests, Bool, DPMSolverSchedulerConfiguration, Float, Int, PixArtDiTConfiguration, SDXLVAEDecoderConfiguration (+4 more)

### Community 8 - "Project Documentation"
Cohesion: 0.09
Nodes (32): AGENTS.md — Universal Agent Documentation, ARCHITECTURE.md — Ecosystem Interface Reference, PixArt Implementation Architecture, DPM-Solver++ Scheduler, PixArt DiT Transformer, SDXL VAE Decoder, T5-XXL Encoder, CLAUDE.md — Claude-Specific Agent Instructions (+24 more)

### Community 9 - "PixArt Weight Conversion"
Cohesion: 0.09
Nodes (23): build_config(), build_key_mapping(), convert_weights(), load_pytorch_state_dict(), main(), ndarray, Tensor, quantize_int4() (+15 more)

### Community 10 - "Telemetry Event Types"
Cohesion: 0.11
Nodes (22): Codable, Float, PixArtInt4InMemoryTests, Set, String, AnomalyKind, inf, nan (+14 more)

### Community 11 - "DPM-Solver Comparison"
Cohesion: 0.15
Nodes (19): diffusers_timesteps(), epsilon_to_x0(), FakeModel, main(), ndarray, Swift epsilon prediction conversion., Run diffusers DPMSolverMultistepScheduler and return per-step latent means., Run Swift DPM-Solver port and return per-step latent means. (+11 more)

### Community 12 - "T5 Weight Conversion"
Cohesion: 0.18
Nodes (16): App Group Configuration (ACERVO_APP_GROUP_ID), ARCHITECTURE.md, Ask the library (SwiftAcervo 0.16 manifest-as-source-of-truth), CDNManifest, Critical Rules for AI Agents, Queryable Codemap (graphify-out), LoRA Target Layer Declarations, Makefile Targets (+8 more)

### Community 13 - "Weight Key Mapping Tests"
Cohesion: 0.22
Nodes (14): convert_weights(), load_t5_state_dict(), main(), ndarray, Path, Tensor, quantize_int4(), Load T5-XXL encoder state dict and config from HuggingFace. (+6 more)

### Community 14 - "Pipeline Integration Tests"
Cohesion: 0.18
Nodes (13): effectiveReporter, PixArtDiT, Telemetry event guidelines (boundaries not internals, no runID, Sendable), PixArtBackbone (product), PrintingReporter, PixArtTelemetryEvent, backboneForwardComplete, errorThrown (+5 more)

### Community 15 - "Aspect Ratio Bucket Selection"
Cohesion: 0.18
Nodes (6): CGImage, Double, computePSNR(), PipelineAssemblyTests, SeedReproducibilityTests, TwoPhaseLoadingTests

### Community 17 - "Attention Layers"
Cohesion: 0.29
Nodes (10): array, copy_config(), dequantize_safetensors(), dequantize_tensor(), main(), ndarray, Path, Copy config.json from input_dir to output_dir if present. (+2 more)

### Community 18 - "Component Registration Tests"
Cohesion: 0.38
Nodes (6): CrossAttention, SelfAttention, Float, Int, Linear, MLXArray

### Community 21 - "Reference Generation Stats"
Cohesion: 0.31
Nodes (9): compute_dpm_timesteps(), compute_linear_schedule(), compute_scaled_linear_schedule(), compute_sigmas(), main(), ndarray, Reference implementation matching diffusers get_timestep_embedding for PixArt-Si, Matches DPMSolverScheduler.configure() in Swift. (+1 more)

### Community 22 - "Backbone Configuration Tests"
Cohesion: 0.28
Nodes (8): Image, compute_channel_stats(), compute_image_stats(), main(), Tensor, Compute per-channel mean/std of a latent tensor [B, C, H, W]., Compute channel means for a PIL RGB image., run_reference()

### Community 23 - "Attention Tests"
Cohesion: 0.29
Nodes (8): CLITelemetryBootstrap, Dual-Seam Telemetry Pattern, SwiftVinetas INSTRUMENTATION_PATTERN.md, MockPixArtReporter, PixArtDiT.setTelemetry, PixArtTelemetry.setReporter, PixArtTelemetryCLIAdapter, PixArtTelemetryReporter

### Community 26 - "Final Layer Tests"
Cohesion: 0.29
Nodes (3): BackboneFixture, BackboneForwardTests, PixArtDiT

### Community 31 - "Micro-Conditioning Skip Tests"
Cohesion: 0.53
Nodes (6): DPMSolver++, ImageRenderer, PixArt Pipeline Recipe, SDXLVAEDecoder, T5XXLEncoder, TuberiaCatalog

### Community 32 - "Telemetry Reporter"
Cohesion: 0.33
Nodes (6): ModelAvailability, PixArtEngine, PixArtEventEncoding (SwiftVinetas), SwiftVinetas, Acervo.availability, Acervo.ensureComponentReady

### Community 33 - "Weight Mapping Definitions"
Cohesion: 0.47
Nodes (4): PixArtFP16Recipe, PixArtRecipe, PixArtTelemetryReporter, PixArtTelemetry

### Community 34 - "Telemetry Singleton"
Cohesion: 0.33
Nodes (4): Error, PixArtRecipeError, shapeMismatch, PixArtTelemetryReporter

### Community 35 - "Pipeline Components"
Cohesion: 0.60
Nodes (3): PixArtDiTConfiguration, Float, Int

### Community 37 - "CI Workflows"
Cohesion: 0.40
Nodes (3): PixArtTelemetryEvent, NoopPixArtTelemetryReporter, PixArtTelemetryReporter

### Community 38 - "Contributing Guide"
Cohesion: 0.50
Nodes (5): Component IDs drop -mlx suffix; acervo ship must pin --slug, ComponentDescriptor, pixart-sigma-xl-dit-fp16, pixart-sigma-xl-dit-int4, PixArtComponents

### Community 39 - "Package Manifest"
Cohesion: 0.50
Nodes (3): KeyMapping, PixArtDiT, TensorTransform

## Knowledge Gaps
- **91 isolated node(s):** `Float`, `Float`, `Bool`, `Bool`, `Int` (+86 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **14 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `PixArtDiT` connect `DiT Embedding Layers` to `Int4 Quantization Validation`?**
  _High betweenness centrality (0.110) - this node is a cross-community bridge._
- **Why does `PixArtRecipe` connect `VAE Weight Conversion` to `Telemetry Singleton`, `Int4 Quantization Validation`, `DiT Block & Modulation`, `CI Testing TODO`, `Aspect Ratio Bucket Selection`, `Weight Apply Tests`, `Recipe Validation Errors`?**
  _High betweenness centrality (0.099) - this node is a cross-community bridge._
- **Why does `PixArtTelemetryEvent` connect `Pipeline Integration Tests` to `Telemetry Reporter`, `Telemetry Event Types`, `Int4 Quantization Validation`, `Attention Tests`?**
  _High betweenness centrality (0.082) - this node is a cross-community bridge._
- **Are the 31 inferred relationships involving `PixArtRecipe` (e.g. with `.allThreeComponentIds()` and `.recipAssemblesIntoPipeline()`) actually correct?**
  _`PixArtRecipe` has 31 INFERRED edges - model-reasoned connections that need verification._
- **Are the 22 inferred relationships involving `PixArtFP16Recipe` (e.g. with `.allComponentIdsContainsFP16Backbone()` and `.allComponentIdsExactOrder()`) actually correct?**
  _`PixArtFP16Recipe` has 22 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Float`, `Float`, `Bool` to the rest of the system?**
  _160 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Process-Wide Telemetry Tests` be split into smaller, more focused modules?**
  _Cohesion score 0.0467687074829932 - nodes in this community are weakly interconnected._