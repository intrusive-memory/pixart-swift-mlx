# Graph Report - .  (2026-11-22)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 794 nodes · 1181 edges · 54 communities (42 shown, 12 thin omitted)
- Extraction: 89% EXTRACTED · 10% INFERRED · 0% AMBIGUOUS · INFERRED: 124 edges (avg confidence: 0.8)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `1848da15`
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
- [[_COMMUNITY_Quantization Config|Quantization Config]]
- [[_COMMUNITY_CI Workflows|CI Workflows]]
- [[_COMMUNITY_Contributing Guide|Contributing Guide]]
- [[_COMMUNITY_Package Manifest|Package Manifest]]
- [[_COMMUNITY_PixArt Backbone Module|PixArt Backbone Module]]
- [[_COMMUNITY_CI Testing TODO|CI Testing TODO]]
- [[_COMMUNITY_Community 42|Community 42]]
- [[_COMMUNITY_Community 43|Community 43]]
- [[_COMMUNITY_Community 44|Community 44]]
- [[_COMMUNITY_Community 45|Community 45]]
- [[_COMMUNITY_Community 46|Community 46]]
- [[_COMMUNITY_Community 47|Community 47]]
- [[_COMMUNITY_Community 48|Community 48]]
- [[_COMMUNITY_Community 49|Community 49]]
- [[_COMMUNITY_Community 50|Community 50]]
- [[_COMMUNITY_Community 51|Community 51]]
- [[_COMMUNITY_Community 52|Community 52]]

## God Nodes (most connected - your core abstractions)
1. `PixArtRecipe` - 48 edges
2. `PixArtFP16Recipe` - 39 edges
3. `PixArtDiT` - 24 edges
4. `RecipeTests` - 23 edges
5. `PixArtFP16RecipeTests` - 22 edges
6. `MockReporter` - 19 edges
7. `EmbeddingsTests` - 18 edges
8. `PixArtDiT` - 17 edges
9. `compute_psnr()` - 13 edges
10. `PixArtDiT` - 13 edges

## Surprising Connections (you probably didn't know these)
- `Tests workflow (macOS + coverage)` --conceptually_related_to--> `Pass 2 — Test suite improvements`  [INFERRED]
  .github/workflows/tests.yml → docs/complete/REQUIREMENTS.md
- `PixArtDiT` --implements--> `Backbone protocol`  [EXTRACTED]
  AGENTS.md → ARCHITECTURE.md
- `PixArtDiT` --calls--> `PixArtDiT.anomalyKind(for:)`  [EXTRACTED]
  AGENTS.md → docs/REQUIREMENTS-instrumentation.md
- `backboneForwardComplete event` --conceptually_related_to--> `PixArtTelemetryEvent`  [AMBIGUOUS]
  Sources/PixArtBackbone/Telemetry/README.md → AGENTS.md
- `Failure 3: position embedding scale (baseSize/peInterpolation)` --conceptually_related_to--> `resolutionBins1024 / binnedResolution(width:height:)`  [AMBIGUOUS]
  docs/complete/sigma-foundry-diagnosis.md → AGENTS.md

## Import Cycles
- None detected.

## Communities (54 total, 12 thin omitted)

### Community 0 - "Process-Wide Telemetry Tests"
Cohesion: 0.07
Nodes (17): MockReporter, PixArtProcessWideTelemetryTests, PixArtTelemetryAnomalyTests, PixArtTelemetryLockContentionTests, PixArtTelemetryWeightApplyFP16Tests, PixArtTelemetryWeightApplyINT4Tests, PixArtTelemetryReporter, PixArtTelemetryEvent (+9 more)

### Community 1 - "DiT Embedding Layers"
Cohesion: 0.10
Nodes (18): Module, AspectRatioEmbedder, CaptionProjection, get2DSinusoidalPositionEmbeddings(), MicroConditionEmbedder, sinusoidalEmbedding1D(), SizeEmbedder, TimestepEmbedder (+10 more)

### Community 2 - "Int4 Quantization Validation"
Cohesion: 0.07
Nodes (29): ndarray, Tests for int4 quantization and dequantization roundtrip., Quantize then dequantize and return PSNR vs original., Random weights should survive int4 roundtrip with PSNR > 25 dB., Constant weights should survive roundtrip with very high PSNR.          Not exac, All-zero weights should roundtrip exactly., Verify packed weight, scales, biases have correct shapes., PixArt quantizer requires N divisible by GROUP_SIZE. (+21 more)

### Community 3 - "FP16 Recipe Config"
Cohesion: 0.07
Nodes (16): PipelineRecipe, PixArtFP16Recipe, PixArtFP16RecipeTests, Bool, DPMSolverSchedulerConfiguration, Float, Int, PipelineRole (+8 more)

### Community 4 - "PixArt DiT Backbone"
Cohesion: 0.08
Nodes (21): AspectRatioEmbedder, Backbone, CaptionProjection, Configuration, Conv2d, FinalLayer, PixArtDiT, PatchEmbeddingTests (+13 more)

### Community 5 - "DiT Block & Modulation"
Cohesion: 0.09
Nodes (16): CrossAttention, DiTBlock, GEGLUFFN, FinalLayer, t2iModulate(), ModulationHelperTests, SelfAttention, Float (+8 more)

### Community 6 - "Pipeline Recipe Config"
Cohesion: 0.09
Nodes (12): PixArtRecipe, RecipeTests, Bool, DPMSolverSchedulerConfiguration, Float, Int, PixArtDiTConfiguration, SDXLVAEDecoderConfiguration (+4 more)

### Community 7 - "VAE Weight Conversion"
Cohesion: 0.07
Nodes (19): build_config(), convert_weights(), load_vae_state_dict(), main(), ndarray, Tensor, Transpose Conv2d weight from PyTorch [O,I,kH,kW] to MLX [O,kH,kW,I]., Load SDXL VAE state dict and config from HuggingFace. (+11 more)

### Community 8 - "Project Documentation"
Cohesion: 0.09
Nodes (23): build_config(), build_key_mapping(), convert_weights(), load_pytorch_state_dict(), main(), ndarray, Tensor, quantize_int4() (+15 more)

### Community 9 - "PixArt Weight Conversion"
Cohesion: 0.09
Nodes (25): Codable, String, AnomalyKind, inf, nan, outOfRange, zeroLatent, AnomalyPhase (+17 more)

### Community 10 - "Telemetry Event Types"
Cohesion: 0.15
Nodes (19): diffusers_timesteps(), epsilon_to_x0(), FakeModel, main(), ndarray, Swift epsilon prediction conversion., Run diffusers DPMSolverMultistepScheduler and return per-step latent means., Run Swift DPM-Solver port and return per-step latent means. (+11 more)

### Community 11 - "DPM-Solver Comparison"
Cohesion: 0.12
Nodes (6): PixArtComponents, ComponentRegistrationTests, PixArtComponents (Acervo registration), ModelRegistry, Bool, SwiftAcervo

### Community 12 - "T5 Weight Conversion"
Cohesion: 0.16
Nodes (13): PixArt-Alpha paper (arXiv:2310.00426), PixArt-Sigma paper (arXiv:2403.04692), Incomplete/Archived Documentation README, PixArtCLI, Testing Strategy (no timed tests, 90% coverage), diffusers==0.32.2 pin, FLUX.2 vs PixArt-Sigma comparison, PixArtCore (+5 more)

### Community 13 - "Weight Key Mapping Tests"
Cohesion: 0.19
Nodes (17): PixArtTelemetryEvent, REQUIREMENTS-instrumentation.md (archived, shipped form), STETHOSCOPE FURNACE execution plan, OPERATION STETHOSCOPE FURNACE brief, Coalesced single-Task event dispatch pattern, RECONCILIATION.md (brief vs shipped), Commit b585120 telemetry pare-down, REQUIREMENTS-instrumentation.md (docs/) (+9 more)

### Community 14 - "Pipeline Integration Tests"
Cohesion: 0.15
Nodes (15): Final Layer / Unpatchify, Patch Embedding, 2D Sinusoidal Position Embedding, Variance channel discard (pred_sigma), PixArtDiT, Backbone protocol, BackboneInput, WeightedSegment protocol (+7 more)

### Community 15 - "Aspect Ratio Bucket Selection"
Cohesion: 0.22
Nodes (14): convert_weights(), load_t5_state_dict(), main(), ndarray, Path, Tensor, quantize_int4(), Load T5-XXL encoder state dict and config from HuggingFace. (+6 more)

### Community 16 - "Int4 Dequantization Script"
Cohesion: 0.13
Nodes (8): Tests for convert_pixart_weights.build_key_mapping()., 23 global keys + 28 blocks * 23 per-block keys = 667 total., All MLX key values should be unique (no two HF keys map to same MLX key)., Every block index 0..27 should appear in both keys and values., Spot-check that all global key groups exist., Discarded keys should not appear in the mapping table., Block 0 should have all expected sub-keys., TestBuildKeyMapping

### Community 17 - "Attention Layers"
Cohesion: 0.22
Nodes (13): DPMSolverScheduler, ImageRenderer, End-to-end pipeline data flow, PipelineRecipe protocol, PixArtRecipe, SDXLVAEDecoder, T5XXLEncoder, Micro-conditioning skip regression pin (+5 more)

### Community 18 - "Component Registration Tests"
Cohesion: 0.18
Nodes (6): CGImage, computePSNR(), PipelineAssemblyTests, SeedReproducibilityTests, TwoPhaseLoadingTests, Double

### Community 20 - "Beta Schedule Validation"
Cohesion: 0.16
Nodes (16): Dual-seam telemetry pattern, PixArtDiT, PixArtDiT.setTelemetry(_:), SwiftVinetas, BackboneInput, Weight key mapping (~200 keys), WeightedSegment protocol, OPERATION PIXEL FORGE brief (+8 more)

### Community 21 - "Reference Generation Stats"
Cohesion: 0.23
Nodes (12): Linear beta schedule (not shifted cosine), VAE Mid-Block Attention, VAE ResNet Block, VAE Nearest-Neighbor Upsampling, DPM-Solver++ Scheduler, PixArtRecipe, ImageRenderer, PixArtBackbone library (+4 more)

### Community 22 - "Backbone Configuration Tests"
Cohesion: 0.32
Nodes (6): PixArtInt4InMemoryTests, Set, DiTBlock, Float, MLXArray, String

### Community 23 - "Attention Tests"
Cohesion: 0.20
Nodes (10): PixArtComponents, Component id vs HF repo slug rule, CatalogRegistration.shared, ComponentDescriptor, Memory profiles (~2 GB int4), REQUIREMENTS.md (archived, 3 passes), Pass 3 — PixArtCLI removal, PixArtCLI (removed) (+2 more)

### Community 24 - "Backbone Forward Pass Tests"
Cohesion: 0.20
Nodes (11): AdaLN-Single / AdaLN-Zero Conditioning, CaptionProjection, DiT Cross-Attention, PixArt DiT Block, Micro-conditions (csize/ar embedders), scale_shift_table, DiT Self-Attention, t2i_modulate (+3 more)

### Community 25 - "DiT Block Tests"
Cohesion: 0.29
Nodes (10): array, copy_config(), dequantize_safetensors(), dequantize_tensor(), main(), ndarray, Path, Copy config.json from input_dir to output_dir if present. (+2 more)

### Community 26 - "Final Layer Tests"
Cohesion: 0.38
Nodes (6): CrossAttention, SelfAttention, Float, Int, Linear, MLXArray

### Community 28 - "Weight Mapping Tests"
Cohesion: 0.31
Nodes (9): compute_dpm_timesteps(), compute_linear_schedule(), compute_scaled_linear_schedule(), compute_sigmas(), main(), ndarray, Reference implementation matching diffusers get_timestep_embedding for PixArt-Si, Matches DPMSolverScheduler.configure() in Swift. (+1 more)

### Community 29 - "Recipe Validation Errors"
Cohesion: 0.28
Nodes (8): Image, compute_channel_stats(), compute_image_stats(), main(), Tensor, Compute per-channel mean/std of a latent tensor [B, C, H, W]., Compute channel means for a PIL RGB image., run_reference()

### Community 30 - "DiT Configuration"
Cohesion: 0.40
Nodes (6): effectiveReporter, PixArtTelemetry (process-wide seam), PixArtTelemetry.setReporter(_:), PrintingReporter example, TODO: Process-wide telemetry seam (resolved), CLITelemetryBootstrap

### Community 31 - "Micro-Conditioning Skip Tests"
Cohesion: 0.25
Nodes (8): Backbone protocol, Fresh PixArtDiT per test (fixture contamination), Sync forward vs async capture mismatch, MockReporter test helper, MyPixArtReporter example, NoopPixArtTelemetryReporter, PixArtTelemetryReporter, PixArtTelemetryCLIAdapter

### Community 32 - "Telemetry Reporter"
Cohesion: 0.46
Nodes (4): PixArtDiTConfiguration, Double, Float, Int

### Community 34 - "Telemetry Singleton"
Cohesion: 0.38
Nodes (7): App Group config (ACERVO_APP_GROUP_ID), SwiftAcervo 0.16 'ask the library' contract, ModelAvailability (4 cases incl. .partial), SwiftAcervo, Build for iOS workflow, Tests workflow (macOS + coverage), TODO.md — SwiftAcervo 0.16.0 upgrade (archived)

### Community 36 - "Quantization Config"
Cohesion: 0.29
Nodes (6): pixart-swift-mlx, SwiftTuberia, Commit conventions, WeightLoader, swift-tokenizers 0.5.x pin constraint, SwiftTuberia 0.7.0 Pin Bump (Sortie 0)

### Community 38 - "Contributing Guide"
Cohesion: 0.29
Nodes (3): BackboneFixture, BackboneForwardTests, PixArtDiT

### Community 43 - "Community 43"
Cohesion: 0.31
Nodes (5): graphify-out codemap, Makefile targets (build/test/lint), resolutionBins1024 / binnedResolution(width:height:), PixArtDiTConfiguration, Failure 3: position embedding scale (baseSize/peInterpolation)

### Community 44 - "Community 44"
Cohesion: 0.33
Nodes (6): GeGLU FFN, T5 Relative Position Bias, T5 RMSNorm, T5 SentencePiece Tokenizer, T5 Unscaled Attention, T5-XXL Encoder

### Community 45 - "Community 45"
Cohesion: 0.33
Nodes (4): Error, PixArtRecipeError, shapeMismatch, PixArtTelemetryReporter

### Community 47 - "Community 47"
Cohesion: 0.33
Nodes (6): PixArtConfig, PixArtGenerationResult, PixArtLoRAConfig, PixArtMemoryOptimization, PixArtPipeline actor, Two-phase loading (iPad memory strategy)

### Community 48 - "Community 48"
Cohesion: 0.40
Nodes (3): PixArtTelemetryEvent, NoopPixArtTelemetryReporter, PixArtTelemetryReporter

### Community 50 - "Community 50"
Cohesion: 0.50
Nodes (3): KeyMapping, PixArtDiT, TensorTransform

## Ambiguous Edges - Review These
- `PixArtTelemetryEvent` → `backboneForwardComplete event`  [AMBIGUOUS]
  Sources/PixArtBackbone/Telemetry/README.md · relation: conceptually_related_to
- `resolutionBins1024 / binnedResolution(width:height:)` → `Failure 3: position embedding scale (baseSize/peInterpolation)`  [AMBIGUOUS]
  docs/complete/sigma-foundry-diagnosis.md · relation: conceptually_related_to
- `backboneForwardComplete event` → `Commit b585120 telemetry pare-down`  [AMBIGUOUS]
  Sources/PixArtBackbone/Telemetry/README.md · relation: conceptually_related_to
- `Baseline overhead test (Sortie 9)` → `Testing Strategy (no timed tests, 90% coverage)`  [AMBIGUOUS]
  docs/complete/stethoscope-furnace-01/SUPERVISOR_STATE.md · relation: conceptually_related_to
- `Linear beta schedule (not shifted cosine)` → `REQUIREMENTS_STANDALONE.md`  [AMBIGUOUS]
  docs/incomplete/REQUIREMENTS_STANDALONE.md · relation: conceptually_related_to

## Knowledge Gaps
- **107 isolated node(s):** `Float`, `Float`, `Bool`, `Bool`, `Int` (+102 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **12 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **What is the exact relationship between `PixArtTelemetryEvent` and `backboneForwardComplete event`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **What is the exact relationship between `resolutionBins1024 / binnedResolution(width:height:)` and `Failure 3: position embedding scale (baseSize/peInterpolation)`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **What is the exact relationship between `backboneForwardComplete event` and `Commit b585120 telemetry pare-down`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **What is the exact relationship between `Baseline overhead test (Sortie 9)` and `Testing Strategy (no timed tests, 90% coverage)`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **What is the exact relationship between `Linear beta schedule (not shifted cosine)` and `REQUIREMENTS_STANDALONE.md`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **Why does `PixArtRecipe` connect `Pipeline Recipe Config` to `DiT Embedding Layers`, `FP16 Recipe Config`, `CI Testing TODO`, `DPM-Solver Comparison`, `Community 45`, `Component Registration Tests`, `Community 52`?**
  _High betweenness centrality (0.060) - this node is a cross-community bridge._
- **Why does `PixArtFP16Recipe` connect `FP16 Recipe Config` to `DiT Embedding Layers`, `CI Testing TODO`?**
  _High betweenness centrality (0.036) - this node is a cross-community bridge._