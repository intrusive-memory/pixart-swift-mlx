# AGENTS.md

This file provides comprehensive documentation for AI agents working with the pixart-swift-mlx codebase.

**Status**: Pre-implementation — see [REQUIREMENTS.md](REQUIREMENTS.md) for full specification.

---

## Project Overview

pixart-swift-mlx is a model plugin for SwiftTuberia providing the PixArt-Sigma DiT (Diffusion Transformer) backbone. It contributes only the model-specific delta — the unique neural network architecture, weight key mapping, configuration, and pipeline recipe. All infrastructure comes from SwiftTuberia.

## Architecture

**Pipeline recipe**:
```
T5XXLEncoder (catalog) -> PixArtDiT (this repo) -> SDXLVAEDecoder (catalog) -> ImageRenderer (catalog)
                              ^
                       DPMSolver++ (catalog)
```

**This repo provides**:
- PixArt-Sigma DiT backbone (~28 blocks, 1152 hidden dim, ~600M params)
- Weight key mapping (~200 keys, PyTorch -> MLX safetensors)
- Acervo component descriptors for model registration
- LoRA target layer declarations
- Pipeline recipe assembly
- Weight conversion scripts
- CLI tool

**~400 lines of model-specific code total.**

## Dependencies

- `SwiftTuberia` (Tuberia + TuberiaCatalog) — pipeline protocols + shared components
- `SwiftAcervo` (transitive via SwiftTuberia) — model registry

## Platform Requirements

- iOS 26.0+, macOS 26.0+ exclusively
- Swift 6.2+, Xcode 26+
- Apple Silicon only (M1+)
- ~2 GB total (int4 quantized), iPad-viable

## Build and Test

```bash
xcodebuild build -scheme pixart-swift-mlx -destination 'platform=macOS'
xcodebuild test -scheme pixart-swift-mlx -destination 'platform=macOS'
```

See [REQUIREMENTS.md](REQUIREMENTS.md) for the complete specification.
