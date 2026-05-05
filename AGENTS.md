# AGENTS.md

This file provides comprehensive documentation for AI agents working with the pixart-swift-mlx codebase.

**Version**: 0.5.3-dev
**Purpose**: Guide AI agents working on pixart-swift-mlx
**Audience**: Claude Code, Gemini, and other AI development assistants

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

This project uses a Makefile. Available targets:

```bash
make resolve      # Resolve SPM dependencies
make build        # Debug build
make test         # Run Swift unit tests
make test-python  # Run Python conversion script tests
make test-all     # Run all tests (Swift + Python)
make lint         # Format Swift sources with swift-format
make clean        # Remove build artifacts and DerivedData
make help         # Show all targets
```

## Critical Rules for AI Agents

1. NEVER commit directly to `main` — use `development` branch
2. ONLY support iOS 26.0+ and macOS 26.0+ (NEVER add code for older platforms)
3. ALWAYS run `make lint` before committing
4. ALWAYS read files before editing
5. NEVER create files unless necessary
6. Follow agent-specific instructions — see [CLAUDE.md](CLAUDE.md) or [GEMINI.md](GEMINI.md)

## App Group configuration (required)

This package depends on [SwiftAcervo](https://github.com/intrusive-memory/SwiftAcervo) for shared model storage. SwiftAcervo v0.10.0 resolves its App Group ID in this order: `ACERVO_APP_GROUP_ID` env var → `com.apple.security.application-groups` entitlement (macOS only) → `fatalError`. There is **no silent fallback**.

- **Signed UI apps (macOS / iOS)**: declare `com.apple.security.application-groups` with `group.intrusive-memory.models` in your `.entitlements` file. iOS apps additionally need `ACERVO_APP_GROUP_ID=group.intrusive-memory.models` in the launch environment.
- **Scripts, CI jobs, test runners**: export `ACERVO_APP_GROUP_ID=group.intrusive-memory.models` in the shell or job environment. The standard place is `~/.zprofile`:

    ```sh
    export ACERVO_APP_GROUP_ID=group.intrusive-memory.models
    ```

Without this, `Acervo.sharedModelsDirectory` traps with `fatalError`. See [SwiftAcervo's USAGE.md](https://github.com/intrusive-memory/SwiftAcervo/blob/main/USAGE.md) for full details.

## Documentation Index

- [AGENTS.md](AGENTS.md) — Universal agent documentation (this file)
- [CLAUDE.md](CLAUDE.md) — Claude-specific instructions
- [GEMINI.md](GEMINI.md) — Gemini-specific instructions
- [REQUIREMENTS.md](REQUIREMENTS.md) — Full specification
- [ARCHITECTURE.md](ARCHITECTURE.md) — Detailed architecture notes
- [README.md](README.md) — User-facing documentation
