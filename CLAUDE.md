# Claude-Specific Agent Instructions

**⚠️ Read [AGENTS.md](AGENTS.md) first** for universal project documentation.

This file contains instructions specific to Claude Code agents.

## Build Preferences

- NEVER use `swift build` or `swift test` — use `make` targets or `xcodebuild`
- Use `make build`, `make test`, `make lint` for standard operations
- See [AGENTS.md](AGENTS.md) for full list of Makefile targets

## Key Components

- PixArt-Sigma DiT backbone (Backbone protocol conformance)
- Weight key mapping (PyTorch -> MLX safetensors)
- Pipeline recipe (T5 + DPM + PixArtDiT + SDXL VAE + ImageRenderer)
- Acervo component descriptors
- CLI tool for image generation (`PixArtCLI`)

## Claude-Specific Critical Rules

1. NEVER use `swift build` or `swift test` — always use Makefile targets
2. ONLY supports iOS 26.0+ and macOS 26.0+ (NEVER add code for older platforms)
3. ALWAYS run `make lint` before committing
4. See [AGENTS.md](AGENTS.md) for universal rules
