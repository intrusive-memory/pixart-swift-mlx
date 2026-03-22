# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

For detailed project documentation, see **[AGENTS.md](AGENTS.md)**.

## Quick Reference

**Project**: pixart-swift-mlx - PixArt-Sigma model plugin for SwiftTuberia

**Platforms**: iOS 26.0+, macOS 26.0+

**Key Components**:
- PixArt-Sigma DiT backbone (Backbone protocol conformance)
- Weight key mapping (PyTorch -> MLX safetensors)
- Pipeline recipe (T5 + DPM + PixArtDiT + SDXL VAE + ImageRenderer)
- Acervo component descriptors
- CLI tool for image generation

**Important Notes**:
- ONLY supports iOS 26.0+ and macOS 26.0+ (NEVER add code for older platforms)
- ~400 lines of model-specific code — everything else comes from SwiftTuberia
- ~2 GB total (int4), iPad-viable
- See [AGENTS.md](AGENTS.md) for complete documentation
- See [REQUIREMENTS.md](REQUIREMENTS.md) for full specification
