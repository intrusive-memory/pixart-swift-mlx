# pixart-swift-mlx

PixArt-Sigma model plugin for [SwiftTuberia](https://github.com/intrusive-memory/SwiftTuberia).

## Overview

Provides the PixArt-Sigma DiT (Diffusion Transformer) backbone for the SwiftTuberia pipeline system. ~400 lines of model-specific code — all infrastructure comes from SwiftTuberia.

### Features

- PixArt-Sigma DiT backbone (~28 blocks, 1152 hidden dim, ~600M params)
- Weight key mapping (PyTorch → MLX safetensors)
- Acervo component descriptors for model registration
- Pipeline recipe (T5 + DPM + PixArtDiT + SDXL VAE + ImageRenderer)
- CLI tool for image generation (`PixArtCLI`)
- Weight conversion scripts (PyTorch -> int4/fp16 MLX safetensors)
- ~2 GB total pipeline (int4 quantized), iPad-viable

## Requirements

- macOS 26.0+ / iOS 26.0+
- Swift 6.2+
- Xcode 26+
- Apple Silicon (M1+)

## Installation

Add to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/intrusive-memory/pixart-swift-mlx.git", from: "0.5.1")
]
```

Then add the dependency to your target:

```swift
.target(name: "YourTarget", dependencies: [
    .product(name: "PixArtBackbone", package: "pixart-swift-mlx")
])
```

## Building

```bash
make build     # Debug build
make install   # Build + install PixArtCLI to ./bin
make test      # Run Swift tests
make test-all  # Run all tests (Swift + Python)
make lint      # Format Swift sources
make help      # Show all targets
```

## License

MIT License. See [LICENSE](LICENSE) for details.
