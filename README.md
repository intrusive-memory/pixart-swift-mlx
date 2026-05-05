# pixart-swift-mlx

PixArt-Sigma model plugin for [SwiftTuberia](https://github.com/intrusive-memory/SwiftTuberia).

## Overview

Provides the PixArt-Sigma DiT (Diffusion Transformer) backbone for the SwiftTuberia pipeline system. ~400 lines of model-specific code — all infrastructure comes from SwiftTuberia.

### Features

- PixArt-Sigma DiT backbone (~28 blocks, 1152 hidden dim, ~600M params)
- Weight key mapping (PyTorch → MLX safetensors)
- Acervo component descriptors for model registration
- Pipeline recipe (T5 + DPM + PixArtDiT + SDXL VAE + ImageRenderer)
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
    .package(url: "https://github.com/intrusive-memory/pixart-swift-mlx.git", from: "0.6.0")
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
make test      # Run Swift tests
make test-all  # Run all tests (Swift + Python)
make lint      # Format Swift sources
make help      # Show all targets
```

## App Group configuration (required)

This package depends on [SwiftAcervo](https://github.com/intrusive-memory/SwiftAcervo) for shared model storage. SwiftAcervo v0.10.0 resolves its App Group ID in this order: `ACERVO_APP_GROUP_ID` env var → `com.apple.security.application-groups` entitlement (macOS only) → `fatalError`. There is **no silent fallback**.

- **Signed UI apps (macOS / iOS)**: declare `com.apple.security.application-groups` with `group.intrusive-memory.models` in your `.entitlements` file. iOS apps additionally need `ACERVO_APP_GROUP_ID=group.intrusive-memory.models` in the launch environment.
- **Scripts, CI jobs, test runners**: export `ACERVO_APP_GROUP_ID=group.intrusive-memory.models` in the shell or job environment. The standard place is `~/.zprofile`:

    ```sh
    export ACERVO_APP_GROUP_ID=group.intrusive-memory.models
    ```

Without this, `Acervo.sharedModelsDirectory` traps with `fatalError`. See [SwiftAcervo's USAGE.md](https://github.com/intrusive-memory/SwiftAcervo/blob/main/USAGE.md) for full details.

## License

MIT License. See [LICENSE](LICENSE) for details.
