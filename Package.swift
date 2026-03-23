// swift-tools-version: 6.2

import PackageDescription

let package = Package(
    name: "pixart-swift-mlx",
    platforms: [
        .macOS(.v26),
        .iOS(.v26)
    ],
    products: [
        .library(
            name: "PixArtBackbone",
            targets: ["PixArtBackbone"]
        ),
        .executable(
            name: "PixArtCLI",
            targets: ["PixArtCLI"]
        )
    ],
    dependencies: [
        .package(path: "../SwiftTubería"),
        .package(url: "https://github.com/apple/swift-argument-parser", from: "1.3.0")
    ],
    targets: [
        .target(
            name: "PixArtBackbone",
            dependencies: [
                .product(name: "Tuberia", package: "SwiftTubería"),
                .product(name: "TuberiaCatalog", package: "SwiftTubería")
            ]
        ),
        .executableTarget(
            name: "PixArtCLI",
            dependencies: [
                "PixArtBackbone",
                .product(name: "ArgumentParser", package: "swift-argument-parser")
            ]
        ),
        .testTarget(
            name: "PixArtBackboneTests",
            dependencies: ["PixArtBackbone"]
        )
    ],
    swiftLanguageModes: [.v6]
)
