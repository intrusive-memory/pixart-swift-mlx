// swift-tools-version: 6.2

import PackageDescription

let package = Package(
  name: "pixart-swift-mlx",
  platforms: [
    .macOS(.v26),
    .iOS(.v26),
  ],
  products: [
    .library(
      name: "PixArtBackbone",
      targets: ["PixArtBackbone"]
    ),
    .executable(
      name: "PixArtCLI",
      targets: ["PixArtCLI"]
    ),
  ],
  dependencies: [
    .package(url: "https://github.com/intrusive-memory/SwiftTuberia.git", from: "0.4.0"),
    .package(url: "https://github.com/intrusive-memory/SwiftAcervo.git", from: "0.7.2"),
    .package(url: "https://github.com/apple/swift-argument-parser", from: "1.7.1"),
  ],
  targets: [
    .target(
      name: "PixArtBackbone",
      dependencies: [
        .product(name: "Tuberia", package: "SwiftTuberia"),
        .product(name: "TuberiaCatalog", package: "SwiftTuberia"),
        .product(name: "SwiftAcervo", package: "SwiftAcervo"),
      ]
    ),
    .executableTarget(
      name: "PixArtCLI",
      dependencies: [
        "PixArtBackbone",
        .product(name: "SwiftAcervo", package: "SwiftAcervo"),
        .product(name: "ArgumentParser", package: "swift-argument-parser"),
      ]
    ),
    .testTarget(
      name: "PixArtBackboneTests",
      dependencies: ["PixArtBackbone"]
    ),
  ],
  swiftLanguageModes: [.v6]
)
