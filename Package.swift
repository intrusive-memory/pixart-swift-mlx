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
    )
  ],
  dependencies: [
    .package(
      url: "https://github.com/intrusive-memory/SwiftTuberia.git", .upToNextMajor(from: "0.7.2")),
    .package(
      url: "https://github.com/intrusive-memory/SwiftAcervo.git", .upToNextMajor(from: "0.13.1")),
    // Pinned to 0.5.x. swift-tokenizers 0.6.0 switched the Rust binary target
    // from an XCFramework to a UniFFI-based artifactbundle, which breaks the
    // `#if canImport(TokenizersRust)` path under xcodebuild. The 0.6.2 tag
    // ships an explicit "Temporary fix for Xcode builds" commit (37f999a)
    // that the maintainer flagged as a possible Xcode bug — i.e. 0.6.x is not
    // yet stable for Xcode-driven builds. Hold this constraint until a
    // 0.6.x release ships without these Xcode compile issues.
    .package(
      url: "https://github.com/DePasqualeOrg/swift-tokenizers.git",
      .upToNextMinor(from: "0.5.0")),
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
    .testTarget(
      name: "PixArtBackboneTests",
      dependencies: ["PixArtBackbone"]
    ),
  ],
  swiftLanguageModes: [.v6]
)
