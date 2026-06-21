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
      url: "https://github.com/intrusive-memory/SwiftTuberia.git", .upToNextMajor(from: "0.7.7")),
    .package(
      url: "https://github.com/intrusive-memory/SwiftAcervo.git", .upToNextMajor(from: "0.20.0")),
    // 0.7.1 carries upstream 0.6.3's "Fixes for Xcode build with artifact
    // bundle", so the UniFFI artifactbundle links cleanly under xcodebuild (the
    // old RustBuffer/module-map blocker that held this at 0.5.x is resolved).
    // Kept in lockstep with SwiftTuberia's tokenizers constraint.
    .package(
      url: "https://github.com/DePasqualeOrg/swift-tokenizers.git",
      .upToNextMinor(from: "0.7.1")),
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
