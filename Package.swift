// swift-tools-version: 6.2

import PackageDescription
import Foundation

// In CI we always pin to released remotes. Locally, prefer a sibling checkout
// at ../<name> if present so in-flight changes can be exercised end-to-end
// without publishing a release. Falls back to the remote pin if the sibling
// directory is missing, so fresh clones still build.
let useLocalSiblings = ProcessInfo.processInfo.environment["CI"] != "true"

func sibling(_ name: String, remote: String, from version: Version) -> Package.Dependency {
  let localPath = "../\(name)"
  if useLocalSiblings && FileManager.default.fileExists(atPath: localPath) {
    return .package(path: localPath)
  }
  return .package(url: remote, from: version)
}

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
    sibling(
      "SwiftTuberia",
      remote: "https://github.com/intrusive-memory/SwiftTuberia.git",
      from: "0.6.0"),
    sibling(
      "SwiftAcervo",
      remote: "https://github.com/intrusive-memory/SwiftAcervo.git",
      from: "0.8.3"),
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
