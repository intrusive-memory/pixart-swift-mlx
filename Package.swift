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
            name: "PixArtMLX",
            targets: ["PixArtMLX"]
        )
    ],
    dependencies: [
        .package(url: "https://github.com/intrusive-memory/SwiftAcervo.git", branch: "main")
    ],
    targets: [
        .target(
            name: "PixArtMLX",
            dependencies: [
                .product(name: "SwiftAcervo", package: "SwiftAcervo")
            ]
        ),
        .testTarget(
            name: "PixArtMLXTests",
            dependencies: ["PixArtMLX"]
        )
    ],
    swiftLanguageModes: [.v6]
)
