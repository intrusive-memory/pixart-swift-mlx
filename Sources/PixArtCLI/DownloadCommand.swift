import ArgumentParser
import Foundation
import PixArtBackbone
import SwiftAcervo
import TuberiaCatalog

struct DownloadCommand: AsyncParsableCommand {
  static let configuration = CommandConfiguration(
    commandName: "download",
    abstract: "Download all PixArt-Sigma model components"
  )

  @Flag(name: .long, help: "Re-download even if components are already present")
  var force: Bool = false

  mutating func run() async throws {
    _ = PixArtComponents.registered

    let registry = CatalogRegistration.shared
    let descriptors = registry.registeredDescriptors()
      .sorted { $0.componentId < $1.componentId }

    print("Downloading \(descriptors.count) component(s)...")

    for descriptor in descriptors {
      print("\n[\(descriptor.componentId)]")
      print("  Repo: \(descriptor.huggingFaceRepo)")
      let sizeGB = Double(descriptor.estimatedSizeBytes) / 1_073_741_824
      print(String(format: "  Size: ~%.1f GB", sizeGB))

      let isAvailable = Acervo.isModelAvailable(descriptor.huggingFaceRepo)
      if isAvailable && !force {
        print("  Status: already downloaded, skipping")
        continue
      }

      print("  Downloading...")
      try await Acervo.ensureAvailable(
        descriptor.huggingFaceRepo,
        files: descriptor.filePatterns
      ) { progress in
        let pct = Int(progress.overallProgress * 100)
        print("  \(progress.fileName): \(pct)%", terminator: "\r")
        fflush(stdout)
      }
      print("  Done.                              ")
    }

    print("\nAll components downloaded.")
  }
}
