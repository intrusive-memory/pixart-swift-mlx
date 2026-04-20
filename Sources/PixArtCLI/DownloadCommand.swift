import ArgumentParser
import Foundation
import PixArtBackbone
import SwiftAcervo

struct DownloadCommand: AsyncParsableCommand {
  static let configuration = CommandConfiguration(
    commandName: "download",
    abstract: "Download all PixArt-Sigma model components"
  )

  @Flag(name: .long, help: "Re-download even if components are already present")
  var force: Bool = false

  mutating func run() async throws {
    _ = PixArtComponents.registered

    let recipe = PixArtRecipe()
    let componentIds = recipe.allComponentIds.sorted()

    print("Downloading \(componentIds.count) component(s)...")

    for componentId in componentIds {
      guard let descriptor = Acervo.component(componentId) else {
        print("\n[\(componentId)] (descriptor not registered, skipping)")
        continue
      }

      print("\n[\(descriptor.id)]")
      print("  Repo: \(descriptor.repoId)")
      let sizeGB = Double(descriptor.estimatedSizeBytes) / 1_073_741_824
      print(String(format: "  Size: ~%.1f GB", sizeGB))

      if Acervo.isComponentReady(descriptor.id) && !force {
        print("  Status: already downloaded, skipping")
        continue
      }

      print("  Downloading...")
      try await Acervo.ensureComponentReady(descriptor.id) { progress in
        let pct = Int(progress.overallProgress * 100)
        print("  \(progress.fileName): \(pct)%", terminator: "\r")
        fflush(stdout)
      }
      print("  Done.                              ")
    }

    print("\nAll components downloaded.")
  }
}
