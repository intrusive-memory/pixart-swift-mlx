import ArgumentParser
import Foundation
import PixArtBackbone
import SwiftAcervo
import TuberiaCatalog

struct InfoCommand: ParsableCommand {
  static let configuration = CommandConfiguration(
    commandName: "info",
    abstract: "Show PixArt-Sigma model details and download status"
  )

  func run() throws {
    _ = PixArtComponents.registered

    let recipe = PixArtRecipe()
    let registry = CatalogRegistration.shared

    print("PixArt-Sigma Pipeline")
    print("=====================")
    print("  Architecture : DiT-XL (Diffusion Transformer)")
    print("  Default steps: \(PixArtRecipe.defaultSteps)")
    print("  Default CFG  : \(PixArtRecipe.defaultGuidanceScale)")
    print("  Max text len : 120 tokens (T5-XXL)")
    print("  Latent dim   : 4 channels, 8x downscale")
    print()

    print("Components")
    print("----------")
    let componentIds = recipe.allComponentIds

    var totalEstimatedBytes = 0
    for componentId in componentIds {
      guard let descriptor = registry.descriptor(for: componentId) else {
        print("  \(componentId): (descriptor not found)")
        continue
      }
      let sizeGB = Double(descriptor.estimatedSizeBytes) / 1_073_741_824
      let available = Acervo.isModelAvailable(descriptor.huggingFaceRepo)
      let status = available ? "downloaded" : "not downloaded"
      print(String(format: "  %-35s  ~%.1f GB  %@", componentId, sizeGB, status))
      totalEstimatedBytes += descriptor.estimatedSizeBytes
    }

    let totalGB = Double(totalEstimatedBytes) / 1_073_741_824
    print(String(format: "\n  Total (all components): ~%.1f GB", totalGB))
    print()

    print("Storage")
    print("-------")
    let modelsDir = Acervo.sharedModelsDirectory
    print("  Models directory: \(modelsDir.path)")
  }
}
