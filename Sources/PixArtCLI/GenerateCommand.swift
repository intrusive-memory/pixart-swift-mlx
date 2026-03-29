import ArgumentParser
import CoreGraphics
import Foundation
import ImageIO
import PixArtBackbone
import Tuberia

struct GenerateCommand: ParsableCommand {
  static let configuration = CommandConfiguration(
    commandName: "generate",
    abstract: "Generate an image from a text prompt"
  )

  @Option(name: .long, help: "Text prompt describing the image to generate")
  var prompt: String

  @Option(name: .long, help: "Image width in pixels")
  var width: Int = 1024

  @Option(name: .long, help: "Image height in pixels")
  var height: Int = 1024

  @Option(name: .long, help: "Output file path")
  var output: String = "output.png"

  @Option(name: .long, help: "Number of denoising steps")
  var steps: Int = PixArtRecipe.defaultSteps

  @Option(name: .long, help: "Classifier-free guidance scale")
  var guidance: Float = PixArtRecipe.defaultGuidanceScale

  @Option(name: .long, help: "Random seed for reproducible generation")
  var seed: UInt32?

  func run() throws {
    _ = PixArtComponents.registered

    let request = DiffusionGenerationRequest(
      prompt: prompt,
      width: width,
      height: height,
      steps: steps,
      guidanceScale: guidance,
      seed: seed
    )

    print("Generating: \"\(prompt)\"")
    print("Size: \(width)x\(height), steps: \(steps), guidance: \(guidance)")
    if let seed { print("Seed: \(seed)") }

    let result = try runAsync {
      let pipeline = try DiffusionPipeline(recipe: PixArtRecipe())
      try await pipeline.loadModels { fraction, component in
        print("Loading \(component): \(Int(fraction * 100))%")
      }
      return try await pipeline.generate(request: request) { progress in
        switch progress {
        case .encoding(let fraction):
          print("Encoding: \(Int(fraction * 100))%")
        case .generating(let step, let total, _):
          print("Step \(step)/\(total)")
        case .decoding:
          print("Decoding latents...")
        case .rendering:
          print("Rendering image...")
        case .complete(let duration):
          print(String(format: "Done in %.1fs", duration))
        default:
          break
        }
      }
    }

    guard case .image(let cgImage) = result.output else {
      throw CLIError.unexpectedOutput("Expected image output")
    }

    try savePNG(cgImage, to: output)
    print("Saved: \(output) (seed: \(result.seed))")
  }

  private func savePNG(_ image: CGImage, to path: String) throws {
    let url = URL(filePath: path)
    guard
      let destination = CGImageDestinationCreateWithURL(
        url as CFURL, "public.png" as CFString, 1, nil
      )
    else {
      throw CLIError.saveFailed("Could not create image destination at \(path)")
    }
    CGImageDestinationAddImage(destination, image, nil)
    guard CGImageDestinationFinalize(destination) else {
      throw CLIError.saveFailed("Could not write PNG to \(path)")
    }
  }
}
