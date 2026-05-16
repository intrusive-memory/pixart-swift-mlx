import Foundation
@preconcurrency import MLX
import Testing
import Tuberia
import TuberiaCatalog

@testable import PixArtBackbone

/// Regression suite for the PixArt-Sigma colour-cast bug investigated 2026-05-16.
///
/// **Symptom**: on the mid-century-modern prompt at seed 42, generated images
/// were 96.4% red+orange with 23.5% crushed blacks, vs. Flux2 Klein-4B's
/// balanced palette for the same prompt.
///
/// **Root cause**: ``SDXLVAEDecoder`` (in SwiftTuberia) was returning the raw
/// SDXL VAE output in roughly `[-1, 1]` (Diffusers `AutoencoderKL` convention),
/// while the downstream ``ImageRenderer`` expects float pixels in `[0, 1]`
/// and applies a `clip(x, 0, 1) * 255`. Without the canonical
/// `(x * 0.5 + 0.5).clamp(0, 1)` remap, negative pixel values were clipped to
/// zero, asymmetrically crushing the dark end of the distribution and biasing
/// the rendered histogram warm.
///
/// **Fix**: ``SDXLVAEDecoder.decode`` now applies the remap internally before
/// returning ``DecodedOutput``. These tests pin the contract from PixArt's
/// side: the decoder configured per ``PixArtRecipe`` / ``PixArtFP16Recipe``
/// must produce pixel data inside `[0, 1]`. If the upstream wrapper or the
/// recipe ever regresses (e.g. somebody re-exposes raw VAE output, switches
/// the scaling factor, or routes through a renderer with a different contract),
/// this suite fails immediately.
@Suite("PixArt VAE pixel range — [0, 1] contract")
struct PixelRangeRegressionTests {

  /// The recipe still selects the SDXL-fp16-fix scaling factor. Switching to
  /// 0.18215 (vanilla SDXL) or another value re-opens H1 from the colour-cast
  /// investigation TODO.
  @Test("PixArtRecipe.decoderConfig pins scalingFactor 0.13025")
  func int4RecipeScalingFactor() {
    let recipe = PixArtRecipe()
    #expect(abs(recipe.decoderConfig.scalingFactor - 0.13025) < 1e-6)
  }

  @Test("PixArtFP16Recipe.decoderConfig pins scalingFactor 0.13025")
  func fp16RecipeScalingFactor() {
    let recipe = PixArtFP16Recipe()
    #expect(abs(recipe.decoderConfig.scalingFactor - 0.13025) < 1e-6)
  }

  /// End-to-end through the SDXLVAEDecoder wrapper that PixArt uses. With the
  /// placeholder forward pass (weights nil) the test stays a fast unit test
  /// — no GPU, no CDN weights — while still exercising the post-decode
  /// `(x * 0.5 + 0.5).clamp(0, 1)` step that gates the renderer contract.
  @Test("SDXLVAEDecoder configured per PixArtRecipe emits pixels in [0, 1]")
  func int4RecipeDecoderEmitsZeroOne() throws {
    let recipe = PixArtRecipe()
    let decoder = try SDXLVAEDecoder(configuration: recipe.decoderConfig)
    let latents = MLXArray.zeros([1, 4, 4, 4]).asType(.float32)
    let output = try decoder.decode(latents)
    let values = output.data.asArray(Float.self)
    let minValue = values.min() ?? .nan
    let maxValue = values.max() ?? .nan
    #expect(
      minValue >= 0.0 && maxValue <= 1.0,
      "Decoder output must be in [0, 1] for the renderer contract — got [\(minValue), \(maxValue)]"
    )
  }

  @Test("SDXLVAEDecoder configured per PixArtFP16Recipe emits pixels in [0, 1]")
  func fp16RecipeDecoderEmitsZeroOne() throws {
    let recipe = PixArtFP16Recipe()
    let decoder = try SDXLVAEDecoder(configuration: recipe.decoderConfig)
    let latents = MLXArray.zeros([1, 4, 4, 4]).asType(.float32)
    let output = try decoder.decode(latents)
    let values = output.data.asArray(Float.self)
    let minValue = values.min() ?? .nan
    let maxValue = values.max() ?? .nan
    #expect(
      minValue >= 0.0 && maxValue <= 1.0,
      "Decoder output must be in [0, 1] for the renderer contract — got [\(minValue), \(maxValue)]"
    )
  }

  /// The unloaded-placeholder contract: an empty latent should round-trip to
  /// mid-gray (0.5) at every pixel. If this drifts off 0.5, either the
  /// placeholder semantics changed or the post-decode remap math regressed
  /// — both are bugs.
  @Test("Unloaded decoder produces mid-gray (0.5) pixels after the [0,1] remap")
  func unloadedPlaceholderIsMidGray() throws {
    let recipe = PixArtRecipe()
    let decoder = try SDXLVAEDecoder(configuration: recipe.decoderConfig)
    let latents = MLXArray.zeros([1, 2, 2, 4]).asType(.float32)
    let output = try decoder.decode(latents)
    let values = output.data.asArray(Float.self)
    let allMidGray = values.allSatisfy { abs($0 - 0.5) < 1e-6 }
    #expect(allMidGray, "Unloaded placeholder must produce 0.5 after the [-1,1] → [0,1] remap")
  }
}
