import Tuberia
import TuberiaCatalog

// MARK: - PixArt-Sigma FP16 Pipeline Recipe

/// Mixed-precision pipeline recipe for diagnosing int4 quantization artifacts.
///
/// Identical to ``PixArtRecipe`` except the backbone uses the fp16 DiT weights
/// (`pixart-sigma-xl-dit-fp16`) instead of the int4 weights (`pixart-sigma-xl-dit-int4`).
///
/// The fp16 weights are produced by ``scripts/dequantize_dit_to_fp16.py``, which
/// dequantizes the int4 safetensors back to float16. This gives the ORIGINAL fp16
/// values (before any quantization was applied), not just dequantized-from-int4 values.
///
/// **Purpose**: If the blue/cyan mosaic artifact disappears with fp16 weights but
/// persists with int4 weights, that confirms int4 quantization errors accumulating
/// across 28 DiT blocks are the root cause of the color bias.
///
/// **Usage**:
/// 1. Run `python3 scripts/dequantize_dit_to_fp16.py` to produce the fp16 safetensors.
/// 2. Run `make test-fixtures-fp16` to generate a fixture using this recipe.
/// 3. Compare `pixart-seed42-fp16.png` against `pixart-seed42.png` (int4 baseline).
///
/// Memory profile: ~2.5 GB (fp16 DiT ~1.2 GB + int4 T5 ~1.2 GB + fp16 VAE ~160 MB).
/// Requires a machine with ≥16 GB unified memory.
public struct PixArtFP16Recipe: PipelineRecipe, Sendable {

  // MARK: - Associated Types

  public typealias Encoder = T5XXLEncoder
  public typealias Sched = DPMSolverScheduler
  public typealias Back = PixArtDiT
  public typealias Dec = SDXLVAEDecoder
  public typealias Rend = ImageRenderer

  // MARK: - Initialization

  public init() {}

  // MARK: - Default Generation Parameters

  /// Default number of denoising steps (same as int4 recipe).
  public static let defaultSteps: Int = 20

  /// Default classifier-free guidance scale (same as int4 recipe).
  public static let defaultGuidanceScale: Float = 4.5

  // MARK: - Encoder Configuration

  /// T5-XXL encoder configuration (same as int4 recipe — encoder stays int4).
  public var encoderConfig: T5XXLEncoderConfiguration {
    T5XXLEncoderConfiguration(
      componentId: "t5-xxl-encoder-int4",
      maxSequenceLength: 120,
      embeddingDim: 4096
    )
  }

  // MARK: - Scheduler Configuration

  /// DPM-Solver++ scheduler configuration (identical to int4 recipe).
  public var schedulerConfig: DPMSolverSchedulerConfiguration {
    DPMSolverSchedulerConfiguration(
      betaSchedule: .scaledLinear(betaStart: 0.0001, betaEnd: 0.02),
      predictionType: .epsilon,
      solverOrder: 2,
      trainTimesteps: 1000
    )
  }

  // MARK: - Backbone Configuration

  /// PixArt-Sigma XL DiT backbone configuration with all default values.
  public var backboneConfig: PixArtDiTConfiguration {
    PixArtDiTConfiguration()
  }

  // MARK: - Decoder Configuration

  /// SDXL VAE decoder configuration (same as int4 recipe).
  public var decoderConfig: SDXLVAEDecoderConfiguration {
    SDXLVAEDecoderConfiguration(
      componentId: "sdxl-vae-decoder-fp16",
      latentChannels: 4,
      scalingFactor: 0.13025
    )
  }

  // MARK: - Renderer Configuration

  /// ImageRenderer is stateless and requires no configuration.
  public var rendererConfig: Void { () }

  // MARK: - PipelineRecipe Properties

  /// PixArt-Sigma is text-to-image only.
  public var supportsImageToImage: Bool { false }

  /// Use empty string encoding for classifier-free guidance unconditional embedding.
  public var unconditionalEmbeddingStrategy: UnconditionalEmbeddingStrategy { .emptyPrompt }

  /// All Acervo component IDs required by this recipe.
  ///
  /// Uses `pixart-sigma-xl-dit-fp16` instead of `pixart-sigma-xl-dit-int4`.
  /// T5-XXL and SDXL VAE remain the same int4/fp16 components as the base recipe.
  public var allComponentIds: [String] {
    [
      "t5-xxl-encoder-int4",
      "pixart-sigma-xl-dit-fp16",
      "sdxl-vae-decoder-fp16",
    ]
  }

  // MARK: - Quantization

  /// All components use weights as stored: fp16 for the DiT (no quantization),
  /// int4 for T5, fp16 for VAE.
  public func quantizationFor(_ role: PipelineRole) -> QuantizationConfig {
    .asStored
  }

  // MARK: - Validation

  /// Validates shape contract consistency (same checks as PixArtRecipe).
  public func validate() throws {
    let config = backboneConfig
    guard encoderConfig.embeddingDim == config.captionChannels else {
      throw PixArtRecipeError.shapeMismatch(
        "encoderConfig.embeddingDim (\(encoderConfig.embeddingDim)) "
          + "!= backboneConfig.captionChannels (\(config.captionChannels))"
      )
    }
    guard encoderConfig.maxSequenceLength == config.maxTextLength else {
      throw PixArtRecipeError.shapeMismatch(
        "encoderConfig.maxSequenceLength (\(encoderConfig.maxSequenceLength)) "
          + "!= backboneConfig.maxTextLength (\(config.maxTextLength))"
      )
    }
    guard decoderConfig.latentChannels == 4 else {
      throw PixArtRecipeError.shapeMismatch(
        "decoderConfig.latentChannels (\(decoderConfig.latentChannels)) "
          + "!= backbone outputLatentChannels (4)"
      )
    }
  }
}
