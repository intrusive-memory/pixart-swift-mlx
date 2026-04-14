import Tuberia
import TuberiaCatalog

// MARK: - PixArt-Sigma Pipeline Recipe

/// Pipeline recipe connecting PixArt-Sigma components:
///
/// ```
/// T5XXLEncoder → PixArtDiT → SDXLVAEDecoder → ImageRenderer
///                    ↑
///             DPMSolverScheduler
/// ```
///
/// Default generation parameters:
/// - Steps: 20
/// - Guidance scale: 4.5
/// - Beta schedule: linear (betaStart=0.0001, betaEnd=0.02) — NOT shifted cosine
/// - Prediction type: epsilon (model predicts noise)
///
/// Memory profiles (int4 quantized):
/// - All components loaded simultaneously: ~2 GB (Mac with 8+ GB unified memory)
/// - Two-phase T5 phase: ~1.4 GB (iPad 8 GB viable)
/// - Two-phase DiT+VAE phase: ~500 MB (iPad 8 GB viable)
public struct PixArtRecipe: PipelineRecipe, Sendable {

  // MARK: - Associated Types

  public typealias Encoder = T5XXLEncoder
  public typealias Sched = DPMSolverScheduler
  public typealias Back = PixArtDiT
  public typealias Dec = SDXLVAEDecoder
  public typealias Rend = ImageRenderer

  // MARK: - Initialization

  public init() {}

  // MARK: - Default Generation Parameters

  /// Default number of denoising steps.
  public static let defaultSteps: Int = 20

  /// Default classifier-free guidance scale.
  public static let defaultGuidanceScale: Float = 4.5

  // MARK: - Encoder Configuration

  /// T5-XXL encoder configuration.
  ///
  /// - componentId: "t5-xxl-encoder-int4" — Acervo ID for weights + tokenizer
  /// - maxSequenceLength: 120 — matches PixArt-Sigma training setup (not T5's max 512)
  /// - embeddingDim: 4096 — T5-XXL hidden dimension
  public var encoderConfig: T5XXLEncoderConfiguration {
    T5XXLEncoderConfiguration(
      componentId: "t5-xxl-encoder-int4",
      maxSequenceLength: 120,
      embeddingDim: 4096
    )
  }

  // MARK: - Scheduler Configuration

  /// DPM-Solver++ scheduler configuration.
  ///
  /// PixArt-Sigma was trained with the HuggingFace `"scaled_linear"` beta schedule:
  ///   betas = linspace(sqrt(0.0001), sqrt(0.02), 1000)²
  /// Using plain `linear` produces wrong alphas_cumprod values which cause the
  /// denoising trajectory to diverge (latents drift toward large positive values).
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

  /// SDXL VAE decoder configuration.
  ///
  /// - componentId: "sdxl-vae-decoder-fp16" — float16 (Conv2d layers do not benefit from int4)
  /// - latentChannels: 4 — matches PixArtDiT.outputLatentChannels
  /// - scalingFactor: 0.13025 — standard SDXL VAE scaling factor
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

  /// PixArt-Sigma is text-to-image only; image-to-image is not supported.
  public var supportsImageToImage: Bool { false }

  /// Use empty string encoding for classifier-free guidance unconditional embedding.
  /// This means CFG encodes "" through T5XXLEncoder for the unconditional branch.
  public var unconditionalEmbeddingStrategy: UnconditionalEmbeddingStrategy { .emptyPrompt }

  /// All Acervo component IDs required by this recipe.
  ///
  /// Order: encoder, backbone, decoder (scheduler and renderer have no weights).
  /// T5-XXL and SDXL VAE are catalog components; PixArt DiT is owned by this package.
  public var allComponentIds: [String] {
    [
      "t5-xxl-encoder-int4",
      "pixart-sigma-xl-dit-int4",
      "sdxl-vae-decoder-fp16",
    ]
  }

  // MARK: - Quantization

  /// All components use weights as stored (int4 for DiT and T5, fp16 for VAE).
  public func quantizationFor(_ role: PipelineRole) -> QuantizationConfig {
    .asStored
  }

  // MARK: - Validation

  /// Validates shape contract consistency.
  ///
  /// The pipeline assembly validates:
  /// - encoderConfig.embeddingDim == backboneConfig.captionChannels (4096)
  /// - encoderConfig.maxSequenceLength == backboneConfig.maxTextLength (120)
  /// - decoderConfig.latentChannels == backbone outputLatentChannels (4)
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

// MARK: - Validation Error

/// Errors thrown by PixArtRecipe.validate().
public enum PixArtRecipeError: Error, Sendable {
  case shapeMismatch(String)
}
