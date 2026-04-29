import Testing

@testable import PixArtBackbone

// MARK: - Configuration Tests

@Suite("PixArtDiTConfiguration")
struct ConfigurationTests {

  @Test("Default values match PixArt-Sigma XL paper (arXiv:2403.04692)")
  func defaults() {
    let config = PixArtDiTConfiguration()
    #expect(config.hiddenSize == 1152)
    #expect(config.numHeads == 16)
    #expect(config.headDim == 72)
    #expect(config.depth == 28)
    #expect(config.patchSize == 2)
    #expect(config.inChannels == 4)
    #expect(config.outChannels == 8)
    #expect(config.mlpRatio == 4.0)
    #expect(config.captionChannels == 4096)
    #expect(config.maxTextLength == 120)
    #expect(config.peInterpolation == 2.0)
    // baseSize is the **latent** base resolution: 1024px training res / 8x VAE downscale.
    // The forward pass divides by patchSize (2) to reach the diffusers
    // PixArtTransformer2DModel.base_size grid of 64. Previously this was 512 (the pixel
    // base size), which multiplied position coordinates by 4× and corrupted attention.
    #expect(config.baseSize == 128)
  }

  @Test("headDim equals hiddenSize / numHeads")
  func headDimConsistency() {
    let config = PixArtDiTConfiguration()
    #expect(config.headDim == config.hiddenSize / config.numHeads)
  }

  @Test("Custom configuration preserves values")
  func customConfig() {
    let config = PixArtDiTConfiguration(hiddenSize: 768, numHeads: 12, headDim: 64, depth: 12)
    #expect(config.hiddenSize == 768)
    #expect(config.numHeads == 12)
    #expect(config.headDim == 64)
    #expect(config.depth == 12)
  }
}

// MARK: - Backbone Initialization Tests

@Suite("PixArtDiT")
struct DiTTests {

  @Test("Initializes with default configuration")
  func initDefault() throws {
    let dit = try PixArtDiT(configuration: PixArtDiTConfiguration())
    #expect(dit.expectedConditioningDim == 4096)
    #expect(dit.outputLatentChannels == 4)
    #expect(dit.expectedMaxSequenceLength == 120)
    #expect(dit.isLoaded == false)
  }

  @Test("Estimated memory matches component descriptor (~300 MB)")
  func estimatedMemory() throws {
    let dit = try PixArtDiT(configuration: PixArtDiTConfiguration())
    #expect(dit.estimatedMemoryBytes == 314_572_800)
  }
}
