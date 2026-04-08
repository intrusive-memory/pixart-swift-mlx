import MLX
import Testing
import Tuberia

@testable import PixArtBackbone

// MARK: - Backbone Forward Pass Tests

/// Tests the full PixArtDiT forward pass with synthetic (zero) inputs.
///
/// Uses the default PixArtDiTConfiguration with a minimal latent size to keep
/// tests fast without real weights:
/// - hiddenSize: 1152 (required — micro-condition embedders output exactly 1152)
/// - depth: 28 (required — PixArtDiT.init asserts blocks.count == 28)
/// - patchSize: 2 (default)
/// - inChannels: 4 (default)
/// - outChannels: 8 (default, forward discards last 4 → output has 4)
///
/// Input latent: [1, 4, 4, 4] — minimal 4×4 VAE latent (4 channels).
/// After patch embed (patchSize=2): gridH=2, gridW=2 → 4 tokens.
/// Only 4 tokens flow through 28 DiT blocks, keeping the test fast.
/// FinalLayer unpatchifies back to [1, 4, 4, 8] then forward() slices first 4 → [1, 4, 4, 4].
///
/// Note: The latent spatial dimensions must be divisible by patchSize (2).
/// Minimum valid latent is [1, 2, 2, 4] → output [1, 2, 2, 4].
@Suite("BackboneForward")
struct BackboneForwardTests {

  // MARK: - Output Shape Tests

  @Test("Forward pass output shape is [B, H, W, 4] for batch size 1")
  func outputShapeBatch1() throws {
    let config = PixArtDiTConfiguration()
    let dit = try PixArtDiT(configuration: config)

    // Minimal latent: [1, 4, 4, 4] — 4x4 spatial, 4 channels
    // gridH = 4/2 = 2, gridW = 4/2 = 2 → 4 tokens through the transformer
    let latents = MLXArray.zeros([1, 4, 4, 4])
    // Conditioning: [1, 120, 4096] — 120 text tokens (maxTextLength), captionChannels=4096
    let conditioning = MLXArray.zeros([1, 120, 4096])
    // Conditioning mask: [1, 120] — all tokens valid
    let conditioningMask = MLXArray.ones([1, 120])
    // Timestep: [1] — scalar timestep per batch element
    let timestep = MLXArray([500 as Float])

    let input = BackboneInput(
      latents: latents,
      conditioning: conditioning,
      conditioningMask: conditioningMask,
      timestep: timestep
    )

    let output = try dit.forward(input)
    eval(output)

    #expect(output.ndim == 4, "Output must be 4-dimensional [B, H, W, C]")
    #expect(output.dim(0) == 1, "Batch dimension must be 1")
    #expect(output.dim(1) == 4, "Spatial height must match latent height (4)")
    #expect(output.dim(2) == 4, "Spatial width must match latent width (4)")
    #expect(output.dim(3) == 4, "Channel dimension must be 4 (variance discarded)")
  }

  @Test("Forward pass output shape matches [B, H, W, 4] contract")
  func outputShapeContractVerified() throws {
    let config = PixArtDiTConfiguration()
    let dit = try PixArtDiT(configuration: config)

    // Use the smallest valid latent: 2 * patchSize = 4 in each spatial dim
    let spatialH = 4
    let spatialW = 4
    let B = 1

    let latents = MLXArray.zeros([B, spatialH, spatialW, config.inChannels])
    let conditioning = MLXArray.zeros([B, config.maxTextLength, config.captionChannels])
    let conditioningMask = MLXArray.ones([B, config.maxTextLength])
    let timestep = MLXArray([250 as Float])

    let input = BackboneInput(
      latents: latents,
      conditioning: conditioning,
      conditioningMask: conditioningMask,
      timestep: timestep
    )

    let output = try dit.forward(input)
    eval(output)

    // Shape contract: [B, spatialH, spatialW, outputLatentChannels]
    #expect(output.shape == [B, spatialH, spatialW, dit.outputLatentChannels])
  }

  @Test("Forward pass output has 4 channels — variance channels are discarded")
  func outputHasFourChannels() throws {
    let dit = try PixArtDiT(configuration: PixArtDiTConfiguration())

    let input = BackboneInput(
      latents: MLXArray.zeros([1, 4, 4, 4]),
      conditioning: MLXArray.zeros([1, 120, 4096]),
      conditioningMask: MLXArray.ones([1, 120]),
      timestep: MLXArray([999 as Float])
    )

    let output = try dit.forward(input)
    eval(output)

    // outChannels=8 internally, but forward() slices [0..<4] before returning
    #expect(output.dim(3) == 4)
    #expect(dit.outputLatentChannels == 4)
  }

  @Test("Forward pass output ndim is exactly 4")
  func outputNdimIs4() throws {
    let dit = try PixArtDiT(configuration: PixArtDiTConfiguration())

    let input = BackboneInput(
      latents: MLXArray.zeros([1, 4, 4, 4]),
      conditioning: MLXArray.zeros([1, 120, 4096]),
      conditioningMask: MLXArray.ones([1, 120]),
      timestep: MLXArray([100 as Float])
    )

    let output = try dit.forward(input)
    eval(output)

    #expect(output.ndim == 4)
  }

  // MARK: - Spatial Dimension Preservation

  @Test("Output spatial dimensions match input latent spatial dimensions")
  func spatialDimensionsPreserved() throws {
    let dit = try PixArtDiT(configuration: PixArtDiTConfiguration())

    // Use 4x4 latent — gridH = 4/2 = 2, gridW = 4/2 = 2
    // FinalLayer unpatchifies back to 4x4 with outChannels=8, then [0..<4] gives 4ch
    let latents = MLXArray.zeros([1, 4, 4, 4])
    let input = BackboneInput(
      latents: latents,
      conditioning: MLXArray.zeros([1, 120, 4096]),
      conditioningMask: MLXArray.ones([1, 120]),
      timestep: MLXArray([500 as Float])
    )

    let output = try dit.forward(input)
    eval(output)

    // Spatial dimensions must be preserved through patch → unpatching cycle
    #expect(output.dim(1) == latents.dim(1), "Height must be preserved")
    #expect(output.dim(2) == latents.dim(2), "Width must be preserved")
  }

  // MARK: - Attention Mask

  @Test("Forward pass with partial attention mask produces correct output shape")
  func forwardWithAttentionMask() throws {
    let dit = try PixArtDiT(configuration: PixArtDiTConfiguration())

    // Mask: first 60 tokens valid, last 60 are padding
    let maskData = [Int32](repeating: 1, count: 60) + [Int32](repeating: 0, count: 60)
    let mask = MLXArray(maskData).reshaped(1, 120)

    let input = BackboneInput(
      latents: MLXArray.zeros([1, 4, 4, 4]),
      conditioning: MLXArray.zeros([1, 120, 4096]),
      conditioningMask: mask,
      timestep: MLXArray([750 as Float])
    )

    let output = try dit.forward(input)
    eval(output)

    #expect(output.shape == [1, 4, 4, 4])
  }

  // MARK: - Protocol Conformance

  @Test("PixArtDiT outputLatentChannels matches forward output channel count")
  func outputLatentChannelsMatchesForward() throws {
    let dit = try PixArtDiT(configuration: PixArtDiTConfiguration())

    let input = BackboneInput(
      latents: MLXArray.zeros([1, 4, 4, 4]),
      conditioning: MLXArray.zeros([1, 120, 4096]),
      conditioningMask: MLXArray.ones([1, 120]),
      timestep: MLXArray([500 as Float])
    )

    let output = try dit.forward(input)
    eval(output)

    // The protocol property and the actual output channel count must agree
    #expect(output.dim(3) == dit.outputLatentChannels)
  }

  // MARK: - Full 1024×1024 Latent Shape

  /// Verifies the forward pass with the canonical 1024×1024 VAE latent shape.
  ///
  /// A 1024×1024 image is encoded by the SDXL VAE at 8× downsampling, producing
  /// a [1, 128, 128, 4] latent tensor. The DiT backbone must accept this shape
  /// and return an output of the same spatial dimensions with 4 channels.
  ///
  /// Grid: 128/2 = 64 × 64 = 4096 tokens through 28 DiT blocks.
  /// This test is deliberately marked `.timeLimit(.minutes(5))` since it processes
  /// 4096 tokens with zero weights — fast on Apple Silicon but non-trivial.
  @Test(
    "Forward pass with [1, 128, 128, 4] latent produces [1, 128, 128, 4] output",
    .timeLimit(.minutes(5))
  )
  func forwardPassWith1024LatentShape() throws {
    let config = PixArtDiTConfiguration()
    let dit = try PixArtDiT(configuration: config)

    // Canonical 1024×1024 VAE latent (SDXL 8× downsampling: 1024/8 = 128)
    let latents = MLXArray.zeros([1, 128, 128, 4])
    // Full T5-XXL conditioning: 120 tokens × 4096 embedding dim
    let conditioning = MLXArray.zeros([1, 120, 4096])
    let conditioningMask = MLXArray.ones([1, 120])
    // Single timestep per batch element
    let timestep = MLXArray([500 as Float])

    let input = BackboneInput(
      latents: latents,
      conditioning: conditioning,
      conditioningMask: conditioningMask,
      timestep: timestep
    )

    let output = try dit.forward(input)
    eval(output)

    // Output must have the same spatial dimensions as the input latent
    #expect(output.shape == [1, 128, 128, 4])
  }
}
