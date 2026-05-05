import MLX
import Testing
import Tuberia

@testable import PixArtBackbone

// MARK: - Shared Fixture (R2.13)
//
// PixArtDiT.init constructs 28 DiT blocks with hiddenSize=1152 — ~600M parameters
// allocated per cold init. Each test in this suite previously rebuilt a fresh
// instance, paying that cost N times per CI run. The fixture below is initialized
// once (Swift guarantees thread-safe lazy init for static stored properties) and
// reused across every test that doesn't need to mutate the module.
//
// Tests that need an isolated instance (e.g. checking that init invariants run
// per-construction) can still call `try PixArtDiT(configuration:)` directly.
enum BackboneFixture {
  static let dit: PixArtDiT = {
    do {
      return try PixArtDiT(configuration: PixArtDiTConfiguration())
    } catch {
      fatalError("Failed to build shared PixArtDiT fixture: \(error)")
    }
  }()
}

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
@Suite("BackboneForward", .serialized)
struct BackboneForwardTests {

  // MARK: - Output Shape Tests

  @Test(
    "Forward pass output shape is [B, H, W, outputLatentChannels] and matches input spatial dims")
  func outputShapeContract() throws {
    // Use the shared fixture (R2.13) to amortize the 28-block init cost.
    let dit = BackboneFixture.dit

    // Minimal latent: [1, 4, 4, 4] — 4x4 spatial, 4 channels
    // gridH = 4/2 = 2, gridW = 4/2 = 2 → 4 tokens through the transformer
    let latents = MLXArray.zeros([1, 4, 4, 4])
    let conditioning = MLXArray.zeros([1, 120, 4096])
    let conditioningMask = MLXArray.ones([1, 120])
    let timestep = MLXArray([500 as Float])

    let input = BackboneInput(
      latents: latents,
      conditioning: conditioning,
      conditioningMask: conditioningMask,
      timestep: timestep
    )

    let output = try dit.forward(input)
    eval(output)

    // Full shape contract pinned in one place:
    // - 4-D
    // - Batch and spatial dims preserved
    // - Channel dim equals outputLatentChannels (variance channels discarded)
    // - outputLatentChannels protocol property matches actual output
    #expect(output.ndim == 4, "Output must be 4-dimensional [B, H, W, C]")
    #expect(output.dim(0) == latents.dim(0), "Batch must be preserved")
    #expect(output.dim(1) == latents.dim(1), "Height must be preserved")
    #expect(output.dim(2) == latents.dim(2), "Width must be preserved")
    #expect(output.dim(3) == 4, "Channel dimension must be 4 (variance channels discarded)")
    #expect(
      output.dim(3) == dit.outputLatentChannels,
      "Output channel count must match outputLatentChannels protocol property")
  }

  // MARK: - Attention Mask

  @Test("Forward pass with partial attention mask produces correct output shape")
  func forwardWithAttentionMask() throws {
    let dit = BackboneFixture.dit

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
    let dit = BackboneFixture.dit

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
