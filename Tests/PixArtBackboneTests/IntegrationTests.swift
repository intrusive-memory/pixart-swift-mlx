#if INTEGRATION_TESTS

  // MARK: - PixArt-Sigma Integration Tests
  //
  // These tests require:
  //   - Real PixArt-Sigma XL model weights (pixart-sigma-xl-dit-int4)
  //   - T5-XXL encoder weights (t5-xxl-encoder-int4)
  //   - SDXL VAE decoder weights (sdxl-vae-decoder-fp16)
  //   - GPU compute (Apple Silicon M1 or later)
  //   - ~2 GB unified memory available
  //
  // They are NOT run in CI. To run locally:
  //
  //   xcodebuild test \
  //     -scheme pixart-swift-mlx-Package \
  //     -destination 'platform=macOS,arch=arm64' \
  //     SWIFT_ACTIVE_COMPILATION_CONDITIONS=INTEGRATION_TESTS
  //
  // Or via make:
  //   make test-integration
  //
  // These tests are gated on #if INTEGRATION_TESTS to ensure they are never
  // accidentally executed in a CI environment where weights are not available.

  import CoreGraphics
  import MLX
  import Testing
  import Tuberia
  import TuberiaCatalog

  @testable import PixArtBackbone

  // MARK: - Pipeline Assembly Tests

  @Suite("Integration: Pipeline Assembly")
  struct PipelineAssemblyTests {

    /// Verifies that PixArtRecipe assembles a DiffusionPipeline without errors.
    ///
    /// This test requires all component weights to be registered with Acervo.
    /// It does NOT generate an image — it only tests the assembly phase.
    @Test("PixArtRecipe assembles into DiffusionPipeline")
    func recipAssemblesIntoPipeline() async throws {
      let recipe = PixArtRecipe()

      // Validate the recipe (shape contract checks)
      try recipe.validate()

      // Assemble the pipeline — this instantiates all components but does not
      // load weights. Weight loading is deferred until first inference.
      let pipeline = try await DiffusionPipeline(recipe: recipe)

      // Verify the pipeline reports components from our recipe
      #expect(pipeline.backboneOutputLatentChannels == 4)
    }

    /// Verifies that PixArtRecipe.validate() passes for all default configurations.
    @Test("PixArtRecipe.validate() passes with default configurations")
    func recipeValidationPasses() throws {
      let recipe = PixArtRecipe()
      // Should not throw — all shape contracts are satisfied by default values
      try recipe.validate()
    }
  }

  // MARK: - Seed Reproducibility Tests

  @Suite("Integration: Seed Reproducibility")
  struct SeedReproducibilityTests {

    /// Validates that two runs with the same seed produce identical outputs.
    ///
    /// Criterion: PSNR > 40 dB between the two output images.
    /// This is a determinism check — not cross-platform reproducibility (which
    /// targets PSNR > 30 dB due to MLX non-determinism across devices).
    @Test("Same seed produces reproducible output (PSNR > 40 dB)")
    func seedReproducibility() async throws {
      let recipe = PixArtRecipe()
      let pipeline = try await DiffusionPipeline(recipe: recipe)

      let request = DiffusionGenerationRequest(
        prompt: "A serene mountain landscape at dawn",
        negativePrompt: nil,
        width: 1024,
        height: 1024,
        steps: 5,  // Use minimal steps for speed in integration tests
        guidanceScale: 4.5,
        seed: 42
      )

      // First run
      let result1 = try await pipeline.generate(request: request)
      guard let image1 = result1.image else {
        #expect(Bool(false), "First generation returned nil image")
        return
      }

      // Second run — same request, same seed
      let result2 = try await pipeline.generate(request: request)
      guard let image2 = result2.image else {
        #expect(Bool(false), "Second generation returned nil image")
        return
      }

      // Compute PSNR between the two outputs
      let psnr = computePSNR(image1: image1, image2: image2)
      #expect(psnr > 40.0, "PSNR \(psnr) dB must exceed 40 dB for same-seed reproducibility")
    }

    /// Validates that different seeds produce different outputs.
    @Test("Different seeds produce distinct outputs (PSNR < 40 dB)")
    func differentSeedsProduceDifferentOutputs() async throws {
      let recipe = PixArtRecipe()
      let pipeline = try await DiffusionPipeline(recipe: recipe)

      let baseRequest = DiffusionGenerationRequest(
        prompt: "A serene mountain landscape at dawn",
        negativePrompt: nil,
        width: 1024,
        height: 1024,
        steps: 5,
        guidanceScale: 4.5,
        seed: 42
      )

      let differentSeedRequest = DiffusionGenerationRequest(
        prompt: "A serene mountain landscape at dawn",
        negativePrompt: nil,
        width: 1024,
        height: 1024,
        steps: 5,
        guidanceScale: 4.5,
        seed: 123
      )

      let result1 = try await pipeline.generate(request: baseRequest)
      let result2 = try await pipeline.generate(request: differentSeedRequest)

      guard let image1 = result1.image, let image2 = result2.image else {
        #expect(Bool(false), "One or both generations returned nil image")
        return
      }

      let psnr = computePSNR(image1: image1, image2: image2)
      // Different seeds should produce meaningfully different images
      #expect(psnr < 40.0, "PSNR \(psnr) dB should be < 40 dB for different-seed outputs")
    }
  }

  // MARK: - Two-Phase Loading Tests

  @Suite("Integration: Two-Phase Loading")
  struct TwoPhaseLoadingTests {

    /// Validates the two-phase loading strategy for memory-constrained devices.
    ///
    /// Phase 1: T5-XXL encoder loaded, DiT + VAE unloaded
    /// Phase 2: T5-XXL unloaded, DiT + VAE loaded
    ///
    /// This test verifies that each phase completes without memory errors
    /// and that the pipeline produces valid output when phases are used correctly.
    @Test("Two-phase loading completes without memory errors")
    func twoPhaseLoadingCompletesSuccessfully() async throws {
      let recipe = PixArtRecipe()
      let pipeline = try await DiffusionPipeline(recipe: recipe)

      // Simulate memory-constrained two-phase generation
      let request = DiffusionGenerationRequest(
        prompt: "A simple geometric pattern",
        negativePrompt: nil,
        width: 512,
        height: 512,
        steps: 5,
        guidanceScale: 4.5,
        seed: 1
      )

      // The DiffusionPipeline two-phase API handles encoder/decoder phasing
      // Phase 1: encode prompt with T5
      let encoderOutput = try await pipeline.encodePrompt(request: request)
      #expect(encoderOutput.embeddings.shape[0] == 1, "Batch dimension must be 1")
      #expect(encoderOutput.embeddings.shape[2] == 4096, "T5-XXL embedding dim must be 4096")

      // Phase 2: denoise latents + decode
      let result = try await pipeline.denoise(request: request, encoderOutput: encoderOutput)
      #expect(result.image != nil, "Two-phase generation must produce a valid image")
    }

    /// Verifies that encoder weights can be unloaded between phases.
    @Test("Encoder unloads cleanly before DiT phase")
    func encoderUnloadsCleanly() async throws {
      let recipe = PixArtRecipe()
      let pipeline = try await DiffusionPipeline(recipe: recipe)

      // Phase 1: Load and run encoder
      let request = DiffusionGenerationRequest(
        prompt: "Test prompt",
        negativePrompt: nil,
        width: 512,
        height: 512,
        steps: 3,
        guidanceScale: 4.5,
        seed: 7
      )

      let encoderOutput = try await pipeline.encodePrompt(request: request)

      // Explicitly unload encoder to free memory before DiT phase
      await pipeline.unloadEncoder()

      // Phase 2: DiT + VAE should still work after encoder unload
      let result = try await pipeline.denoise(request: request, encoderOutput: encoderOutput)
      #expect(result.image != nil, "Pipeline must produce output after encoder unload")
    }
  }

  // MARK: - PSNR Helper

  /// Computes Peak Signal-to-Noise Ratio between two CGImages.
  ///
  /// - Parameters:
  ///   - image1: First image (reference).
  ///   - image2: Second image (to compare).
  /// - Returns: PSNR in decibels. Returns `Double.infinity` if images are identical.
  ///            Returns -1 if images cannot be compared (size mismatch or decode failure).
  private func computePSNR(image1: CGImage, image2: CGImage) -> Double {
    guard image1.width == image2.width, image1.height == image2.height else {
      return -1
    }

    let width = image1.width
    let height = image1.height
    let bytesPerPixel = 4
    let bytesPerRow = width * bytesPerPixel
    let totalBytes = height * bytesPerRow

    var data1 = [UInt8](repeating: 0, count: totalBytes)
    var data2 = [UInt8](repeating: 0, count: totalBytes)

    let colorSpace = CGColorSpaceCreateDeviceRGB()
    let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)

    guard
      let ctx1 = CGContext(
        data: &data1,
        width: width,
        height: height,
        bitsPerComponent: 8,
        bytesPerRow: bytesPerRow,
        space: colorSpace,
        bitmapInfo: bitmapInfo.rawValue
      ),
      let ctx2 = CGContext(
        data: &data2,
        width: width,
        height: height,
        bitsPerComponent: 8,
        bytesPerRow: bytesPerRow,
        space: colorSpace,
        bitmapInfo: bitmapInfo.rawValue
      )
    else {
      return -1
    }

    ctx1.draw(image1, in: CGRect(x: 0, y: 0, width: width, height: height))
    ctx2.draw(image2, in: CGRect(x: 0, y: 0, width: width, height: height))

    // Compute MSE over all pixels and channels
    var mse: Double = 0
    for i in 0..<totalBytes {
      let diff = Double(Int(data1[i]) - Int(data2[i]))
      mse += diff * diff
    }
    mse /= Double(totalBytes)

    guard mse > 0 else {
      return Double.infinity  // Images are identical
    }

    // PSNR = 20 * log10(MAX) - 10 * log10(MSE)
    // For 8-bit: MAX = 255
    return 20.0 * log10(255.0) - 10.0 * log10(mse)
  }

#endif  // INTEGRATION_TESTS
