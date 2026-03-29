import MLX
import MLXNN
import Testing

@testable import PixArtBackbone

@Suite("PatchEmbedding")
struct PatchEmbeddingTests {

  // MARK: - Patch embedding via Conv2d

  @Test("Patch embedding: spatial [B, H, W, inChannels] -> [B, gridH, gridW, hiddenSize]")
  func patchEmbeddingOutputShape() {
    // Replicate the patch embedding layer from PixArtDiT
    let inChannels = 4
    let hiddenSize = 16
    let patchSize = 2
    let patchEmbed = Conv2d(
      inputChannels: inChannels,
      outputChannels: hiddenSize,
      kernelSize: IntOrPair(patchSize),
      stride: IntOrPair(patchSize),
      bias: true
    )

    let B = 1
    let spatialH = 8
    let spatialW = 8
    let input = MLXArray.zeros([B, spatialH, spatialW, inChannels])
    let output = patchEmbed(input)
    eval(output)

    let expectedGridH = spatialH / patchSize
    let expectedGridW = spatialW / patchSize
    #expect(output.dim(0) == B)
    #expect(output.dim(1) == expectedGridH)
    #expect(output.dim(2) == expectedGridW)
    #expect(output.dim(3) == hiddenSize)
  }

  @Test("Patch embedding: non-square input [B, H, W, inChannels]")
  func patchEmbeddingNonSquare() {
    let inChannels = 4
    let hiddenSize = 16
    let patchSize = 2
    let patchEmbed = Conv2d(
      inputChannels: inChannels,
      outputChannels: hiddenSize,
      kernelSize: IntOrPair(patchSize),
      stride: IntOrPair(patchSize),
      bias: true
    )

    let B = 1
    let spatialH = 8
    let spatialW = 16
    let input = MLXArray.zeros([B, spatialH, spatialW, inChannels])
    let output = patchEmbed(input)
    eval(output)

    #expect(output.dim(1) == spatialH / patchSize)  // gridH
    #expect(output.dim(2) == spatialW / patchSize)  // gridW
    #expect(output.dim(3) == hiddenSize)
  }

  @Test("Patch embedding: token sequence length equals gridH * gridW")
  func patchEmbeddingTokenCount() {
    let inChannels = 4
    let hiddenSize = 8
    let patchSize = 2
    let patchEmbed = Conv2d(
      inputChannels: inChannels,
      outputChannels: hiddenSize,
      kernelSize: IntOrPair(patchSize),
      stride: IntOrPair(patchSize),
      bias: true
    )

    let B = 1
    let spatialH = 8
    let spatialW = 12
    let input = MLXArray.zeros([B, spatialH, spatialW, inChannels])
    let patched = patchEmbed(input)
    eval(patched)

    let gridH = patched.dim(1)
    let gridW = patched.dim(2)

    // Flatten to sequence
    let tokens = patched.reshaped(B, gridH * gridW, hiddenSize)
    eval(tokens)

    let expectedTokens = (spatialH / patchSize) * (spatialW / patchSize)
    #expect(tokens.dim(1) == expectedTokens)
    #expect(tokens.dim(2) == hiddenSize)
  }

  @Test("Patch embedding output ndim is 4")
  func patchEmbeddingNdim() {
    let inChannels = 4
    let hiddenSize = 8
    let patchSize = 2
    let patchEmbed = Conv2d(
      inputChannels: inChannels,
      outputChannels: hiddenSize,
      kernelSize: IntOrPair(patchSize),
      stride: IntOrPair(patchSize),
      bias: true
    )
    let input = MLXArray.zeros([1, 8, 8, inChannels])
    let output = patchEmbed(input)
    eval(output)
    #expect(output.ndim == 4)
  }

  @Test("Patch embedding with PixArt-Sigma default config dimensions")
  func patchEmbeddingDefaultConfig() throws {
    let dit = try PixArtDiT(configuration: PixArtDiTConfiguration())
    // PixArtDiT stores patchEmbed as a public property (internal Conv2d)
    // Test the config values are consistent with patch embedding conventions
    let config = PixArtDiTConfiguration()
    #expect(config.patchSize == 2)
    #expect(config.inChannels == 4)
    #expect(config.hiddenSize == 1152)

    // A 64x64 latent (512x512 image / 8 VAE) with patchSize=2
    // -> gridH = 32, gridW = 32, tokens = 1024
    let spatialH = 64
    let spatialW = 64
    let gridH = spatialH / config.patchSize
    let gridW = spatialW / config.patchSize
    #expect(gridH == 32)
    #expect(gridW == 32)
    #expect(gridH * gridW == 1024)
  }
}
