import MLX
import Testing

@testable import PixArtBackbone

@Suite("Embeddings")
struct EmbeddingsTests {

  // MARK: - 2D Sinusoidal Position Embeddings

  @Test("2D sinusoidal embedding output shape: [1, gridH * gridW, hiddenSize]")
  func positionEmbeddingShape() {
    let gridH = 4
    let gridW = 6
    let hiddenSize = 32  // small for test speed
    let result = get2DSinusoidalPositionEmbeddings(
      gridH: gridH,
      gridW: gridW,
      hiddenSize: hiddenSize,
      peInterpolation: 2.0,
      baseSize: 8
    )
    eval(result)
    #expect(result.dim(0) == 1)
    #expect(result.dim(1) == gridH * gridW)
    #expect(result.dim(2) == hiddenSize)
  }

  @Test("2D sinusoidal embedding shape: square grid")
  func positionEmbeddingSquareGrid() {
    let gridH = 8
    let gridW = 8
    let hiddenSize = 16
    let result = get2DSinusoidalPositionEmbeddings(
      gridH: gridH,
      gridW: gridW,
      hiddenSize: hiddenSize,
      peInterpolation: 2.0,
      baseSize: 8
    )
    eval(result)
    #expect(result.dim(0) == 1)
    #expect(result.dim(1) == 64)
    #expect(result.dim(2) == 16)
  }

  @Test("2D sinusoidal embedding ndim is 3")
  func positionEmbeddingNdim() {
    let result = get2DSinusoidalPositionEmbeddings(
      gridH: 4,
      gridW: 4,
      hiddenSize: 16,
      peInterpolation: 1.0,
      baseSize: 4
    )
    eval(result)
    #expect(result.ndim == 3)
  }

  // MARK: - 1D Sinusoidal Embedding

  @Test("1D sinusoidal embedding shape: [N, dim]")
  func sinusoidal1DShape() {
    let n = 5
    let dim = 16
    let positions = MLXArray(0..<n).asType(.float32)
    let result = sinusoidalEmbedding1D(positions: positions, dim: dim)
    eval(result)
    #expect(result.dim(0) == n)
    #expect(result.dim(1) == dim)
  }

  // MARK: - Timestep Sinusoidal Embedding

  @Test("Timestep sinusoidal embedding shape: [B, 256]")
  func timestepEmbeddingShape() {
    let B = 2
    let timestep = MLXArray([100.0, 500.0] as [Float])
    let result = timestepSinusoidalEmbedding(timestep)
    eval(result)
    #expect(result.dim(0) == B)
    #expect(result.dim(1) == 256)
  }

  @Test("Timestep sinusoidal embedding custom dim")
  func timestepEmbeddingCustomDim() {
    let B = 3
    let timestep = MLXArray([10.0, 20.0, 30.0] as [Float])
    let result = timestepSinusoidalEmbedding(timestep, dim: 128)
    eval(result)
    #expect(result.dim(0) == B)
    #expect(result.dim(1) == 128)
  }

  // MARK: - TimestepEmbedder MLP

  @Test("TimestepEmbedder output shape: [B, hiddenSize]")
  func timestepEmbedderShape() {
    let B = 2
    let hiddenSize = 64
    let embedder = TimestepEmbedder(hiddenSize: hiddenSize, frequencyDim: 32)
    let input = MLXArray.zeros([B, 32])
    let result = embedder(input)
    eval(result)
    #expect(result.dim(0) == B)
    #expect(result.dim(1) == hiddenSize)
  }

  // MARK: - SizeEmbedder

  @Test("SizeEmbedder output shape: [B, 768]")
  func sizeEmbedderShape() {
    let B = 2
    let embedder = SizeEmbedder()
    // Input: [B, 2] where columns are (height, width)
    // Create flat array and reshape to [B, 2]
    let sizeInput = MLXArray([512.0, 512.0, 256.0, 768.0] as [Float], [B, 2])
    let result = embedder(sizeInput)
    eval(result)
    #expect(result.dim(0) == B)
    #expect(result.dim(1) == 768)
  }

  // MARK: - AspectRatioEmbedder

  @Test("AspectRatioEmbedder output shape: [B, 384]")
  func aspectRatioEmbedderShape() {
    let B = 2
    let embedder = AspectRatioEmbedder()
    let arInput = MLXArray([1.0, 0.5] as [Float])
    let result = embedder(arInput)
    eval(result)
    #expect(result.dim(0) == B)
    #expect(result.dim(1) == 384)
  }

  // MARK: - CaptionProjection

  @Test("CaptionProjection output shape: [B, seqLen, hiddenSize]")
  func captionProjectionShape() {
    let B = 1
    let seqLen = 10
    let captionChannels = 32
    let hiddenSize = 16
    let proj = CaptionProjection(captionChannels: captionChannels, hiddenSize: hiddenSize)
    let input = MLXArray.zeros([B, seqLen, captionChannels])
    let result = proj(input)
    eval(result)
    #expect(result.dim(0) == B)
    #expect(result.dim(1) == seqLen)
    #expect(result.dim(2) == hiddenSize)
  }

  @Test("CaptionProjection preserves batch and sequence dims")
  func captionProjectionPreservesDims() {
    let B = 2
    let seqLen = 5
    let captionChannels = 16
    let hiddenSize = 8
    let proj = CaptionProjection(captionChannels: captionChannels, hiddenSize: hiddenSize)
    let input = MLXArray.zeros([B, seqLen, captionChannels])
    let result = proj(input)
    eval(result)
    #expect(result.ndim == 3)
    #expect(result.dim(0) == B)
    #expect(result.dim(1) == seqLen)
    #expect(result.dim(2) == hiddenSize)
  }
}
