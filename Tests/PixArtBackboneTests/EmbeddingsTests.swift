import Foundation
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

  // MARK: - Numerical correctness pins
  //
  // The three sinusoidal embedding helpers below have known compatibility
  // footguns that silently degrade denoising quality if regressed:
  //   - timestepSinusoidalEmbedding: must use [cos, sin] order (flip_sin_to_cos=True)
  //     and denominator = halfDim (NOT halfDim - 1, i.e. downscale_freq_shift=0).
  //   - sinusoidalEmbedding1D: must use [sin, cos] order (diffusers convention).
  //   - get2DSinusoidalPositionEmbeddings: W-first / H-second concat order.
  //
  // These tests pin those numerical contracts so a refactor that flips an
  // axis or swaps a denominator fails loudly here instead of silently in
  // image quality.

  /// Helper: read a Float32 scalar at [row, col] from a 2D MLXArray.
  private func scalarAt(_ array: MLXArray, _ row: Int, _ col: Int) -> Float {
    eval(array)
    return array[row, col].asArray(Float.self)[0]
  }

  /// Helper: read a Float32 scalar at [b, n, d] from a 3D MLXArray.
  private func scalarAt3D(_ array: MLXArray, _ b: Int, _ n: Int, _ d: Int) -> Float {
    eval(array)
    return array[b, n, d].asArray(Float.self)[0]
  }

  // MARK: timestep order + denominator pins

  @Test("timestepSinusoidalEmbedding(t=0, dim=4) == [1, 1, 0, 0] (cos FIRST)")
  func timestepZero_pinsCosFirst() {
    // halfDim=2, freqs=[1.0, 0.01]; angles=[0,0].
    // [cos, sin] = [1, 1, 0, 0]. If order is wrongly [sin, cos], result is [0,0,1,1].
    let result = timestepSinusoidalEmbedding(MLXArray([0.0] as [Float]), dim: 4)
    eval(result)
    let tol: Float = 1e-6
    #expect(abs(scalarAt(result, 0, 0) - 1.0) < tol)
    #expect(abs(scalarAt(result, 0, 1) - 1.0) < tol)
    #expect(abs(scalarAt(result, 0, 2) - 0.0) < tol)
    #expect(abs(scalarAt(result, 0, 3) - 0.0) < tol)
  }

  @Test("timestepSinusoidalEmbedding(t=pi, dim=4) leads with cos(pi)=-1 (cos-first regression pin)")
  func timestepPi_pinsCosFirstViaSign() {
    // halfDim=2, freqs=[1.0, 0.01]; angles=[pi, pi*0.01].
    // cos-first: out[0,0]=cos(pi)=-1, out[0,2]=sin(pi)~=0.
    // If sin-first regression: out[0,0]=sin(pi)~=0, NOT -1.
    let result = timestepSinusoidalEmbedding(MLXArray([Float.pi]), dim: 4)
    eval(result)
    let tol: Float = 1e-5
    #expect(abs(scalarAt(result, 0, 0) - (-1.0)) < tol)  // cos(pi)
    // sin(pi) is ~1.2e-7 in float32; allow loose tol.
    #expect(abs(scalarAt(result, 0, 2) - 0.0) < 1e-4)
    // Sanity on the other halves.
    #expect(abs(scalarAt(result, 0, 1) - Foundation.cos(Float.pi * 0.01)) < tol)
    #expect(abs(scalarAt(result, 0, 3) - Foundation.sin(Float.pi * 0.01)) < tol)
  }

  @Test("timestepSinusoidalEmbedding denominator is halfDim, NOT halfDim-1")
  func timestepDenominator_pinsHalfDim() {
    // dim=4 → halfDim=2.
    // Correct denominator (halfDim=2): freqs[1] = exp(-log(10000)/2) ≈ 0.01.
    //   For t=10000: angle[1] = 100, output[0,1] = cos(100) ≈ 0.86231.
    // Wrong denominator (halfDim-1=1): freqs[1] = exp(-log(10000)) = 1e-4.
    //   For t=10000: angle[1] = 1, output[0,1] = cos(1) ≈ 0.5403.
    let result = timestepSinusoidalEmbedding(MLXArray([10000.0] as [Float]), dim: 4)
    eval(result)
    let expected = Foundation.cos(Float(100.0))  // ≈ 0.86231887
    let actual = scalarAt(result, 0, 1)
    // Loose tolerance: cos(100) lives on a fast-rotating phase; float32 path
    // through MLX exp/mul accumulates ~1e-3 error.
    #expect(abs(actual - expected) < 1e-3,
            "expected cos(100)=\(expected), got \(actual). If regressed to halfDim-1, would be ~cos(1)=0.5403.")
  }

  // MARK: sinusoidalEmbedding1D order pin

  @Test("sinusoidalEmbedding1D(pos=0, dim=4) == [0, 0, 1, 1] (sin FIRST, opposite of timestep)")
  func sinusoidal1DZero_pinsSinFirst() {
    // halfDim=2, freqs=[1.0, 0.01]; angles=[[0,0]].
    // [sin, cos] = [0, 0, 1, 1]. If order is wrongly [cos, sin], result is [1,1,0,0].
    let result = sinusoidalEmbedding1D(positions: MLXArray([0.0] as [Float]), dim: 4)
    eval(result)
    let tol: Float = 1e-6
    #expect(abs(scalarAt(result, 0, 0) - 0.0) < tol)
    #expect(abs(scalarAt(result, 0, 1) - 0.0) < tol)
    #expect(abs(scalarAt(result, 0, 2) - 1.0) < tol)
    #expect(abs(scalarAt(result, 0, 3) - 1.0) < tol)
  }

  // MARK: 2D position embedding W-first/H-second pin

  @Test("get2DSinusoidalPositionEmbeddings concatenates W FIRST then H")
  func position2D_pinsWFirstHSecond() {
    // gridH=1, gridW=2, hiddenSize=8 → halfDim per axis = 4.
    // Inside sinusoidalEmbedding1D(_, dim:4): innerHalf=2, freqs=[1.0, 0.01].
    //
    // hCoords = arange(1)/(1/1)/1 = [0]
    //   embedH = [[sin(0), sin(0), cos(0), cos(0)]] = [[0, 0, 1, 1]]
    // wCoords = arange(2)/(2/1)/1 = [0, 0.5]
    //   embedW = [[0, 0, 1, 1],
    //             [sin(0.5), sin(0.005), cos(0.5), cos(0.005)]]
    //
    // Output shape [1, 2, 8]. Concat order is [W, H] along last axis, so:
    //   out[0, 1, :] = [embedW[1] (4 dims), embedH[0] (4 dims)]
    //                = [sin(0.5), sin(0.005), cos(0.5), cos(0.005), 0, 0, 1, 1]
    //
    // Regression pin: if H/W are swapped, out[0,1,0] would be 0 (H position 0)
    // instead of sin(0.5) ≈ 0.4794.
    let result = get2DSinusoidalPositionEmbeddings(
      gridH: 1,
      gridW: 2,
      hiddenSize: 8,
      peInterpolation: 1.0,
      baseSize: 1
    )
    eval(result)
    #expect(result.dim(0) == 1)
    #expect(result.dim(1) == 2)
    #expect(result.dim(2) == 8)

    let tol: Float = 1e-5

    // At (h=0, w=1): first 4 dims are W's embedding for position 0.5 (sin-first),
    // last 4 dims are H's embedding for position 0 = [0,0,1,1].
    #expect(abs(scalarAt3D(result, 0, 1, 0) - Foundation.sin(Float(0.5))) < tol)   // W sin(0.5)
    #expect(abs(scalarAt3D(result, 0, 1, 1) - Foundation.sin(Float(0.005))) < tol) // W sin(0.005)
    #expect(abs(scalarAt3D(result, 0, 1, 2) - Foundation.cos(Float(0.5))) < tol)   // W cos(0.5)
    #expect(abs(scalarAt3D(result, 0, 1, 3) - Foundation.cos(Float(0.005))) < tol) // W cos(0.005)
    #expect(abs(scalarAt3D(result, 0, 1, 4) - 0.0) < tol)                           // H sin(0)
    #expect(abs(scalarAt3D(result, 0, 1, 5) - 0.0) < tol)                           // H sin(0)
    #expect(abs(scalarAt3D(result, 0, 1, 6) - 1.0) < tol)                           // H cos(0)
    #expect(abs(scalarAt3D(result, 0, 1, 7) - 1.0) < tol)                           // H cos(0)

    // At (h=0, w=0): both halves are zero-position embeddings = [0,0,1,1, 0,0,1,1].
    #expect(abs(scalarAt3D(result, 0, 0, 0) - 0.0) < tol)
    #expect(abs(scalarAt3D(result, 0, 0, 2) - 1.0) < tol)
    #expect(abs(scalarAt3D(result, 0, 0, 4) - 0.0) < tol)
    #expect(abs(scalarAt3D(result, 0, 0, 6) - 1.0) < tol)
  }

  // MARK: SizeEmbedder shared-MLP semantic pin

  @Test("SizeEmbedder uses ONE shared MLP for both H and W (halves equal for h==w)")
  func sizeEmbedder_pinsSharedMLP() {
    // If H and W go through the same MicroConditionEmbedder instance with the
    // same input value, the two output halves must be exactly equal. A
    // regression that introduces a second embedder would produce different
    // weights and break this equality.
    let outputDimPerAxis = 384
    let embedder = SizeEmbedder()  // defaults: frequencyDim=256, outputDimPerAxis=384
    // Batch=1, [h, w] = [512, 512].
    let input = MLXArray([512.0, 512.0] as [Float], [1, 2])
    let result = embedder(input)
    eval(result)
    #expect(result.dim(0) == 1)
    #expect(result.dim(1) == outputDimPerAxis * 2)

    // Sample several positions across both halves and verify equality.
    // Tolerance is tight because both halves should be the *same* MLP applied
    // to the *same* scalar; only nondeterministic kernels could differ.
    let tol: Float = 1e-5
    let probeIndices = [0, 1, 17, 100, 200, 383]
    for i in probeIndices {
      let lhs = scalarAt(result, 0, i)
      let rhs = scalarAt(result, 0, i + outputDimPerAxis)
      #expect(abs(lhs - rhs) < tol,
              "SizeEmbedder halves diverged at offset \(i): \(lhs) vs \(rhs). Did it stop sharing the MLP?")
    }
  }

}
