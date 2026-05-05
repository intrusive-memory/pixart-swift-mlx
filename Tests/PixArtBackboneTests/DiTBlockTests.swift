import MLX
import Testing

@testable import PixArtBackbone

@Suite("DiTBlock")
struct DiTBlockTests {

  // Use a small configuration to keep tests fast without real weights.
  // Requirements: hiddenSize divisible by numHeads, mlpRatio >= 1.0
  let hiddenSize = 16
  let numHeads = 2
  let headDim = 8  // hiddenSize / numHeads
  let mlpRatio: Float = 2.0

  @Test("DiTBlock output shape matches input shape [B, T, C]")
  func outputShapeMatchesInput() {
    let B = 1
    let T = 8
    let Ttext = 4

    let block = DiTBlock(
      hiddenSize: hiddenSize,
      numHeads: numHeads,
      headDim: headDim,
      mlpRatio: mlpRatio
    )

    let x = MLXArray.zeros([B, T, hiddenSize])
    let y = MLXArray.zeros([B, Ttext, hiddenSize])
    let t = MLXArray.zeros([B, 6 * hiddenSize])  // t_block output shape

    let result = block(x, y: y, t: t, mask: nil)
    eval(result)

    #expect(result.dim(0) == B)
    #expect(result.dim(1) == T)
    #expect(result.dim(2) == hiddenSize)
  }

  @Test("DiTBlock hidden dimension is preserved")
  func hiddenDimensionPreserved() {
    let B = 2
    let T = 12
    let Ttext = 6

    let block = DiTBlock(
      hiddenSize: hiddenSize,
      numHeads: numHeads,
      headDim: headDim,
      mlpRatio: mlpRatio
    )

    let x = MLXArray.zeros([B, T, hiddenSize])
    let y = MLXArray.zeros([B, Ttext, hiddenSize])
    let t = MLXArray.zeros([B, 6 * hiddenSize])

    let result = block(x, y: y, t: t, mask: nil)
    eval(result)

    // The key exit criterion: output hidden dim matches input hidden dim
    #expect(result.dim(2) == hiddenSize)
  }

  @Test("DiTBlock scaleShiftTable has shape [6, hiddenSize]")
  func scaleShiftTableShape() {
    let block = DiTBlock(
      hiddenSize: hiddenSize,
      numHeads: numHeads,
      headDim: headDim,
      mlpRatio: mlpRatio
    )
    eval(block.scaleShiftTable)
    #expect(block.scaleShiftTable.dim(0) == 6)
    #expect(block.scaleShiftTable.dim(1) == hiddenSize)
  }

  @Test("DiTBlock with attention mask produces correct output shape")
  func outputShapeWithMask() {
    let B = 1
    let T = 6
    let Ttext = 4

    let block = DiTBlock(
      hiddenSize: hiddenSize,
      numHeads: numHeads,
      headDim: headDim,
      mlpRatio: mlpRatio
    )

    let x = MLXArray.zeros([B, T, hiddenSize])
    let y = MLXArray.zeros([B, Ttext, hiddenSize])
    let t = MLXArray.zeros([B, 6 * hiddenSize])
    // mask: [B, T_text] with 1s for real tokens, 0s for padding
    let mask = MLXArray([1, 1, 0, 0] as [Int32]).reshaped(B, Ttext)

    let result = block(x, y: y, t: t, mask: mask)
    eval(result)

    #expect(result.dim(0) == B)
    #expect(result.dim(1) == T)
    #expect(result.dim(2) == hiddenSize)
  }

}
