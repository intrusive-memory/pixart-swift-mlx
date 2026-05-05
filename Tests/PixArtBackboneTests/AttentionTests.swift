import MLX
import Testing

@testable import PixArtBackbone

@Suite("Attention")
struct AttentionTests {

  // MARK: - SelfAttention

  @Test("SelfAttention output shape matches input shape [B, T, C]")
  func selfAttentionOutputShape() {
    let B = 1
    let T = 8
    let hiddenSize = 16
    let numHeads = 2
    let headDim = hiddenSize / numHeads
    let attn = SelfAttention(hiddenSize: hiddenSize, numHeads: numHeads, headDim: headDim)
    let input = MLXArray.zeros([B, T, hiddenSize])
    let result = attn(input)
    eval(result)
    #expect(result.dim(0) == B)
    #expect(result.dim(1) == T)
    #expect(result.dim(2) == hiddenSize)
  }

  @Test("SelfAttention preserves hidden dimension")
  func selfAttentionPreservesHiddenDim() {
    let hiddenSize = 32
    let numHeads = 4
    let headDim = hiddenSize / numHeads
    let attn = SelfAttention(hiddenSize: hiddenSize, numHeads: numHeads, headDim: headDim)
    let input = MLXArray.zeros([2, 16, hiddenSize])
    let result = attn(input)
    eval(result)
    #expect(result.dim(2) == hiddenSize)
  }

  // MARK: - CrossAttention

  @Test("CrossAttention output shape: [B, T_img, C]")
  func crossAttentionOutputShape() {
    let B = 1
    let Timg = 8
    let Ttext = 5
    let hiddenSize = 16
    let numHeads = 2
    let headDim = hiddenSize / numHeads
    let crossAttn = CrossAttention(hiddenSize: hiddenSize, numHeads: numHeads, headDim: headDim)
    let query = MLXArray.zeros([B, Timg, hiddenSize])
    let context = MLXArray.zeros([B, Ttext, hiddenSize])
    let result = crossAttn(query, context: context, mask: nil)
    eval(result)
    #expect(result.dim(0) == B)
    #expect(result.dim(1) == Timg)
    #expect(result.dim(2) == hiddenSize)
  }

  @Test("CrossAttention output image sequence length is independent of text length")
  func crossAttentionQKVSplit() {
    let B = 1
    let Timg = 10
    let TtextShort = 3
    let TtextLong = 20
    let hiddenSize = 16
    let numHeads = 2
    let headDim = hiddenSize / numHeads
    let crossAttn = CrossAttention(hiddenSize: hiddenSize, numHeads: numHeads, headDim: headDim)
    let query = MLXArray.zeros([B, Timg, hiddenSize])
    let contextShort = MLXArray.zeros([B, TtextShort, hiddenSize])
    let contextLong = MLXArray.zeros([B, TtextLong, hiddenSize])
    let resultShort = crossAttn(query, context: contextShort, mask: nil)
    let resultLong = crossAttn(query, context: contextLong, mask: nil)
    eval(resultShort, resultLong)
    // Both results should have the image sequence length, not text sequence length
    #expect(resultShort.dim(1) == Timg)
    #expect(resultLong.dim(1) == Timg)
  }

  @Test("CrossAttention with attention mask produces correct output shape")
  func crossAttentionWithMask() {
    let B = 1
    let Timg = 6
    let Ttext = 4
    let hiddenSize = 16
    let numHeads = 2
    let headDim = hiddenSize / numHeads
    let crossAttn = CrossAttention(hiddenSize: hiddenSize, numHeads: numHeads, headDim: headDim)
    let query = MLXArray.zeros([B, Timg, hiddenSize])
    let context = MLXArray.zeros([B, Ttext, hiddenSize])
    // Mask: 1 = real token, 0 = padding
    let mask = MLXArray([1, 1, 1, 0] as [Int32]).reshaped(B, Ttext)
    let result = crossAttn(query, context: context, mask: mask)
    eval(result)
    #expect(result.dim(0) == B)
    #expect(result.dim(1) == Timg)
    #expect(result.dim(2) == hiddenSize)
  }

}
