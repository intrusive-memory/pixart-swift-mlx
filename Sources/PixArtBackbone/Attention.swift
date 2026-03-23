import Foundation
@preconcurrency import MLX
import MLXNN

// MARK: - Self-Attention

/// Multi-head self-attention with QK normalization (PixArt-Sigma enables this).
///
/// Separate Q, K, V projections (matching diffusers weight format).
/// After reshape to multi-head form, LayerNorm is applied to Q and K.
///
/// Input: [B, T, C] -> Output: [B, T, C]
final class SelfAttention: Module, @unchecked Sendable {
  @ModuleInfo(key: "to_q") var toQ: Linear
  @ModuleInfo(key: "to_k") var toK: Linear
  @ModuleInfo(key: "to_v") var toV: Linear
  @ModuleInfo(key: "to_out") var toOut: Linear
  @ModuleInfo(key: "q_norm") var qNorm: LayerNorm
  @ModuleInfo(key: "k_norm") var kNorm: LayerNorm

  let numHeads: Int
  let headDim: Int
  let scale: Float

  init(hiddenSize: Int, numHeads: Int, headDim: Int) {
    self.numHeads = numHeads
    self.headDim = headDim
    self.scale = 1.0 / Foundation.sqrt(Float(headDim))

    self._toQ.wrappedValue = Linear(hiddenSize, hiddenSize)
    self._toK.wrappedValue = Linear(hiddenSize, hiddenSize)
    self._toV.wrappedValue = Linear(hiddenSize, hiddenSize)
    self._toOut.wrappedValue = Linear(hiddenSize, hiddenSize)
    self._qNorm.wrappedValue = LayerNorm(dimensions: headDim, eps: 1e-6)
    self._kNorm.wrappedValue = LayerNorm(dimensions: headDim, eps: 1e-6)
  }

  /// - Parameter x: Input tensor of shape [B, T, C].
  /// - Returns: Output tensor of shape [B, T, C].
  func callAsFunction(_ x: MLXArray) -> MLXArray {
    let B = x.dim(0)
    let T = x.dim(1)

    // Project to Q, K, V: [B, T, C] -> [B, T, C]
    var q = toQ(x)
    var k = toK(x)
    let v = toV(x)

    // Reshape to multi-head: [B, T, numHeads, headDim] -> [B, numHeads, T, headDim]
    q = q.reshaped(B, T, numHeads, headDim).transposed(0, 2, 1, 3)
    k = k.reshaped(B, T, numHeads, headDim).transposed(0, 2, 1, 3)
    let vReshaped = v.reshaped(B, T, numHeads, headDim).transposed(0, 2, 1, 3)

    // QK normalization (PixArt-Sigma specific)
    q = qNorm(q)
    k = kNorm(k)

    // Scaled dot-product attention: [B, numHeads, T, headDim]
    let attnOut = MLXFast.scaledDotProductAttention(
      queries: q, keys: k, values: vReshaped, scale: scale, mask: nil
    )

    // Reshape back: [B, numHeads, T, headDim] -> [B, T, C]
    let output = attnOut.transposed(0, 2, 1, 3).reshaped(B, T, numHeads * headDim)

    // Output projection
    return toOut(output)
  }
}

// MARK: - Cross-Attention

/// Multi-head cross-attention: Q from image tokens, K/V from projected text embeddings.
///
/// Cross-attention receives NO timestep modulation (no AdaLN shift/scale/gate).
///
/// Input: query [B, T_img, C], context [B, T_text, C], mask [B, T_text]
/// Output: [B, T_img, C]
final class CrossAttention: Module, @unchecked Sendable {
  @ModuleInfo(key: "to_q") var toQ: Linear
  @ModuleInfo(key: "to_k") var toK: Linear
  @ModuleInfo(key: "to_v") var toV: Linear
  @ModuleInfo(key: "to_out") var toOut: Linear

  let numHeads: Int
  let headDim: Int
  let scale: Float

  init(hiddenSize: Int, numHeads: Int, headDim: Int) {
    self.numHeads = numHeads
    self.headDim = headDim
    self.scale = 1.0 / Foundation.sqrt(Float(headDim))

    self._toQ.wrappedValue = Linear(hiddenSize, hiddenSize)
    self._toK.wrappedValue = Linear(hiddenSize, hiddenSize)
    self._toV.wrappedValue = Linear(hiddenSize, hiddenSize)
    self._toOut.wrappedValue = Linear(hiddenSize, hiddenSize)
  }

  /// - Parameters:
  ///   - x: Image tokens, shape [B, T_img, C].
  ///   - context: Projected text embeddings, shape [B, T_text, C].
  ///   - mask: Text attention mask, shape [B, T_text]. 1 = real token, 0 = padding.
  /// - Returns: Output tensor of shape [B, T_img, C].
  func callAsFunction(_ x: MLXArray, context: MLXArray, mask: MLXArray?) -> MLXArray {
    let B = x.dim(0)
    let Timg = x.dim(1)
    let Ttext = context.dim(1)

    // Q from image, K/V from text
    var q = toQ(x)  // [B, T_img, C]
    var k = toK(context)  // [B, T_text, C]
    let v = toV(context)  // [B, T_text, C]

    // Reshape to multi-head
    q = q.reshaped(B, Timg, numHeads, headDim).transposed(0, 2, 1, 3)
    k = k.reshaped(B, Ttext, numHeads, headDim).transposed(0, 2, 1, 3)
    let vReshaped = v.reshaped(B, Ttext, numHeads, headDim).transposed(0, 2, 1, 3)

    // Build attention mask if provided
    // mask: [B, T_text] -> [B, 1, 1, T_text] for broadcasting with [B, numHeads, T_img, T_text]
    let attnMask: MLXArray?
    if let mask {
      // Convert 0/1 mask to additive mask: 0 -> 0, 1 -> 0 (keep), 0 -> -inf (mask out)
      // Actually: 1 = real token (keep), 0 = padding (mask out)
      // For scaled_dot_product_attention, mask is additive: 0 = keep, -inf = mask
      let expandedMask = mask.expandedDimensions(axes: [1, 2])  // [B, 1, 1, T_text]
      attnMask = MLX.where(expandedMask .> 0, MLXArray(Float(0)), MLXArray(Float(-1e9)))
    } else {
      attnMask = nil
    }

    // Scaled dot-product attention
    let attnOut = MLXFast.scaledDotProductAttention(
      queries: q, keys: k, values: vReshaped, scale: scale, mask: attnMask
    )

    // Reshape back: [B, numHeads, T_img, headDim] -> [B, T_img, C]
    let output = attnOut.transposed(0, 2, 1, 3).reshaped(B, Timg, numHeads * headDim)

    // Output projection
    return toOut(output)
  }
}
