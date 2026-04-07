@preconcurrency import MLX
import MLXNN

// MARK: - GEGLU Feed-Forward Network

/// GEGLU FFN: fc1 projects to 2 * ffnDim, splits for gate * value, then fc2 projects back.
///
/// Uses GELU(tanh) approximation for the gating activation.
///
/// Input: [B, T, C] -> Output: [B, T, C]
final class GEGLUFFN: Module, @unchecked Sendable {
  @ModuleInfo var fc1: Linear
  @ModuleInfo var fc2: Linear

  init(hiddenSize: Int, ffnHiddenSize: Int) {
    // fc1 projects to 2 * ffnHiddenSize for GEGLU split
    self._fc1.wrappedValue = Linear(hiddenSize, 2 * ffnHiddenSize)
    self._fc2.wrappedValue = Linear(ffnHiddenSize, hiddenSize)
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    // Project to 2 * ffnHiddenSize: [B, T, 2 * ffnDim]
    let projected = fc1(x)

    // Split into gate and value along last dimension
    let chunks = projected.split(parts: 2, axis: -1)
    let gate = chunks[0]  // [B, T, ffnDim]
    let value = chunks[1]  // [B, T, ffnDim]

    // GEGLU: GELU(tanh)(gate) * value
    // geluApproximate uses compile(shapeless:true) which can return 0-D tensors under
    // memory pressure. Replace with direct gelu_new (tanh approximation) math.
    let activated = gate * 0.5 * (1.0 + MLX.tanh(0.7978845608 * (gate + 0.044715 * gate * gate * gate))) * value

    // Project back: [B, T, hiddenSize]
    return fc2(activated)
  }
}

// MARK: - DiT Block

/// A single DiT transformer block with AdaLN-Zero conditioning.
///
/// Structure: Self-Attention -> Cross-Attention -> FFN
///
/// The 6 modulation parameters per block (from scale_shift_table + t_block output):
/// 1. shift_msa — additive shift before self-attention
/// 2. scale_msa — multiplicative scale before self-attention
/// 3. gate_msa — multiplicative gate after self-attention (zero-initialized)
/// 4. shift_mlp — additive shift before FFN
/// 5. scale_mlp — multiplicative scale before FFN
/// 6. gate_mlp — multiplicative gate after FFN (zero-initialized)
///
/// Cross-attention receives NO timestep modulation.
final class DiTBlock: Module, @unchecked Sendable {
  let norm1: LayerNorm
  let attn: SelfAttention
  @ModuleInfo(key: "cross_attn") var crossAttn: CrossAttention
  let norm2: LayerNorm
  let mlp: GEGLUFFN
  let scaleShiftTable: MLXArray

  init(hiddenSize: Int, numHeads: Int, headDim: Int, mlpRatio: Float) {
    let ffnHiddenSize = Int(Float(hiddenSize) * mlpRatio)

    // LayerNorm without learned affine parameters (elementwise_affine=False)
    self.norm1 = LayerNorm(dimensions: hiddenSize, eps: 1e-6, affine: false, bias: false)
    self.attn = SelfAttention(hiddenSize: hiddenSize, numHeads: numHeads, headDim: headDim)
    self._crossAttn.wrappedValue = CrossAttention(
      hiddenSize: hiddenSize, numHeads: numHeads, headDim: headDim)
    self.norm2 = LayerNorm(dimensions: hiddenSize, eps: 1e-6, affine: false, bias: false)
    self.mlp = GEGLUFFN(hiddenSize: hiddenSize, ffnHiddenSize: ffnHiddenSize)

    // 6 modulation parameters per block, shape [6, hiddenSize]
    self.scaleShiftTable = MLXArray.zeros([6, hiddenSize])
  }

  /// Forward pass through a single DiT block.
  ///
  /// - Parameters:
  ///   - x: Image token sequence, shape [B, T, C].
  ///   - y: Projected text embeddings, shape [B, T_text, C].
  ///   - t: Block conditioning from t_block, shape [B, 6 * C].
  ///   - mask: Text attention mask, shape [B, T_text].
  /// - Returns: Updated token sequence, shape [B, T, C].
  func callAsFunction(_ x: MLXArray, y: MLXArray, t: MLXArray, mask: MLXArray?) -> MLXArray {
    let B = x.dim(0)

    // Unpack 6 modulation params: scale_shift_table[None] + t.reshape(B, 6, -1)
    // scaleShiftTable: [6, C], t: [B, 6*C] -> [B, 6, C]
    let tReshaped = t.reshaped(B, 6, -1)
    let modulation = scaleShiftTable.expandedDimensions(axis: 0) + tReshaped  // [B, 6, C]

    // Extract each parameter: [B, 1, C] for broadcasting with [B, T, C]
    let shiftMsa = modulation[0..., 0...0, 0...]  // [B, 1, C]
    let scaleMsa = modulation[0..., 1...1, 0...]  // [B, 1, C]
    let gateMsa = modulation[0..., 2...2, 0...]  // [B, 1, C]
    let shiftMlp = modulation[0..., 3...3, 0...]  // [B, 1, C]
    let scaleMlp = modulation[0..., 4...4, 0...]  // [B, 1, C]
    let gateMlp = modulation[0..., 5...5, 0...]  // [B, 1, C]

    var out = x

    // 1. Self-Attention with AdaLN-Zero
    let normed1 = t2iModulate(norm1(out), shift: shiftMsa, scale: scaleMsa)
    out = out + gateMsa * attn(normed1)

    // 2. Cross-Attention (NO modulation, NO norm on query)
    out = out + crossAttn(out, context: y, mask: mask)

    // 3. FFN with AdaLN-Zero
    let normed2 = t2iModulate(norm2(out), shift: shiftMlp, scale: scaleMlp)
    out = out + gateMlp * mlp(normed2)

    return out
  }
}
