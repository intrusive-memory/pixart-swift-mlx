@preconcurrency import MLX
import MLXNN

// MARK: - AdaLN Modulation Helper

/// AdaLN-Zero modulation: `x * (1 + scale) + shift`.
///
/// Used by both `DiTBlock` and `FinalLayer` for adaptive layer normalization.
/// Defined once here and reused everywhere.
///
/// - Parameters:
///   - x: Input tensor, shape [B, T, C].
///   - shift: Additive shift, shape [B, 1, C].
///   - scale: Multiplicative scale, shape [B, 1, C].
/// - Returns: Modulated tensor, shape [B, T, C].
func t2iModulate(_ x: MLXArray, shift: MLXArray, scale: MLXArray) -> MLXArray {
    x * (1 + scale) + shift
}

// MARK: - Final Layer

/// Final layer of the PixArt DiT backbone.
///
/// Applies AdaLN with 2 modulation parameters (shift + scale, no gate),
/// then projects from hiddenSize to patchSize^2 * outChannels,
/// and unpatchifies back to spatial dimensions.
///
/// The scale_shift_table has shape [2, hiddenSize] (2 params: shift and scale).
final class FinalLayer: Module, @unchecked Sendable {
    let normFinal: LayerNorm
    @ModuleInfo(key: "linear") var linear: Linear
    let scaleShiftTable: MLXArray
    let patchSize: Int
    let outChannels: Int

    init(hiddenSize: Int, patchSize: Int, outChannels: Int) {
        self.normFinal = LayerNorm(dimensions: hiddenSize, eps: 1e-6, affine: false, bias: false)
        self._linear.wrappedValue = Linear(hiddenSize, patchSize * patchSize * outChannels)
        self.scaleShiftTable = MLXArray.zeros([2, hiddenSize])
        self.patchSize = patchSize
        self.outChannels = outChannels
    }

    /// Forward pass through the final layer.
    ///
    /// - Parameters:
    ///   - x: Token sequence, shape [B, T, hiddenSize].
    ///   - t: Raw timestep embedding (before t_block), shape [B, hiddenSize].
    ///   - gridH: Spatial height of the token grid (H / patchSize).
    ///   - gridW: Spatial width of the token grid (W / patchSize).
    /// - Returns: Spatial output, shape [B, gridH * patchSize, gridW * patchSize, outChannels].
    func callAsFunction(_ x: MLXArray, t: MLXArray, gridH: Int, gridW: Int) -> MLXArray {
        let B = x.dim(0)

        // Unpack 2 modulation params: shift and scale
        // scaleShiftTable: [2, C] + t: [B, C] -> [B, 2, C] -> split into 2 x [B, 1, C]
        let modulation = scaleShiftTable.expandedDimensions(axis: 0) + t.reshaped(B, 1, -1)
        let shift = modulation[0..., 0...0, 0...]  // [B, 1, C]
        let scale = modulation[0..., 1...1, 0...]  // [B, 1, C]

        // Apply AdaLN (shift + scale, no gate)
        var out = normFinal(x)
        out = t2iModulate(out, shift: shift, scale: scale)

        // Project: [B, T, hiddenSize] -> [B, T, patchSize^2 * outChannels]
        out = linear(out)

        // Unpatchify: [B, T, p*p*C_out] -> [B, gridH*p, gridW*p, C_out]
        out = unpatchify(out, gridH: gridH, gridW: gridW)

        return out
    }

    /// Rearranges tokens back to spatial dimensions.
    ///
    /// [B, gridH * gridW, patchSize * patchSize * outChannels]
    /// -> [B, gridH, gridW, patchSize, patchSize, outChannels]
    /// -> [B, gridH * patchSize, gridW * patchSize, outChannels]
    private func unpatchify(_ x: MLXArray, gridH: Int, gridW: Int) -> MLXArray {
        let B = x.dim(0)
        let p = patchSize
        let c = outChannels

        // [B, gridH * gridW, p * p * c] -> [B, gridH, gridW, p, p, c]
        var out = x.reshaped(B, gridH, gridW, p, p, c)

        // Rearrange: [B, gridH, gridW, p, p, c] -> [B, gridH, p, gridW, p, c]
        out = out.transposed(0, 1, 3, 2, 4, 5)

        // Merge spatial dimensions: [B, gridH * p, gridW * p, c]
        out = out.reshaped(B, gridH * p, gridW * p, c)

        return out
    }
}
