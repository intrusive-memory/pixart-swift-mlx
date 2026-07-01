import Foundation
import MLX
import MLXNN
import MLXRandom
import Testing
import Tuberia

@testable import PixArtBackbone

// MARK: - int4-in-memory tests
//
// Pins the "keep the DiT int4 in memory" contract added to `PixArtDiT.apply(weights:)`:
//
//   * A quantizable projection that ships an int4 sidecar (uint32 `.weight` +
//     `.scales` + `.biases`) is converted to `QuantizedLinear` and the packed
//     values are routed in directly — NO `dequantized(...).asType(.float16)`.
//   * The forward pass through the resulting `QuantizedLinear` is numerically
//     equivalent to the legacy path (a plain `Linear` loaded with the dequantized
//     fp16 weight), because `quantizedMM(packed)` == `dequantized(packed) @ x`.
//   * The resident weight bytes are int4-scale (~1/4 of fp16), not fp16.
//
// The parity test operates at `DiTBlock` granularity (the "DiT block output"
// referenced by the spec) and drives the exact `quantize(model:filter:)` +
// `update(parameters:)` mechanism that `PixArtDiT.apply(weights:)` uses.

@Suite("PixArtInt4InMemory", .serialized)
struct PixArtInt4InMemoryTests {

  private static let hidden = 1152
  private static let heads = 16
  private static let headDim = 72
  private static let mlpRatio: Float = 4

  /// Builds, from a block's own initialized parameters, both an int4 weight dict
  /// (packed uint32 weight + scales + biases per 2-D `.weight`) and the matching
  /// legacy fp16 dict (the same weights, dequantized). Both dicts describe the
  /// identical effective weights, so a forward through each must agree.
  private static func makeInt4AndFp16Dicts(
    from params: [(String, MLXArray)]
  ) -> (int4: [String: MLXArray], fp16: [String: MLXArray], quantizedBases: Set<String>) {
    var int4: [String: MLXArray] = [:]
    var fp16: [String: MLXArray] = [:]
    var quantizedBases = Set<String>()

    for (key, w) in params {
      if key.hasSuffix(".weight"), w.ndim == 2 {
        let base = String(key.dropLast(".weight".count))
        let q = quantized(w, groupSize: 64, bits: 4)
        int4[key] = q.wq
        int4[base + ".scales"] = q.scales
        int4[base + ".biases"] = q.biases!
        // Legacy path: materialize the dequantized weight in the original dtype.
        fp16[key] = dequantized(
          q.wq, scales: q.scales, biases: q.biases, groupSize: 64, bits: 4
        ).asType(w.dtype)
        quantizedBases.insert(base)
      } else {
        int4[key] = w
        fp16[key] = w
      }
    }
    return (int4, fp16, quantizedBases)
  }

  @Test(
    "apply-style int4 routing converts projections to QuantizedLinear (no fp16 weight materialized)"
  )
  func routingProducesQuantizedLinear() throws {
    let block = DiTBlock(
      hiddenSize: Self.hidden, numHeads: Self.heads, headDim: Self.headDim,
      mlpRatio: Self.mlpRatio)
    let (int4, _, quantizedBases) = Self.makeInt4AndFp16Dicts(
      from: block.parameters().flattened())

    quantize(model: block) { path, _ in
      quantizedBases.contains(path) ? (groupSize: 64, bits: 4, mode: .affine) : nil
    }
    block.update(parameters: ModuleParameters.unflattened(int4))

    // Every quantized base must now be a QuantizedLinear leaf with a uint32 weight.
    let leaves = Dictionary(uniqueKeysWithValues: block.leafModules().flattened())
    for base in quantizedBases {
      let leaf = try #require(leaves[base], "missing leaf \(base)")
      let ql = try #require(leaf as? QuantizedLinear, "\(base) is not QuantizedLinear")
      #expect(ql.weight.dtype == .uint32, "\(base).weight must stay packed uint32")
      #expect(ql.groupSize == 64)
      #expect(ql.bits == 4)
    }
  }

  @Test(
    "DiTBlock forward: QuantizedLinear (int4-resident) matches dequantized-fp16 within tight tolerance, no NaN"
  )
  func forwardParityQuantizedVsDequantized() throws {
    MLXRandom.seed(20_260_701)

    // Reference block loaded with dequantized fp16 weights (the legacy behavior).
    let blockRef = DiTBlock(
      hiddenSize: Self.hidden, numHeads: Self.heads, headDim: Self.headDim,
      mlpRatio: Self.mlpRatio)
    let (int4, fp16, quantizedBases) = Self.makeInt4AndFp16Dicts(
      from: blockRef.parameters().flattened())
    blockRef.update(parameters: ModuleParameters.unflattened(fp16))

    // Quantized block: identical effective weights, kept int4-resident.
    let blockQ = DiTBlock(
      hiddenSize: Self.hidden, numHeads: Self.heads, headDim: Self.headDim,
      mlpRatio: Self.mlpRatio)
    quantize(model: blockQ) { path, _ in
      quantizedBases.contains(path) ? (groupSize: 64, bits: 4, mode: .affine) : nil
    }
    blockQ.update(parameters: ModuleParameters.unflattened(int4))

    // Fixed latent + caption embedding + block conditioning.
    let B = 1
    let T = 64
    let C = Self.hidden
    let tText = 8
    let x = MLXRandom.normal([B, T, C])
    let y = MLXRandom.normal([B, tText, C])
    let t = MLXRandom.normal([B, 6 * C])
    let mask = MLXArray.ones([B, tText])

    let outRef = blockRef(x, y: y, t: t, mask: mask)
    let outQ = blockQ(x, y: y, t: t, mask: mask)
    eval(outRef, outQ)

    #expect(
      !isNaN(outQ.asType(.float32)).any().item(Bool.self), "quantized forward produced NaN")

    let diff = abs(outQ.asType(.float32) - outRef.asType(.float32))
    let maxAbs = diff.max().item(Float.self)
    // Both paths consume the identical packed int4 values; the only difference is
    // fused quantizedMM vs materialized-weight matmul rounding.
    #expect(
      maxAbs < 1e-2,
      "QuantizedLinear vs dequantized forward diverged by \(maxAbs) — expected them to match")
  }

  @Test("resident int4 weight bytes are ~1/4 of the fp16 equivalent")
  func residentBytesAreInt4Scale() throws {
    let block = DiTBlock(
      hiddenSize: Self.hidden, numHeads: Self.heads, headDim: Self.headDim,
      mlpRatio: Self.mlpRatio)
    let params = block.parameters().flattened()
    let (int4, _, quantizedBases) = Self.makeInt4AndFp16Dicts(from: params)

    Self.applyInt4(block: block, bases: quantizedBases, int4: int4)

    var quantizedBytes = 0
    var fp16EquivalentBytes = 0
    let leaves = Dictionary(uniqueKeysWithValues: block.leafModules().flattened())
    for base in quantizedBases {
      guard let ql = leaves[base] as? QuantizedLinear else { continue }
      quantizedBytes += ql.weight.nbytes + ql.scales.nbytes + (ql.biases?.nbytes ?? 0)
      // fp16 equivalent: [out, in] at 2 bytes each. weight packed uint32 is [out, in/8],
      // so in = weight.dim(1) * 8, out = weight.dim(0).
      let out = ql.weight.dim(0)
      let inDim = ql.weight.dim(1) * 8
      fp16EquivalentBytes += out * inDim * 2
    }

    #expect(quantizedBytes > 0)
    let ratio = Double(quantizedBytes) / Double(fp16EquivalentBytes)
    #expect(
      ratio < 0.35,
      "resident quantized bytes \(quantizedBytes) vs fp16 \(fp16EquivalentBytes) (ratio \(ratio)) — int4 should be ~1/4"
    )
  }

  private static func applyInt4(block: DiTBlock, bases: Set<String>, int4: [String: MLXArray]) {
    MLXNN.quantize(model: block) { path, _ in
      bases.contains(path) ? (groupSize: 64, bits: 4, mode: .affine) : nil
    }
    block.update(parameters: ModuleParameters.unflattened(int4))
  }
}
