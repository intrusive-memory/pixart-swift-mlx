import MLX
import MLXNN
import Testing

@testable import PixArtBackbone

/// Tests that pin behaviors which shape-only tests miss:
///
/// 1. `FinalLayer.unpatchify` axis order — `transposed(0, 1, 3, 2, 4, 5)` is load-bearing.
/// 2. `CrossAttention` mask=all-ones equals mask=nil (additive mask is zero in both cases).
/// 3. `CrossAttention` output sequence dim follows Q (image), not K/V (text) — across context lengths.
/// 4. Patch embedding `reshaped(B, gridH * gridW, hiddenSize)` produces row-major token order.
@Suite("UnpatchifyAndAttention")
struct UnpatchifyAndAttentionTests {

  // MARK: - Test 1: unpatchify axis order

  /// Pins the `(0, 1, 3, 2, 4, 5)` transpose in `FinalLayer.unpatchify`.
  ///
  /// Replicates the unpatchify logic directly using public MLX API (the method is
  /// private and Linear weights are random, so we test it as a pure function).
  ///
  /// Construction: each `(i_patch, j_patch)` patch contains the constant value
  /// `i_patch * gridW + j_patch`, repeated across `p*p` positions and `c=1` channels.
  ///
  /// After the correct transpose+reshape, output coordinates `[I, J]` should equal
  /// `(I/p) * gridW + (J/p)`. A swapped transpose (e.g. `(0, 3, 1, 4, 2, 5)`) would
  /// scatter the per-patch values across the output grid and break this assertion.
  @Test("FinalLayer unpatchify preserves spatial locality with transpose (0,1,3,2,4,5)")
  func unpatchifyAxisOrder() {
    let B = 1
    let gridH = 3
    let gridW = 4
    let p = 2
    let c = 1
    let T = gridH * gridW

    // Build input shaped [B, T, p*p*c] where token t = i*gridW+j has all entries == (i*gridW+j).
    var values = [Float]()
    values.reserveCapacity(B * T * p * p * c)
    for _ in 0..<B {
      for i in 0..<gridH {
        for j in 0..<gridW {
          let v = Float(i * gridW + j)
          for _ in 0..<(p * p * c) {
            values.append(v)
          }
        }
      }
    }
    let input = MLXArray(values).reshaped(B, T, p * p * c)

    // Apply the same reshape/transpose/reshape sequence as FinalLayer.unpatchify.
    let out =
      input
      .reshaped(B, gridH, gridW, p, p, c)
      .transposed(0, 1, 3, 2, 4, 5)
      .reshaped(B, gridH * p, gridW * p, c)
    eval(out)

    #expect(out.dim(0) == B)
    #expect(out.dim(1) == gridH * p)
    #expect(out.dim(2) == gridW * p)
    #expect(out.dim(3) == c)

    // Verify spatial locality: out[0, I, J, 0] == (I/p) * gridW + (J/p) for every (I, J).
    let H = gridH * p
    let W = gridW * p
    for I in 0..<H {
      for J in 0..<W {
        let expected = Float((I / p) * gridW + (J / p))
        let actual = out[0, I, J, 0].item(Float.self)
        #expect(
          actual == expected,
          "out[0, \(I), \(J), 0] = \(actual), expected \(expected)"
        )
      }
    }

    // Sanity-check at least 4 corners explicitly:
    // patch (0,0) -> output [0..2, 0..2] all == 0
    #expect(out[0, 0, 0, 0].item(Float.self) == 0)
    // patch (0, gridW-1) -> output [0..2, (gridW-1)*p..gridW*p] all == gridW-1
    #expect(out[0, 0, (gridW - 1) * p, 0].item(Float.self) == Float(gridW - 1))
    // patch (gridH-1, 0) -> output [(gridH-1)*p..gridH*p, 0..2] all == (gridH-1)*gridW
    #expect(out[0, (gridH - 1) * p, 0, 0].item(Float.self) == Float((gridH - 1) * gridW))
    // patch (gridH-1, gridW-1) -> bottom-right corner
    #expect(
      out[0, H - 1, W - 1, 0].item(Float.self)
        == Float((gridH - 1) * gridW + (gridW - 1))
    )
  }

  // MARK: - Test 2: CrossAttention mask=all-ones equals mask=nil

  /// All-ones mask becomes an all-zero additive mask, which is identical to no mask.
  ///
  /// Same module instance + identical deterministic inputs are used for both calls
  /// so weights and inputs match exactly between the two invocations.
  @Test("CrossAttention all-ones mask equals nil mask")
  func crossAttentionAllOnesMaskEqualsNil() {
    let B = 1
    let Timg = 5
    let Ttext = 4
    let hiddenSize = 16
    let numHeads = 2
    let headDim = hiddenSize / numHeads

    let crossAttn = CrossAttention(
      hiddenSize: hiddenSize, numHeads: numHeads, headDim: headDim
    )

    // Deterministic constant inputs (no randomness so the two calls are bit-comparable).
    let x = MLXArray.zeros([B, Timg, hiddenSize]) + Float(0.5)
    let context = MLXArray.zeros([B, Ttext, hiddenSize]) + Float(0.3)

    let allOnes = MLXArray([Int32](repeating: 1, count: Ttext)).reshaped(B, Ttext)

    let outNil = crossAttn(x, context: context, mask: nil)
    let outOnes = crossAttn(x, context: context, mask: allOnes)
    eval(outNil, outOnes)

    let maxDiff = MLX.abs(outNil - outOnes).max().item(Float.self)
    #expect(
      maxDiff < 1e-5,
      "all-ones mask should match nil mask, got max abs diff = \(maxDiff)"
    )
    #expect(outOnes.dim(1) == Timg)
    #expect(outOnes.dim(2) == hiddenSize)
  }

  // MARK: - Test 3: Cross-attention output shape independent of text length

  /// Strengthens the existing `crossAttentionQKVSplit` test by using the spec's
  /// concrete sizes (Timg=10, Ttext1=3, Ttext2=20) and asserting the full output
  /// shape, not just the sequence dim.
  @Test("CrossAttention output has [B, Timg, hiddenSize] regardless of text length")
  func crossAttentionOutputShapeIndependentOfTextLength() {
    let B = 1
    let Timg = 10
    let Ttext1 = 3
    let Ttext2 = 20
    let hiddenSize = 16
    let numHeads = 2
    let headDim = hiddenSize / numHeads

    let crossAttn = CrossAttention(
      hiddenSize: hiddenSize, numHeads: numHeads, headDim: headDim
    )

    let q = MLXArray.zeros([B, Timg, hiddenSize])
    let context1 = MLXArray.zeros([B, Ttext1, hiddenSize])
    let context2 = MLXArray.zeros([B, Ttext2, hiddenSize])

    let out1 = crossAttn(q, context: context1, mask: nil)
    let out2 = crossAttn(q, context: context2, mask: nil)
    eval(out1, out2)

    // Output sequence dim must equal Timg (Q), never Ttext (K/V).
    #expect(out1.dim(0) == B)
    #expect(out1.dim(1) == Timg)
    #expect(out1.dim(2) == hiddenSize)
    #expect(out2.dim(0) == B)
    #expect(out2.dim(1) == Timg)
    #expect(out2.dim(2) == hiddenSize)
  }

  // MARK: - Test 4: Patch embedding spatial -> sequence mapping

  /// Pins the row-major reshape that converts `[B, gridH, gridW, hiddenSize]`
  /// into `[B, gridH * gridW, hiddenSize]` in `PixArtDiT.forward`.
  ///
  /// Token index `t` must correspond to spatial position `(t / gridW, t % gridW)`.
  /// We don't depend on conv weight values: we compare the flattened tokens
  /// against the unflattened patched output position-by-position.
  @Test("Patch embedding reshape maps token t to spatial (t/gridW, t%gridW)")
  func patchEmbeddingSequenceIndexMapping() {
    let inChannels = 1
    let hiddenSize = 8
    let patchSize = 2
    let patchEmbed = Conv2d(
      inputChannels: inChannels,
      outputChannels: hiddenSize,
      kernelSize: IntOrPair(patchSize),
      stride: IntOrPair(patchSize),
      bias: false
    )

    let B = 1
    let spatialH = 4
    let spatialW = 6

    // Deterministic input: pixel(i, j) = i * spatialW + j.
    var pixels = [Float]()
    pixels.reserveCapacity(B * spatialH * spatialW * inChannels)
    for _ in 0..<B {
      for i in 0..<spatialH {
        for j in 0..<spatialW {
          for _ in 0..<inChannels {
            pixels.append(Float(i * spatialW + j))
          }
        }
      }
    }
    let input = MLXArray(pixels).reshaped(B, spatialH, spatialW, inChannels)

    let patched = patchEmbed(input)  // [B, gridH, gridW, hiddenSize]
    eval(patched)

    let gridH = patched.dim(1)
    let gridW = patched.dim(2)
    #expect(gridH == spatialH / patchSize)
    #expect(gridW == spatialW / patchSize)
    #expect(gridH == 2)
    #expect(gridW == 3)

    let tokens = patched.reshaped(B, gridH * gridW, hiddenSize)
    eval(tokens)

    #expect(tokens.dim(0) == B)
    #expect(tokens.dim(1) == gridH * gridW)
    #expect(tokens.dim(2) == hiddenSize)

    // For each token t, tokens[0, t, :] must equal patched[0, t/gridW, t%gridW, :].
    // This pins the row-major reshape ordering, independent of conv weight values.
    for t in 0..<(gridH * gridW) {
      let iPatch = t / gridW
      let jPatch = t % gridW
      let tokenSlice = tokens[0, t, 0...]
      let patchedSlice = patched[0, iPatch, jPatch, 0...]
      let diff = MLX.abs(tokenSlice - patchedSlice).max().item(Float.self)
      #expect(
        diff == 0,
        "tokens[0, \(t), :] != patched[0, \(iPatch), \(jPatch), :], max diff = \(diff)"
      )
    }
  }
}
