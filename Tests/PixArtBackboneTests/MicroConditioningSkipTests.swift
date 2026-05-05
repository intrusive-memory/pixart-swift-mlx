import MLX
import Testing
import Tuberia

@testable import PixArtBackbone

// MARK: - Micro-Conditioning Skip Tests
//
// PixArtDiT.forward(_:) deliberately omits micro-conditioning (resolution +
// aspect ratio) from the timestep embedding `t` that drives all 28 DiT blocks.
//
// Source pin: Sources/PixArtBackbone/PixArtDiT.swift, forward(_:) lines 144-162
//   - Line 146:  let tEmb = timestepSinusoidalEmbedding(timestep)
//   - Line 149:  let t = timestepEmbedder(tEmb)
//   - Lines 151-154: comment "Skip adding micro-conditions: t remains as
//                     timestepEmbedder(tEmb)."
//   - Line 159:  let tBlock = tBlockLinear(t * MLX.sigmoid(t))
//   - Line 162:  let tRaw = t  // saved for FinalLayer (NOT tBlock)
//
// Why this is load-bearing:
//   The int4-quantized safetensors checkpoint (pixart-sigma-xl-dit-int4) was
//   produced by a converter that left out the micro-conditioning tower.
//   `adaln_single.linear.weight` ships at shape [6912, 1152], not the
//   [6912, 1152*N] needed for size+AR conditioning. PixArtDiT still
//   instantiates `sizeEmbedder` and `arEmbedder` (init lines 81-83) so the
//   module shape contract stays compatible, but they are NEVER called inside
//   `forward(_:)`. Folding them back in (the "obvious fix") silently breaks
//   every existing int4 weight checkpoint — outputs become garbage with no
//   shape error to flag the regression.
//
// These tests pin the structural property by exercising forward() under
// random init (no real weights) and asserting the determinism / shape /
// timestep-plumbing invariants that follow from skipping micro-cond.
@Suite("MicroConditioningSkip", .serialized)
struct MicroConditioningSkipTests {

  // MARK: - Determinism

  /// Forward determinism: same `BackboneInput` must produce bit-exact identical
  /// outputs across two consecutive calls.
  ///
  /// If `forward()` were ever modified to incorporate `sizeEmbedder` or
  /// `arEmbedder` based on hidden state (e.g. running statistics, a stochastic
  /// sampling of size buckets, or any non-deterministic micro-cond pathway),
  /// this test would fail. Random init is deterministic, the input is fixed,
  /// and no micro-cond hidden state is allowed to contaminate the result.
  @Test("Forward is deterministic for identical BackboneInput")
  func forwardDeterminism() throws {
    let dit = BackboneFixture.dit

    // Small latent: [1, 4, 4, 4] keeps the test fast (gridH=2, gridW=2 → 4 tokens).
    let input = BackboneInput(
      latents: MLXArray.zeros([1, 4, 4, 4]),
      conditioning: MLXArray.zeros([1, 120, 4096]),
      conditioningMask: MLXArray.ones([1, 120]),
      timestep: MLXArray([500 as Float])
    )

    let out1 = try dit.forward(input)
    eval(out1)
    let out2 = try dit.forward(input)
    eval(out2)

    #expect(out1.shape == out2.shape, "Two calls with identical input must produce same shape")

    // Bit-exact equality across the full output tensor.
    let allEqual = MLX.all(out1 .== out2).item(Bool.self)
    #expect(allEqual, "Forward must be bit-exact deterministic for identical inputs")
  }

  // MARK: - Micro-Conditioning Embedders Are Instantiated But Unused

  /// `PixArtDiT.init` instantiates `sizeEmbedder` and `arEmbedder` (init lines
  /// 81-83) so the module's parameter tree matches the historical state-dict
  /// shape. They exist as properties on the module but are never called inside
  /// `forward(_:)`.
  ///
  /// We can't easily mutate their weights from outside without `apply(weights:)`
  /// (and the int4 checkpoint has no weights for them anyway), so the
  /// pragmatic structural test is: the embedders are reachable as properties
  /// (so the module tree still matches the converter's expected shape), and
  /// `forward()` produces a well-shaped output without ever consulting them.
  ///
  /// Future engineer warning: do NOT add `sizeEmbedder(...)` or `arEmbedder(...)`
  /// calls into forward(_:). See file-header comment.
  @Test("sizeEmbedder and arEmbedder are instantiated on the module but unused by forward")
  func microCondEmbeddersExistButUnused() throws {
    let dit = BackboneFixture.dit

    // Property access compiles → embedders exist on the module. If a future
    // refactor removes them entirely the int4 checkpoint key tree would no
    // longer round-trip through `apply(weights:)`.
    _ = dit.sizeEmbedder
    _ = dit.arEmbedder

    // forward() runs to completion with a well-shaped output, without any
    // micro-cond signal supplied (BackboneInput has no size/AR fields).
    let input = BackboneInput(
      latents: MLXArray.zeros([1, 4, 4, 4]),
      conditioning: MLXArray.zeros([1, 120, 4096]),
      conditioningMask: MLXArray.ones([1, 120]),
      timestep: MLXArray([500 as Float])
    )
    let output = try dit.forward(input)
    eval(output)
    #expect(output.shape == [1, 4, 4, 4])
  }

  // MARK: - BackboneInput Has No Micro-Cond Fields

  /// The `BackboneInput` struct (defined in Tuberia) has exactly four fields:
  /// `latents`, `conditioning`, `conditioningMask`, `timestep`. No `width`,
  /// `height`, `aspectRatio`, or `resolution` field flows into the model.
  ///
  /// We pin this indirectly: build inputs at two distinct latent spatial sizes
  /// (4×4 and 6×6) at the same timestep, and assert each produces a
  /// correctly-shaped output. The position embedding is recomputed dynamically
  /// from gridH/gridW (PixArtDiT.swift forward lines 132-138), so the only
  /// spatial-dependent variable is the on-the-fly position embedding — NOT a
  /// learned size embedding driven by a micro-cond field.
  @Test("Forward accepts varying latent spatial sizes without any size/AR input")
  func forwardAcceptsVaryingSpatialSizesWithoutMicroCond() throws {
    let dit = BackboneFixture.dit

    // 4×4 latent → gridH=2, gridW=2
    let small = BackboneInput(
      latents: MLXArray.zeros([1, 4, 4, 4]),
      conditioning: MLXArray.zeros([1, 120, 4096]),
      conditioningMask: MLXArray.ones([1, 120]),
      timestep: MLXArray([500 as Float])
    )
    let smallOut = try dit.forward(small)
    eval(smallOut)
    #expect(smallOut.shape == [1, 4, 4, 4], "4×4 latent must produce 4×4 output")

    // 6×6 latent → gridH=3, gridW=3 (different from small; size embedder would
    // need to know about this if micro-cond were active — it doesn't).
    let large = BackboneInput(
      latents: MLXArray.zeros([1, 6, 6, 4]),
      conditioning: MLXArray.zeros([1, 120, 4096]),
      conditioningMask: MLXArray.ones([1, 120]),
      timestep: MLXArray([500 as Float])
    )
    let largeOut = try dit.forward(large)
    eval(largeOut)
    #expect(largeOut.shape == [1, 6, 6, 4], "6×6 latent must produce 6×6 output")
  }

  // MARK: - Timestep Is Plumbed Through (tRaw vs tBlock Distinct Paths)

  /// `PixArtDiT.swift` forward line 162 saves `tRaw = t` BEFORE `tBlock` is
  /// computed (line 159), so the FinalLayer sees the raw timestep embedding
  /// while the 28 DiT blocks see `tBlock` (the SiLU+Linear projection).
  /// Swapping these — passing `tBlock` to FinalLayer or `tRaw` to blocks —
  /// would silently corrupt output without any shape error.
  ///
  /// We can't directly inspect intermediate tensors from outside, but we can
  /// prove timestep IS reaching the model: zero-timestep and non-zero-timestep
  /// inputs (otherwise identical) MUST produce different outputs. If a future
  /// refactor accidentally short-circuited the timestep path (e.g. using a
  /// constant in place of `t`), this test would fail.
  @Test("Different timesteps produce different outputs (timestep is plumbed through)")
  func differentTimestepsProduceDifferentOutputs() throws {
    let dit = BackboneFixture.dit

    let latents = MLXArray.zeros([1, 4, 4, 4])
    let conditioning = MLXArray.zeros([1, 120, 4096])
    let mask = MLXArray.ones([1, 120])

    let zeroT = BackboneInput(
      latents: latents,
      conditioning: conditioning,
      conditioningMask: mask,
      timestep: MLXArray([0 as Float])
    )
    let bigT = BackboneInput(
      latents: latents,
      conditioning: conditioning,
      conditioningMask: mask,
      timestep: MLXArray([900 as Float])
    )

    let zeroOut = try dit.forward(zeroT)
    eval(zeroOut)
    let bigOut = try dit.forward(bigT)
    eval(bigOut)

    #expect(zeroOut.shape == bigOut.shape, "Outputs at differing timesteps must share shape")

    // At least one element must differ. If timestep were ignored (broken
    // plumbing), every element would match.
    let anyDifferent = MLX.any(zeroOut .!= bigOut).item(Bool.self)
    #expect(anyDifferent, "Distinct timesteps must influence the output (tRaw + tBlock plumbing)")
  }
}
