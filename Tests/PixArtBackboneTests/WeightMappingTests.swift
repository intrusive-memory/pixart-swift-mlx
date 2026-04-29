import Testing

@testable import PixArtBackbone

@Suite("Weight Key Mapping")
struct WeightMappingTests {

  // MARK: - Identity Passthrough Mapping
  //
  // After the SwiftTuberia 0.6 / SwiftAcervo 0.8 dep bump, the int4-quantized
  // model.safetensors stores keys that ARE the MLX module property paths directly
  // (e.g. "blocks.0.attn.to_q.weight", "patchEmbed.weight"). No PyTorch/HF-diffusers
  // → MLX renaming is required, so `PixArtDiT.keyMapping` is an identity passthrough
  // and `PixArtDiT.tensorTransform` is nil.
  //
  // These tests pin that contract.

  @Test("keyMapping is identity for global keys")
  func globalKeysPassthrough() throws {
    let dit = try PixArtDiT(configuration: PixArtDiTConfiguration())
    let mapping = dit.keyMapping

    // Patch embedding (Conv2d already in MLX NHWC layout — no transpose needed)
    #expect(mapping("patchEmbed.weight") == "patchEmbed.weight")
    #expect(mapping("patchEmbed.bias") == "patchEmbed.bias")

    // Caption projection
    #expect(mapping("captionProjection.linear1.weight") == "captionProjection.linear1.weight")
    #expect(mapping("captionProjection.linear2.bias") == "captionProjection.linear2.bias")

    // Timestep embedder
    #expect(mapping("timestepEmbedder.linear1.weight") == "timestepEmbedder.linear1.weight")

    // Resolution / aspect-ratio embedders (absent from int4 safetensors but the
    // mapping itself is still identity for any key that might appear)
    #expect(
      mapping("sizeEmbedder.embedder.linear1.weight") == "sizeEmbedder.embedder.linear1.weight")
    #expect(mapping("arEmbedder.embedder.linear2.bias") == "arEmbedder.embedder.linear2.bias")

    // t_block
    #expect(mapping("t_block_linear.weight") == "t_block_linear.weight")

    // Final layer
    #expect(mapping("finalLayer.linear.weight") == "finalLayer.linear.weight")
    #expect(mapping("finalLayer.scaleShiftTable") == "finalLayer.scaleShiftTable")
  }

  @Test("keyMapping is identity for per-block keys (block 0 and block 27)")
  func perBlockKeysPassthrough() throws {
    let dit = try PixArtDiT(configuration: PixArtDiTConfiguration())
    let mapping = dit.keyMapping

    // Block 0: self-attention
    #expect(mapping("blocks.0.attn.to_q.weight") == "blocks.0.attn.to_q.weight")
    #expect(mapping("blocks.0.attn.to_out.bias") == "blocks.0.attn.to_out.bias")

    // Block 0: cross-attention
    #expect(mapping("blocks.0.cross_attn.to_k.weight") == "blocks.0.cross_attn.to_k.weight")

    // Block 0: FFN (MLP fc1/fc2)
    #expect(mapping("blocks.0.mlp.fc1.weight") == "blocks.0.mlp.fc1.weight")
    #expect(mapping("blocks.0.mlp.fc2.weight") == "blocks.0.mlp.fc2.weight")

    // Block 0: scale_shift_table
    #expect(mapping("blocks.0.scaleShiftTable") == "blocks.0.scaleShiftTable")

    // Block 27 (last) — spot-check
    #expect(mapping("blocks.27.attn.to_q.weight") == "blocks.27.attn.to_q.weight")
    #expect(mapping("blocks.27.cross_attn.to_v.bias") == "blocks.27.cross_attn.to_v.bias")
    #expect(mapping("blocks.27.mlp.fc2.bias") == "blocks.27.mlp.fc2.bias")
  }

  @Test("keyMapping is identity for QK norm keys")
  func qkNormKeysPassthrough() throws {
    let dit = try PixArtDiT(configuration: PixArtDiTConfiguration())
    let mapping = dit.keyMapping
    #expect(mapping("blocks.0.attn.q_norm.weight") == "blocks.0.attn.q_norm.weight")
    #expect(mapping("blocks.0.attn.k_norm.bias") == "blocks.0.attn.k_norm.bias")
  }

  @Test("keyMapping does not discard keys (no HF-diffusers renaming)")
  func noKeysDiscarded() throws {
    let dit = try PixArtDiT(configuration: PixArtDiTConfiguration())
    let mapping = dit.keyMapping
    // The MLX-native safetensors only contains keys that exist in the model, so the
    // mapping never returns nil — even arbitrary strings pass through unchanged.
    // Filtering is the responsibility of the safetensors author, not the mapping.
    #expect(mapping("totally_bogus_key") == "totally_bogus_key")
    #expect(mapping("blocks.99.attn.to_q.weight") == "blocks.99.attn.to_q.weight")
  }

  @Test("tensorTransform is nil (Conv2d patch embed already in MLX NHWC layout)")
  func tensorTransformIsNil() throws {
    let dit = try PixArtDiT(configuration: PixArtDiTConfiguration())
    // The int4-quantized safetensors stores patchEmbed.weight already in MLX NHWC
    // layout [O, kH, kW, I] — no PyTorch [O, I, kH, kW] → MLX transposition is
    // required, so the per-tensor transform hook is unused.
    #expect(dit.tensorTransform == nil)
  }
}
