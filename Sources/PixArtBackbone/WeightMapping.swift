@preconcurrency import MLX
import Tuberia

// MARK: - PixArt-Sigma Weight Key Mapping
//
// The model.safetensors for the int4-quantized PixArt-Sigma XL DiT uses MLX-native
// property paths directly as keys — no HF diffusers renaming is needed.
//
// Example keys in the safetensors:
//   blocks.0.attn.to_q.weight   (U32, packed int4)
//   blocks.0.attn.to_q.scales   (F16)
//   blocks.0.attn.to_q.biases   (F16)
//   blocks.0.attn.to_q.bias     (F16)
//   captionProjection.linear1.weight
//   patchEmbed.weight  (F16, only float layer)
//   t_block_linear.weight
//   finalLayer.linear.weight
//   ...
//
// The key mapping is therefore an identity passthrough: every key in the file maps
// directly to the corresponding MLX module property path.
//
// The only exception is patchEmbed.weight which is a Conv2d tensor stored as
// [O, kH, kW, I] in MLX NHWC format — no transposition is needed since the
// safetensors was already converted to MLX layout.

// MARK: - PixArtDiT Extension: WeightedSegment

extension PixArtDiT {

  /// Identity key mapping: safetensors keys ARE the MLX module property paths.
  ///
  /// All keys pass through unchanged. No keys are discarded (the safetensors
  /// was produced from the MLX module and only contains keys that exist in the model).
  ///
  /// Called by WeightLoader for every key in the safetensors file.
  public var keyMapping: KeyMapping {
    { key in key }
  }

  /// Per-tensor transform applied after key remapping.
  ///
  /// The int4-quantized safetensors stores patchEmbed.weight already in MLX NHWC layout
  /// [O, kH, kW, I] — no transposition is needed. All tensors pass through unchanged.
  public var tensorTransform: TensorTransform? {
    nil
  }
}
