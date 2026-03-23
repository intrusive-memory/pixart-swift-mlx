@preconcurrency import MLX
import Tuberia

// MARK: - PixArt-Sigma Weight Key Mapping
//
// Maps HuggingFace diffusers safetensors keys to MLX module paths for the PixArtDiT backbone.
//
// Source format: PixArt-alpha/PixArt-Sigma-XL-2-1024-MS (diffusers format)
// Target format: MLX module property paths reflecting PixArtDiT's Swift class structure
//
// Total mappings: ~14 global + 28 blocks × ~8 per-block groups = ~238 key pairs
// Two keys are explicitly discarded (return nil):
//   - "pos_embed": 2D sinusoidal position embeddings are recomputed dynamically each
//     forward pass; the stored positional embedding is not used.
//   - "y_embedder.y_embedding": unconditional text embedding; not used at inference.
//
// LoRA-eligible keys (8 projections × 28 blocks = 224 keys):
// For each block i in 0..<28, the following attention projection keys are LoRA targets:
//
//   Self-attention (attn1 / attn in MLX):
//     transformer_blocks.{i}.attn1.to_q.weight  -> blocks.{i}.attn.to_q.weight
//     transformer_blocks.{i}.attn1.to_q.bias    -> blocks.{i}.attn.to_q.bias
//     transformer_blocks.{i}.attn1.to_k.weight  -> blocks.{i}.attn.to_k.weight
//     transformer_blocks.{i}.attn1.to_k.bias    -> blocks.{i}.attn.to_k.bias
//     transformer_blocks.{i}.attn1.to_v.weight  -> blocks.{i}.attn.to_v.weight
//     transformer_blocks.{i}.attn1.to_v.bias    -> blocks.{i}.attn.to_v.bias
//     transformer_blocks.{i}.attn1.to_out.0.weight -> blocks.{i}.attn.to_out.weight
//     transformer_blocks.{i}.attn1.to_out.0.bias   -> blocks.{i}.attn.to_out.bias
//
//   Cross-attention (attn2 / cross_attn in MLX):
//     transformer_blocks.{i}.attn2.to_q.weight  -> blocks.{i}.cross_attn.to_q.weight
//     transformer_blocks.{i}.attn2.to_q.bias    -> blocks.{i}.cross_attn.to_q.bias
//     transformer_blocks.{i}.attn2.to_k.weight  -> blocks.{i}.cross_attn.to_k.weight
//     transformer_blocks.{i}.attn2.to_k.bias    -> blocks.{i}.cross_attn.to_k.bias
//     transformer_blocks.{i}.attn2.to_v.weight  -> blocks.{i}.cross_attn.to_v.weight
//     transformer_blocks.{i}.attn2.to_v.bias    -> blocks.{i}.cross_attn.to_v.bias
//     transformer_blocks.{i}.attn2.to_out.0.weight -> blocks.{i}.cross_attn.to_out.weight
//     transformer_blocks.{i}.attn2.to_out.0.bias   -> blocks.{i}.cross_attn.to_out.bias
//
// LoRA key count: 8 weight keys + 8 bias keys = 16 keys per block × 28 blocks = 448 keys total.
// The 224 "eligible" LoRA keys referenced in REQUIREMENTS.md P6 refer to the weight tensors
// only (excluding bias tensors), matching LoRA convention: 8 projections × 28 blocks = 224.

// MARK: - Build the static key table

/// All HuggingFace diffusers keys that map to MLX module paths, built once.
private let pixArtKeyTable: [String: String] = {
    var table: [String: String] = [:]

    // -------------------------------------------------------------------------
    // Global keys (~22 pairs)
    // -------------------------------------------------------------------------

    // Patch embedding Conv2d: pos_embed.proj -> patchEmbed
    // Weight requires Conv2d transposition: [O,I,kH,kW] -> [O,kH,kW,I]
    table["pos_embed.proj.weight"] = "patchEmbed.weight"
    table["pos_embed.proj.bias"]   = "patchEmbed.bias"

    // Caption projection: Linear(4096,1152) -> GELU(tanh) -> Linear(1152,1152)
    table["caption_projection.linear_1.weight"] = "captionProjection.linear1.weight"
    table["caption_projection.linear_1.bias"]   = "captionProjection.linear1.bias"
    table["caption_projection.linear_2.weight"] = "captionProjection.linear2.weight"
    table["caption_projection.linear_2.bias"]   = "captionProjection.linear2.bias"

    // Timestep embedder MLP: Linear(256,1152) -> SiLU -> Linear(1152,1152)
    table["adaln_single.emb.timestep_embedder.linear_1.weight"] = "timestepEmbedder.linear1.weight"
    table["adaln_single.emb.timestep_embedder.linear_1.bias"]   = "timestepEmbedder.linear1.bias"
    table["adaln_single.emb.timestep_embedder.linear_2.weight"] = "timestepEmbedder.linear2.weight"
    table["adaln_single.emb.timestep_embedder.linear_2.bias"]   = "timestepEmbedder.linear2.bias"

    // Resolution embedder: single MLP shared for height and width embedding.
    // SizeEmbedder.embedder applies the same MLP to H and W independently.
    table["adaln_single.emb.resolution_embedder.linear_1.weight"] = "sizeEmbedder.embedder.linear1.weight"
    table["adaln_single.emb.resolution_embedder.linear_1.bias"]   = "sizeEmbedder.embedder.linear1.bias"
    table["adaln_single.emb.resolution_embedder.linear_2.weight"] = "sizeEmbedder.embedder.linear2.weight"
    table["adaln_single.emb.resolution_embedder.linear_2.bias"]   = "sizeEmbedder.embedder.linear2.bias"

    // Aspect ratio embedder
    table["adaln_single.emb.aspect_ratio_embedder.linear_1.weight"] = "arEmbedder.embedder.linear1.weight"
    table["adaln_single.emb.aspect_ratio_embedder.linear_1.bias"]   = "arEmbedder.embedder.linear1.bias"
    table["adaln_single.emb.aspect_ratio_embedder.linear_2.weight"] = "arEmbedder.embedder.linear2.weight"
    table["adaln_single.emb.aspect_ratio_embedder.linear_2.bias"]   = "arEmbedder.embedder.linear2.bias"

    // t_block: SiLU -> Linear(1152, 6*1152)
    // @ModuleInfo(key: "t_block_linear") in PixArtDiT
    table["adaln_single.linear.weight"] = "t_block_linear.weight"
    table["adaln_single.linear.bias"]   = "t_block_linear.bias"

    // Final layer projection: Linear(1152, patchSize^2 * outChannels)
    // @ModuleInfo(key: "linear") in FinalLayer
    table["proj_out.weight"] = "finalLayer.linear.weight"
    table["proj_out.bias"]   = "finalLayer.linear.bias"

    // Final layer AdaLN scale_shift_table: [2, 1152]
    // Plain MLXArray property in FinalLayer: reflected as "scaleShiftTable"
    table["scale_shift_table"] = "finalLayer.scaleShiftTable"

    // -------------------------------------------------------------------------
    // Per-block keys: 28 blocks × ~8 groups
    // -------------------------------------------------------------------------
    for i in 0..<28 {
        let hf = "transformer_blocks.\(i)"
        let mlx = "blocks.\(i)"

        // scale_shift_table: [6, 1152] — AdaLN-Single per-block learned parameters.
        // Plain MLXArray in DiTBlock, reflected as "scaleShiftTable".
        table["\(hf).scale_shift_table"] = "\(mlx).scaleShiftTable"

        // Self-attention Q/K/V/out projections
        // SelfAttention uses @ModuleInfo(key:) overrides: to_q, to_k, to_v, to_out
        table["\(hf).attn1.to_q.weight"] = "\(mlx).attn.to_q.weight"
        table["\(hf).attn1.to_q.bias"]   = "\(mlx).attn.to_q.bias"
        table["\(hf).attn1.to_k.weight"] = "\(mlx).attn.to_k.weight"
        table["\(hf).attn1.to_k.bias"]   = "\(mlx).attn.to_k.bias"
        table["\(hf).attn1.to_v.weight"] = "\(mlx).attn.to_v.weight"
        table["\(hf).attn1.to_v.bias"]   = "\(mlx).attn.to_v.bias"
        // diffusers stores output projection as "to_out.0.*" (Sequential wrapper)
        table["\(hf).attn1.to_out.0.weight"] = "\(mlx).attn.to_out.weight"
        table["\(hf).attn1.to_out.0.bias"]   = "\(mlx).attn.to_out.bias"

        // Self-attention QK norms: LayerNorm on q and k after reshape
        // @ModuleInfo(key: "q_norm") and @ModuleInfo(key: "k_norm") in SelfAttention
        table["\(hf).attn1.q_norm.weight"] = "\(mlx).attn.q_norm.weight"
        table["\(hf).attn1.q_norm.bias"]   = "\(mlx).attn.q_norm.bias"
        table["\(hf).attn1.k_norm.weight"] = "\(mlx).attn.k_norm.weight"
        table["\(hf).attn1.k_norm.bias"]   = "\(mlx).attn.k_norm.bias"

        // Cross-attention Q/K/V/out projections
        // DiTBlock uses @ModuleInfo(key: "cross_attn") for crossAttn property.
        // CrossAttention uses @ModuleInfo(key:) overrides: to_q, to_k, to_v, to_out
        // Cross-attention receives NO AdaLN modulation (PixArt design).
        table["\(hf).attn2.to_q.weight"] = "\(mlx).cross_attn.to_q.weight"
        table["\(hf).attn2.to_q.bias"]   = "\(mlx).cross_attn.to_q.bias"
        table["\(hf).attn2.to_k.weight"] = "\(mlx).cross_attn.to_k.weight"
        table["\(hf).attn2.to_k.bias"]   = "\(mlx).cross_attn.to_k.bias"
        table["\(hf).attn2.to_v.weight"] = "\(mlx).cross_attn.to_v.weight"
        table["\(hf).attn2.to_v.bias"]   = "\(mlx).cross_attn.to_v.bias"
        table["\(hf).attn2.to_out.0.weight"] = "\(mlx).cross_attn.to_out.weight"
        table["\(hf).attn2.to_out.0.bias"]   = "\(mlx).cross_attn.to_out.bias"

        // FFN: GEGLU (fc1 projects to 2*ffnHiddenSize, split for gate*value)
        // @ModuleInfo var fc1 and fc2 in GEGLUFFN, module key is "mlp" in DiTBlock
        table["\(hf).ff.net.0.proj.weight"] = "\(mlx).mlp.fc1.weight"
        table["\(hf).ff.net.0.proj.bias"]   = "\(mlx).mlp.fc1.bias"
        table["\(hf).ff.net.2.weight"] = "\(mlx).mlp.fc2.weight"
        table["\(hf).ff.net.2.bias"]   = "\(mlx).mlp.fc2.bias"
    }

    return table
}()

// MARK: - Discarded Keys

/// Keys that should be silently dropped during weight loading.
/// - "pos_embed": 2D sinusoidal embeddings recomputed dynamically per forward pass.
/// - "y_embedder.y_embedding": unconditional embedding not used at inference.
private let pixArtDiscardedKeys: Set<String> = [
    "pos_embed",
    "y_embedder.y_embedding",
]

// MARK: - PixArtDiT Extension: WeightedSegment

extension PixArtDiT {

    /// Key mapping from HuggingFace diffusers safetensors keys to MLX module paths.
    ///
    /// Returns nil for discarded keys (pos_embed, y_embedder.y_embedding) and for any
    /// key not present in the mapping table (unknown keys are silently skipped).
    ///
    /// Called by WeightLoader for every key in the safetensors file.
    public var keyMapping: KeyMapping {
        { key in
            // Explicitly discarded keys
            if pixArtDiscardedKeys.contains(key) {
                return nil
            }
            // Table lookup — unknown keys return nil (silently skipped)
            return pixArtKeyTable[key]
        }
    }

    /// Per-tensor transform applied after key remapping.
    ///
    /// Transposes Conv2d weights from PyTorch layout [O, I, kH, kW] to MLX NHWC layout
    /// [O, kH, kW, I] via transpose(0, 2, 3, 1). Only applied to the patch embedding
    /// conv weight ("patchEmbed.weight" in MLX), which is a 4D tensor.
    ///
    /// All other tensors pass through unchanged.
    public var tensorTransform: TensorTransform? {
        { mlxKey, tensor in
            // Apply only to the patch embedding Conv2d weight: [O, I, kH, kW] -> [O, kH, kW, I]
            if mlxKey == "patchEmbed.weight" && tensor.ndim == 4 {
                return tensor.transposed(0, 2, 3, 1)
            }
            return tensor
        }
    }
}
