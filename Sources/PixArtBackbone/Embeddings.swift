import Foundation
@preconcurrency import MLX
import MLXNN

// MARK: - 2D Sinusoidal Position Embeddings

/// Computes 2D sinusoidal position embeddings dynamically based on spatial dimensions.
///
/// This enables variable resolution natively — embeddings are recomputed per forward pass
/// using the actual spatial grid size rather than being stored as learned parameters.
///
/// Grid coordinates: `arange(gridSize) / (gridSize / baseGridSize) / peInterpolation`
/// Half the dimensions encode height, half encode width.
/// Frequencies: `1 / 10000^(2i/d)`, standard sinusoidal.
///
/// - Parameters:
///   - gridH: Spatial height of the token grid (H/patchSize).
///   - gridW: Spatial width of the token grid (W/patchSize).
///   - hiddenSize: Embedding dimension (split evenly between H and W).
///   - peInterpolation: Interpolation factor (2 for PixArt-Sigma XL).
///   - baseSize: Base resolution / patchSize for normalization.
/// - Returns: Position embeddings of shape [1, gridH * gridW, hiddenSize].
func get2DSinusoidalPositionEmbeddings(
  gridH: Int,
  gridW: Int,
  hiddenSize: Int,
  peInterpolation: Float,
  baseSize: Int
) -> MLXArray {
  let baseGridSize = Float(baseSize)
  let halfDim = hiddenSize / 2

  // Compute grid coordinates normalized by base resolution and interpolation
  let hCoords =
    MLXArray(0..<gridH).asType(.float32) / (Float(gridH) / baseGridSize) / peInterpolation
  let wCoords =
    MLXArray(0..<gridW).asType(.float32) / (Float(gridW) / baseGridSize) / peInterpolation

  // Compute sinusoidal embeddings for each axis (each gets halfDim / 2 sin + halfDim / 2 cos)
  let embedH = sinusoidalEmbedding1D(positions: hCoords, dim: halfDim)  // [gridH, halfDim]
  let embedW = sinusoidalEmbedding1D(positions: wCoords, dim: halfDim)  // [gridW, halfDim]

  // Create 2D grid: for each (h, w) pair, concatenate embedH[h] and embedW[w]
  // embedH: [gridH, 1, halfDim] broadcast with embedW: [1, gridW, halfDim]
  let embedHExpanded = embedH.expandedDimensions(axis: 1)  // [gridH, 1, halfDim]
  let embedWExpanded = embedW.expandedDimensions(axis: 0)  // [1, gridW, halfDim]

  // Broadcast and tile to create full grid
  let embedHTiled = MLX.broadcast(embedHExpanded, to: [gridH, gridW, halfDim])
  let embedWTiled = MLX.broadcast(embedWExpanded, to: [gridH, gridW, halfDim])

  // Concatenate along embedding dimension: [gridH, gridW, hiddenSize]
  let posEmbed = concatenated([embedHTiled, embedWTiled], axis: -1)

  // Flatten spatial dims and add batch dim: [1, gridH * gridW, hiddenSize]
  return posEmbed.reshaped(1, gridH * gridW, hiddenSize)
}

/// Computes 1D sinusoidal embeddings for a set of positions.
///
/// Matches the diffusers `get_1d_sincos_pos_embed_from_grid_np` formula:
///   omega = arange(D/2) / (D/2)
///   freqs = 1 / 10000^omega = exp(-log(10000) * i / (D/2))
///   angles = pos * freqs
///   embedding = [sin(angles), cos(angles)]   ← sin FIRST, matching diffusers
///
/// The output order is [sin, cos] to match how PixArt-Sigma was trained.
///
/// - Parameters:
///   - positions: 1D array of position values, shape [N].
///   - dim: Output embedding dimension.
/// - Returns: Embeddings of shape [N, dim].
func sinusoidalEmbedding1D(positions: MLXArray, dim: Int) -> MLXArray {
  let halfDim = dim / 2
  let logBase: Float = Foundation.log(10000.0)
  let indices = MLXArray(0..<halfDim).asType(.float32)
  let freqs = MLX.exp(-logBase * indices / Float(halfDim))  // [halfDim]

  // positions: [N], freqs: [halfDim] -> outer product: [N, halfDim]
  let angles = positions.expandedDimensions(axis: -1) * freqs.expandedDimensions(axis: 0)

  // Concatenate sin and cos: [N, dim] — sin FIRST to match diffusers convention
  return concatenated([MLX.sin(angles), MLX.cos(angles)], axis: -1)
}

// MARK: - Timestep Sinusoidal Embedding

/// Computes sinusoidal embedding for diffusion timesteps.
///
/// Matches the diffusers `get_timestep_embedding` formula exactly:
///   exponent = -log(10000) * arange(0, halfDim) / (halfDim - downscale_freq_shift)
///   freqs = exp(exponent)
///   angles = timestep * freqs
///   embedding = [sin(angles), cos(angles)]   ← sin FIRST, matching diffusers default
///
/// PixArt-Sigma uses `downscale_freq_shift=1` (the diffusers default), meaning the
/// denominator is `(halfDim - 1)` not `halfDim`. This ensures frequencies span
/// exactly [1, 1/10000] rather than [1, 10000^(-127/128)].
///
/// Output order is [sin, cos] to match the order the timestep MLP weights were trained on.
///
/// - Parameters:
///   - timestep: Timestep values, shape [B].
///   - dim: Embedding dimension (default 256).
/// - Returns: Timestep embedding, shape [B, dim].
func timestepSinusoidalEmbedding(_ timestep: MLXArray, dim: Int = 256) -> MLXArray {
  let halfDim = dim / 2
  let logBase: Float = Foundation.log(10000.0)
  let indices = MLXArray(0..<halfDim).asType(.float32)
  // Denominator is (halfDim - 1) matching diffusers downscale_freq_shift=1
  let freqs = MLX.exp(-logBase * indices / Float(halfDim - 1))  // [halfDim]

  // timestep: [B], freqs: [halfDim] -> [B, halfDim]
  // Cast timestep to float32 to avoid int32 * float16 precision issues
  let angles =
    timestep.asType(.float32).expandedDimensions(axis: -1) * freqs.expandedDimensions(axis: 0)

  // [B, dim]: sin FIRST, then cos — matching diffusers get_timestep_embedding output order
  return concatenated([MLX.sin(angles), MLX.cos(angles)], axis: -1)
}

// MARK: - Timestep MLP

/// Timestep embedder MLP: Linear(256, hiddenSize) -> SiLU -> Linear(hiddenSize, hiddenSize).
///
/// Projects the sinusoidal timestep embedding into the model's hidden dimension.
final class TimestepEmbedder: Module, @unchecked Sendable {
  @ModuleInfo var linear1: Linear
  @ModuleInfo var linear2: Linear

  init(hiddenSize: Int, frequencyDim: Int = 256) {
    self._linear1.wrappedValue = Linear(frequencyDim, hiddenSize)
    self._linear2.wrappedValue = Linear(hiddenSize, hiddenSize)
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    var h = linear1(x)
    // silu uses compile(shapeless:true) which can return 0-D tensors under memory pressure.
    // Replace with direct math: silu(x) = x * sigmoid(x)
    h = h * MLX.sigmoid(h)
    h = linear2(h)
    return h
  }
}

// MARK: - Micro-Condition Embedders

/// Micro-condition embedder for a single scalar value (height, width, or aspect ratio).
///
/// sinusoidal(256) -> MLP(256 -> outputDim -> outputDim)
///
/// Used for resolution and aspect ratio conditioning in PixArt-Sigma.
final class MicroConditionEmbedder: Module, @unchecked Sendable {
  @ModuleInfo var linear1: Linear
  @ModuleInfo var linear2: Linear
  let frequencyDim: Int

  init(frequencyDim: Int = 256, outputDim: Int) {
    self.frequencyDim = frequencyDim
    self._linear1.wrappedValue = Linear(frequencyDim, outputDim)
    self._linear2.wrappedValue = Linear(outputDim, outputDim)
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    let emb = timestepSinusoidalEmbedding(x, dim: frequencyDim)
    var h = linear1(emb)
    // silu uses compile(shapeless:true) which can return 0-D tensors under memory pressure.
    // Replace with direct math: silu(x) = x * sigmoid(x)
    h = h * MLX.sigmoid(h)
    h = linear2(h)
    return h
  }
}

// MARK: - Size Embedder (Resolution)

/// Resolution embedder: embeds height and width separately through the SAME MLP, then concatenates.
///
/// Matches the original PixArt-Sigma `csize_embedder` which uses a single shared MLP
/// applied independently to both H and W. In HuggingFace diffusers format, the shared
/// weights are stored as `adaln_single.emb.resolution_embedder.linear_{1,2}`.
///
/// Each dimension: sinusoidal(256) -> MLP(256 -> 384 -> 384)
/// Output: [B, 768] (2 x 384)
final class SizeEmbedder: Module, @unchecked Sendable {
  /// Single shared MLP applied to both height and width independently.
  let embedder: MicroConditionEmbedder

  init(frequencyDim: Int = 256, outputDimPerAxis: Int = 384) {
    self.embedder = MicroConditionEmbedder(frequencyDim: frequencyDim, outputDim: outputDimPerAxis)
  }

  /// - Parameter size: [B, 2] where columns are (height, width).
  /// - Returns: [B, outputDimPerAxis * 2]
  func callAsFunction(_ size: MLXArray) -> MLXArray {
    let h = size[0..., 0]  // [B]
    let w = size[0..., 1]  // [B]
    let hEmb = embedder(h)  // [B, 384]
    let wEmb = embedder(w)  // [B, 384] — same MLP applied to width
    return concatenated([hEmb, wEmb], axis: -1)  // [B, 768]
  }
}

// MARK: - Aspect Ratio Embedder

/// Aspect ratio embedder: sinusoidal(256) -> MLP(256 -> 384 -> 384).
///
/// Output: [B, 384]
final class AspectRatioEmbedder: Module, @unchecked Sendable {
  let embedder: MicroConditionEmbedder

  init(frequencyDim: Int = 256, outputDim: Int = 384) {
    self.embedder = MicroConditionEmbedder(frequencyDim: frequencyDim, outputDim: outputDim)
  }

  /// - Parameter ar: Aspect ratio values, shape [B].
  /// - Returns: [B, 384]
  func callAsFunction(_ ar: MLXArray) -> MLXArray {
    embedder(ar)
  }
}

// MARK: - Caption Projection

/// Projects T5-XXL embeddings from captionChannels (4096) to hiddenSize (1152).
///
/// Linear(4096, 1152) -> GELU(tanh) -> Linear(1152, 1152)
/// Applied once before the DiT blocks.
final class CaptionProjection: Module, @unchecked Sendable {
  @ModuleInfo var linear1: Linear
  @ModuleInfo var linear2: Linear

  init(captionChannels: Int, hiddenSize: Int) {
    self._linear1.wrappedValue = Linear(captionChannels, hiddenSize)
    self._linear2.wrappedValue = Linear(hiddenSize, hiddenSize)
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    var h = linear1(x)
    // geluApproximate uses compile(shapeless:true) which can return 0-D tensors under
    // memory pressure. Replace with direct gelu_new (tanh approximation) math.
    h = h * 0.5 * (1.0 + MLX.tanh(0.7978845608 * (h + 0.044715 * h * h * h)))
    h = linear2(h)
    return h
  }
}
