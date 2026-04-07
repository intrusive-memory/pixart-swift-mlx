@preconcurrency import MLX
import MLXNN
import Tuberia

/// PixArt-Sigma DiT transformer backbone.
///
/// Implements the full PixArt-Sigma XL architecture (~600M parameters):
/// - Patch embedding (Conv2d + 2D sinusoidal position embeddings)
/// - Caption projection (T5 4096 -> 1152)
/// - Timestep conditioning (sinusoidal + MLP + micro-conditions + t_block)
/// - 28 DiT transformer blocks (self-attention + cross-attention + FFN)
/// - Final layer (AdaLN + linear projection + unpatchify)
///
/// Conforms to SwiftTubería's `Backbone` and `WeightedSegment` protocols.
///
/// Shape contract:
/// ```
/// inlet:  BackboneInput {
///             latents:          [B, H/8, W/8, 4]
///             conditioning:     [B, 120, 4096]
///             conditioningMask: [B, 120]
///             timestep:         [B]
///         }
/// outlet: MLXArray [B, H/8, W/8, 4]  (variance channels discarded)
/// ```
public final class PixArtDiT: Module, Backbone, @unchecked Sendable {
  public typealias Configuration = PixArtDiTConfiguration

  private let configuration: Configuration
  private var weights: Tuberia.ModuleParameters?
  public private(set) var isLoaded: Bool = false

  // -- Patch Embedding --
  let patchEmbed: Conv2d

  // -- Caption Projection --
  let captionProjection: CaptionProjection

  // -- Timestep Conditioning --
  let timestepEmbedder: TimestepEmbedder
  let sizeEmbedder: SizeEmbedder
  let arEmbedder: AspectRatioEmbedder

  // -- t_block: SiLU -> Linear(hiddenSize, 6 * hiddenSize) --
  @ModuleInfo(key: "t_block_linear") var tBlockLinear: Linear

  // -- 28 DiT Blocks --
  let blocks: [DiTBlock]

  // -- Final Layer --
  let finalLayer: FinalLayer

  // MARK: - Backbone Protocol

  public var expectedConditioningDim: Int { configuration.captionChannels }
  public var outputLatentChannels: Int { 4 }
  public var expectedMaxSequenceLength: Int { configuration.maxTextLength }

  // MARK: - Initialization

  public required init(configuration: Configuration) throws {
    self.configuration = configuration

    // Patch embedding: Conv2d(inChannels, hiddenSize, kernel=patchSize, stride=patchSize)
    // MLX Conv2d weight layout: [O, kH, kW, I]
    self.patchEmbed = Conv2d(
      inputChannels: configuration.inChannels,
      outputChannels: configuration.hiddenSize,
      kernelSize: IntOrPair(configuration.patchSize),
      stride: IntOrPair(configuration.patchSize),
      bias: true
    )

    // Caption projection: Linear(4096, 1152) -> GELU(tanh) -> Linear(1152, 1152)
    self.captionProjection = CaptionProjection(
      captionChannels: configuration.captionChannels,
      hiddenSize: configuration.hiddenSize
    )

    // Timestep conditioning pipeline
    self.timestepEmbedder = TimestepEmbedder(hiddenSize: configuration.hiddenSize)
    self.sizeEmbedder = SizeEmbedder()
    self.arEmbedder = AspectRatioEmbedder()

    // t_block: SiLU -> Linear(hiddenSize, 6 * hiddenSize)
    self._tBlockLinear.wrappedValue = Linear(configuration.hiddenSize, 6 * configuration.hiddenSize)

    // 28 DiT blocks
    self.blocks = (0..<configuration.depth).map { _ in
      DiTBlock(
        hiddenSize: configuration.hiddenSize,
        numHeads: configuration.numHeads,
        headDim: configuration.headDim,
        mlpRatio: configuration.mlpRatio
      )
    }

    // Final layer
    self.finalLayer = FinalLayer(
      hiddenSize: configuration.hiddenSize,
      patchSize: configuration.patchSize,
      outChannels: configuration.outChannels
    )

    super.init()
    assert(blocks.count == 28, "PixArt-Sigma XL must have exactly 28 DiT blocks")
  }

  // MARK: - Forward Pass

  public func forward(_ input: BackboneInput) throws -> MLXArray {
    let latents = input.latents  // [B, H/8, W/8, 4]
    let conditioning = input.conditioning  // [B, seqLen, 4096]
    let conditioningMask = input.conditioningMask  // [B, seqLen]
    let timestep = input.timestep  // [B] or scalar

    let B = latents.dim(0)
    let spatialH = latents.dim(1)  // H/8
    let spatialW = latents.dim(2)  // W/8

    // Grid dimensions after patch embedding
    let gridH = spatialH / configuration.patchSize
    let gridW = spatialW / configuration.patchSize

    // 1. Patch embedding: [B, H/8, W/8, 4] -> Conv2d -> [B, gridH, gridW, 1152]
    let patched = patchEmbed(latents)

    // Flatten to token sequence: [B, gridH * gridW, 1152]
    var x = patched.reshaped(B, gridH * gridW, configuration.hiddenSize)

    // Add 2D sinusoidal position embeddings
    let posEmbed = get2DSinusoidalPositionEmbeddings(
      gridH: gridH,
      gridW: gridW,
      hiddenSize: configuration.hiddenSize,
      peInterpolation: configuration.peInterpolation,
      baseSize: configuration.baseSize / configuration.patchSize
    )
    x = x + posEmbed

    // 2. Caption projection: [B, seqLen, 4096] -> [B, seqLen, 1152]
    let y = captionProjection(conditioning)

    // 3. Timestep conditioning pipeline
    // Stage 1: Sinusoidal embedding [B] -> [B, 256]
    let tEmb = timestepSinusoidalEmbedding(timestep)

    // Stage 2: MLP projection [B, 256] -> [B, 1152]
    var t = timestepEmbedder(tEmb)

    // Stage 3: Add micro-conditions (resolution + aspect ratio)
    // For inference, use the target spatial dimensions as micro-conditions
    let targetH = Float(spatialH * 8)  // Full pixel height
    let targetW = Float(spatialW * 8)  // Full pixel width
    let ar = targetH / targetW

    let sizeInput = MLXArray([targetH, targetW]).reshaped(1, 2)
    let sizeInputBroadcast = MLX.broadcast(sizeInput, to: [B, 2])
    let sizeEmb = sizeEmbedder(sizeInputBroadcast)  // [B, 768]

    let arInput = MLXArray([ar]).reshaped(1)
    let arInputBroadcast = MLX.broadcast(arInput, to: [B])
    let arEmb = arEmbedder(arInputBroadcast)  // [B, 384]

    // Concatenate micro-conditions: [B, 768] + [B, 384] = [B, 1152]
    let microCond = concatenated([sizeEmb, arEmb], axis: -1)
    t = t + microCond

    // Stage 4: t_block: SiLU -> Linear(1152, 6*1152) = [B, 6912]
    // silu uses compile(shapeless:true) which can return 0-D tensors under memory pressure.
    // Replace with direct math: silu(x) = x * sigmoid(x)
    let tBlock = tBlockLinear(t * MLX.sigmoid(t))

    // Save raw timestep embedding for final layer (before t_block)
    let tRaw = t  // [B, 1152]

    // 4. Run through 28 DiT blocks
    for block in blocks {
      x = block(x, y: y, t: tBlock, mask: conditioningMask)
    }

    // 5. Final layer: AdaLN(2-param) + linear + unpatchify
    // Uses raw timestep embedding (before t_block)
    var output = finalLayer(x, t: tRaw, gridH: gridH, gridW: gridW)

    // output: [B, H/8, W/8, 8]
    // Discard variance channels (last 4), keep noise prediction (first 4)
    output = output[0..., 0..., 0..., 0..<4]

    return output  // [B, H/8, W/8, 4]
  }

  // MARK: - WeightedSegment Protocol

  public var estimatedMemoryBytes: Int {
    // ~300 MB for int4 quantized PixArt-Sigma XL
    314_572_800
  }

  public var currentWeights: Tuberia.ModuleParameters? { weights }

  public func apply(weights: Tuberia.ModuleParameters) throws {
    let mlxParams = MLXNN.ModuleParameters.unflattened(weights.parameters)
    self.update(parameters: mlxParams)
    self.weights = weights
    self.isLoaded = true
  }

  public func unload() {
    self.weights = nil
    self.isLoaded = false
  }
}
