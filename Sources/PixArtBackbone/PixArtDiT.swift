import Foundation
@preconcurrency import MLX
import MLXNN
import Tuberia
import os.lock

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

  // MARK: - Telemetry Seam

  private let _telemetryLock = OSAllocatedUnfairLock<(any PixArtTelemetryReporter)?>(
    initialState: nil)

  public func setTelemetry(_ reporter: (any PixArtTelemetryReporter)?) {
    _telemetryLock.withLock { state in
      state = reporter
    }
  }

  fileprivate func currentTelemetry() -> (any PixArtTelemetryReporter)? {
    _telemetryLock.withLock { $0 }
  }

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
    // Slim telemetry: zero happy-path events. We sample the output stat ONCE at
    // exit and emit `numericalAnomaly(phase: .ditForward)` only when the output
    // is NaN/Inf/out-of-range/zero-latent. The host pipeline owns denoise-loop
    // boundary events; PixArt is the choke point that signals "the backbone
    // produced bad output" without flooding per-step.
    let telemetry = currentTelemetry()

    let latents = input.latents  // [B, H/8, W/8, 4]
    let conditioning = input.conditioning  // [B, seqLen, 4096]
    let conditioningMask = input.conditioningMask  // [B, seqLen]
    let timestep = input.timestep  // [B] or scalar

    let B = latents.dim(0)
    let spatialH = latents.dim(1)  // H/8
    let spatialW = latents.dim(2)  // W/8

    let gridH = spatialH / configuration.patchSize
    let gridW = spatialW / configuration.patchSize

    let patched = patchEmbed(latents)
    var x = patched.reshaped(B, gridH * gridW, configuration.hiddenSize)

    let posEmbed = get2DSinusoidalPositionEmbeddings(
      gridH: gridH,
      gridW: gridW,
      hiddenSize: configuration.hiddenSize,
      peInterpolation: configuration.peInterpolation,
      baseSize: configuration.baseSize / configuration.patchSize
    )
    x = x + posEmbed

    let y = captionProjection(conditioning)

    let tEmb = timestepSinusoidalEmbedding(timestep)
    let t = timestepEmbedder(tEmb)
    // silu uses compile(shapeless:true) which can return 0-D tensors under
    // memory pressure. Use silu(x) = x * sigmoid(x) directly.
    let tBlock = tBlockLinear(t * MLX.sigmoid(t))
    let tRaw = t

    for block in blocks {
      x = block(x, y: y, t: tBlock, mask: conditioningMask)
    }

    var output = finalLayer(x, t: tRaw, gridH: gridH, gridW: gridW)

    // output: [B, H/8, W/8, 8] — discard variance channels.
    output = output[0..., 0..., 0..., 0..<4]

    if let telemetry {
      let outputStat = TuberiaTensorStat.sample(output)
      if let anomaly = anomalyKind(for: outputStat) {
        let event = PixArtTelemetryEvent.numericalAnomaly(
          phase: .ditForward, kind: anomaly, stat: outputStat)
        Task { await telemetry.capture(event) }
      }
    }

    return output  // [B, H/8, W/8, 4]
  }

  // MARK: - WeightedSegment Protocol

  public var estimatedMemoryBytes: Int {
    // ~300 MB for int4 quantized PixArt-Sigma XL
    314_572_800
  }

  public var currentWeights: Tuberia.ModuleParameters? { weights }

  public func apply(weights: Tuberia.ModuleParameters) throws {
    // Load weight tensors into the model, handling both int4-quantized and fp16 safetensors.
    //
    // INT4 safetensors (pixart-sigma-xl-dit-int4):
    //   <key>.weight  — U32 packed, shape [outDim, inDim/8]
    //   <key>.scales  — F16, shape [outDim, inDim/64]
    //   <key>.biases  — F16, shape [outDim, inDim/64]   (zero-point = min value)
    // FP16 safetensors (pixart-sigma-xl-dit-fp16): <key>.weight is F16 [outDim, inDim].
    let telemetry = currentTelemetry()
    let start = Date()

    var params: [String: MLXArray] = [:]
    var scalesMap: [String: MLXArray] = [:]
    var biasesMap: [String: MLXArray] = [:]

    for (key, tensor) in weights.parameters {
      if key.hasSuffix(".scales") {
        scalesMap[String(key.dropLast(".scales".count))] = tensor
      } else if key.hasSuffix(".biases") {
        biasesMap[String(key.dropLast(".biases".count))] = tensor
      }
    }

    var paramCount = 0
    for (key, tensor) in weights.parameters {
      if key.hasSuffix(".scales") || key.hasSuffix(".biases") {
        continue
      }
      paramCount += 1

      let base = String(key.dropLast(".weight".count))
      if key.hasSuffix(".weight"),
        let scales = scalesMap[base],
        let biases = biasesMap[base],
        tensor.dtype == .uint32
      {
        let floatWeight = dequantized(
          tensor, scales: scales, biases: biases, groupSize: 64, bits: 4)
        params[key] = floatWeight.asType(.float16)
      } else {
        params[key] = tensor
      }
    }

    let mlxParams = MLXNN.ModuleParameters.unflattened(params)
    self.update(parameters: mlxParams)
    self.weights = weights
    self.isLoaded = true

    if let telemetry {
      let durationSeconds = Date().timeIntervalSince(start)
      let event = PixArtTelemetryEvent.weightLoadComplete(
        component: .dit,
        paramCount: paramCount,
        durationSeconds: durationSeconds)
      Task { await telemetry.capture(event) }
    }
  }

  public func unload() {
    let telemetry = currentTelemetry()
    self.weights = nil
    self.isLoaded = false

    if let telemetry {
      Task { await telemetry.capture(.weightUnloadComplete) }
    }
  }

  // MARK: - Anomaly classification

  /// Returns the anomaly kind for a sampled stat, or nil if the stat looks healthy.
  /// Used by `forward(_:)` to emit `numericalAnomaly` only on bad output.
  fileprivate func anomalyKind(for stat: TuberiaTensorStat) -> PixArtTelemetryEvent.AnomalyKind? {
    if stat.hasNaN { return .nan }
    if stat.hasInf { return .inf }
    if abs(stat.max) > TuberiaTensorStat.defaultOutOfRangeThreshold { return .outOfRange }
    if abs(stat.mean) < 1e-6 && stat.std < 1e-6 { return .zeroLatent }
    return nil
  }
}
