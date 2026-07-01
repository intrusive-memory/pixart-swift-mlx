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

  /// Returns the reporter to use for the current emission site.
  ///
  /// The instance reporter (installed via `setTelemetry(_:)`) takes precedence.
  /// If no instance reporter is installed, falls back to the process-wide
  /// `PixArtTelemetry.current` reporter.  Returns `nil` when neither is set.
  private var effectiveReporter: (any PixArtTelemetryReporter)? {
    currentTelemetry() ?? PixArtTelemetry.current
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
    // Telemetry: emit `backboneForwardComplete(stat:)` once per forward,
    // carrying the sampled output tensor statistics, and additionally emit
    // `numericalAnomaly(phase: .ditForward, ...)` when the output is
    // NaN/Inf/out-of-range/zero-latent. The happy-path event was added so
    // downstream pipelines (SwiftTuberia / Vinetas) can attribute quality
    // regressions like color cast or saturation clipping to either the
    // conditioning origin (visible from step 0) or the denoise loop
    // (accumulating over steps); the anomaly event remains the smoke alarm.
    let telemetry = effectiveReporter

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
      let completeEvent = PixArtTelemetryEvent.backboneForwardComplete(stat: outputStat)
      let anomalyEvent: PixArtTelemetryEvent? = anomalyKind(for: outputStat).map { kind in
        .numericalAnomaly(phase: .ditForward, kind: kind, stat: outputStat)
      }
      Task {
        await telemetry.capture(completeEvent)
        if let anomalyEvent {
          await telemetry.capture(anomalyEvent)
        }
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
    //
    // Memory contract: the int4 checkpoint MUST stay quantized in memory. We do NOT
    // dequantize to fp16 (that would blow the DiT up from ~0.3 GB to ~1.2 GB). Instead
    // we convert the quantizable `Linear` projections that ship an int4 sidecar into
    // `QuantizedLinear` and route the packed uint32 weight + scales + biases straight in.
    // The forward pass then runs the int4 `quantizedMM` kernel — mathematically identical
    // to `dequantized(w) @ x` — with no fp16 weight tensor ever materialized.
    let telemetry = effectiveReporter
    let start = Date()

    // 1. Discover which module paths ship as int4 in this weight set. A quantized
    //    projection has all three of: <base>.weight (uint32 packed), <base>.scales,
    //    and <base>.biases. fp16-stored layers (patchEmbed, embedders, .bias, etc.)
    //    have none of these and take the passthrough path unchanged.
    var scalesBases = Set<String>()
    var biasesBases = Set<String>()
    for (key, _) in weights.parameters {
      if key.hasSuffix(".scales") {
        scalesBases.insert(String(key.dropLast(".scales".count)))
      } else if key.hasSuffix(".biases") {
        biasesBases.insert(String(key.dropLast(".biases".count)))
      }
    }

    var quantizedBases = Set<String>()
    var paramCount = 0
    for (key, tensor) in weights.parameters {
      if key.hasSuffix(".scales") || key.hasSuffix(".biases") {
        continue
      }
      paramCount += 1

      guard key.hasSuffix(".weight"), tensor.dtype == .uint32 else { continue }
      let base = String(key.dropLast(".weight".count))
      if scalesBases.contains(base) && biasesBases.contains(base) {
        quantizedBases.insert(base)
      }
    }

    // 2. Convert the matching `Linear` leaves into `QuantizedLinear` (groupSize 64,
    //    4 bits, affine) so they expose weight/scales/biases parameters shaped to
    //    receive the packed int4 values. `quantizeSingle` skips leaves that are
    //    already quantized, so re-applying an int4 set is idempotent.
    if !quantizedBases.isEmpty {
      quantize(model: self) { path, _ in
        quantizedBases.contains(path) ? (groupSize: 64, bits: 4, mode: .affine) : nil
      }
    }

    // 3. Route every tensor straight into the model with no dequantization. The
    //    QuantizedLinear leaves accept the packed uint32 weight + fp16 scales/biases
    //    (+ fp16 bias); every other layer loads its fp16 weight unchanged.
    let mlxParams = MLXNN.ModuleParameters.unflattened(weights.parameters)
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
    let telemetry = effectiveReporter

    // Drop the stored input params, then replace every resident module parameter
    // (including the packed int4 weight/scales/biases held by any QuantizedLinear
    // leaves) with an empty array so the underlying buffers are released rather than
    // lingering until the DiT is deallocated. A subsequent `apply(weights:)` routes
    // the full, correctly-shaped weight set back in.
    self.weights = nil
    let resident = self.parameters().flattened()
    if !resident.isEmpty {
      var cleared: [String: MLXArray] = [:]
      cleared.reserveCapacity(resident.count)
      for (key, array) in resident {
        cleared[key] = MLXArray.zeros([0], dtype: array.dtype)
      }
      self.update(parameters: MLXNN.ModuleParameters.unflattened(cleared))
    }
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
