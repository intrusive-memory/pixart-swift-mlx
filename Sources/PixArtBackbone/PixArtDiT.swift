@preconcurrency import MLX
import Foundation
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

  private let _telemetryLock = OSAllocatedUnfairLock<(any PixArtTelemetryReporter)?>(initialState: nil)

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
    // Hot-path telemetry: ONE lock acquisition per forward. Sorties 3 and 6
    // reuse these bindings; no additional telemetry-lock reads below.
    let telemetry = currentTelemetry()
    let forwardStart = Date()
    _ = forwardStart  // Sortie 6 consumes this for the per-step duration field.

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

    // Sortie 5b: collect all telemetry events into a single array during synchronous forward
    // work, then dispatch exactly ONE Task at the end that awaits captures sequentially.
    // This guarantees deterministic event ordering at the reporter actor (Sortie 7a invariant).
    var pendingEvents: [PixArtTelemetryEvent] = []

    // Telemetry: forward-start event — fired after input extraction, before patchEmbed.
    // BackboneInput does not currently expose `stepIndex` (Q5.1 default: pass nil).
    if telemetry != nil {
      let stepIndex: Int? = nil
      let inputLatentStat = TuberiaTensorStat.sample(latents)
      let conditioningStat = TuberiaTensorStat.sample(conditioning)
      pendingEvents.append(
        .ditForwardStart(
          stepIndex: stepIndex,
          batch: latents.shape[0],
          latentShape: latents.shape,
          conditioningShape: conditioning.shape,
          timestepShape: timestep.shape,
          inputLatentStat: inputLatentStat,
          conditioningStat: conditioningStat))
      if inputLatentStat.hasNaN || inputLatentStat.hasInf {
        pendingEvents.append(
          .numericalAnomaly(
            phase: "pixart_dit_forward_start_input_latent",
            kind: inputLatentStat.hasNaN ? .nan : .inf,
            stepIndex: stepIndex,
            stat: inputLatentStat))
      }
      if conditioningStat.hasNaN || conditioningStat.hasInf {
        pendingEvents.append(
          .numericalAnomaly(
            phase: "pixart_dit_forward_start_conditioning",
            kind: conditioningStat.hasNaN ? .nan : .inf,
            stepIndex: stepIndex,
            stat: conditioningStat))
      }
    }

    // 1. Patch embedding: [B, H/8, W/8, 4] -> Conv2d -> [B, gridH, gridW, 1152]
    let patched = patchEmbed(latents)

    // Telemetry: patch-embed-complete event (gridH/gridW are the post-patch spatial grid).
    if telemetry != nil {
      let patchedStat = TuberiaTensorStat.sample(patched)
      pendingEvents.append(
        .patchEmbedComplete(stat: patchedStat, gridH: gridH, gridW: gridW))
      if patchedStat.hasNaN || patchedStat.hasInf {
        pendingEvents.append(
          .numericalAnomaly(
            phase: "pixart_patch_embed",
            kind: patchedStat.hasNaN ? .nan : .inf,
            stepIndex: nil,
            stat: patchedStat))
      }
    }

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

    // Telemetry: caption-projection-complete event.
    if telemetry != nil {
      let yStat = TuberiaTensorStat.sample(y)
      pendingEvents.append(.captionProjectionComplete(stat: yStat))
      if yStat.hasNaN || yStat.hasInf {
        pendingEvents.append(
          .numericalAnomaly(
            phase: "pixart_caption_proj",
            kind: yStat.hasNaN ? .nan : .inf,
            stepIndex: nil,
            stat: yStat))
      }
    }

    // 3. Timestep conditioning pipeline
    // Stage 1: Sinusoidal embedding [B] -> [B, 256]
    let tEmb = timestepSinusoidalEmbedding(timestep)

    // Stage 2: MLP projection [B, 256] -> [B, 1152]
    let t = timestepEmbedder(tEmb)

    // Stage 3: Micro-conditions (resolution + aspect ratio) are NOT included in the
    // int4-quantized safetensors — the sizeEmbedder and arEmbedder weights are absent,
    // meaning micro-conditioning was omitted from the weight conversion.
    // Skip adding micro-conditions: t remains as timestepEmbedder(tEmb).

    // Stage 4: t_block: SiLU -> Linear(1152, 6*1152) = [B, 6912]
    // silu uses compile(shapeless:true) which can return 0-D tensors under memory pressure.
    // Replace with direct math: silu(x) = x * sigmoid(x)
    let tBlock = tBlockLinear(t * MLX.sigmoid(t))

    // Telemetry: timestep-embedding-complete event + the silu-workaround marker.
    // tEmb = sinusoidal, t = projected (post-timestepEmbedder MLP), tBlock = post-silu+linear.
    if telemetry != nil {
      let sinusoidalStat = TuberiaTensorStat.sample(tEmb)
      let projectedStat = TuberiaTensorStat.sample(t)
      let tBlockStat = TuberiaTensorStat.sample(tBlock)
      pendingEvents.append(
        .timestepEmbeddingComplete(
          sinusoidalStat: sinusoidalStat,
          projectedStat: projectedStat,
          tBlockStat: tBlockStat))
      pendingEvents.append(.siluWorkaroundExecuted)
      if sinusoidalStat.hasNaN || sinusoidalStat.hasInf {
        pendingEvents.append(
          .numericalAnomaly(
            phase: "pixart_timestep_emb_sinusoidal",
            kind: sinusoidalStat.hasNaN ? .nan : .inf,
            stepIndex: nil,
            stat: sinusoidalStat))
      }
      if projectedStat.hasNaN || projectedStat.hasInf {
        pendingEvents.append(
          .numericalAnomaly(
            phase: "pixart_timestep_emb_projected",
            kind: projectedStat.hasNaN ? .nan : .inf,
            stepIndex: nil,
            stat: projectedStat))
      }
      if tBlockStat.hasNaN || tBlockStat.hasInf {
        pendingEvents.append(
          .numericalAnomaly(
            phase: "pixart_timestep_emb_t_block",
            kind: tBlockStat.hasNaN ? .nan : .inf,
            stepIndex: nil,
            stat: tBlockStat))
      }
    }

    // Save raw timestep embedding for final layer (before t_block)
    let tRaw = t  // [B, 1152]

    // 4. Run through 28 DiT blocks
    for block in blocks {
      x = block(x, y: y, t: tBlock, mask: conditioningMask)
    }

    // 5. Final layer: AdaLN(2-param) + linear + unpatchify
    // Uses raw timestep embedding (before t_block)
    var output = finalLayer(x, t: tRaw, gridH: gridH, gridW: gridW)

    // Telemetry: final-layer-complete event — 8-channel output BEFORE Sortie 3's variance discard.
    if telemetry != nil {
      let finalStat = TuberiaTensorStat.sample(output)
      pendingEvents.append(.finalLayerComplete(stat: finalStat))
      if finalStat.hasNaN || finalStat.hasInf {
        pendingEvents.append(
          .numericalAnomaly(
            phase: "pixart_final_layer",
            kind: finalStat.hasNaN ? .nan : .inf,
            stepIndex: nil,
            stat: finalStat))
      }
    }

    // Sortie 3: variance-discard cast-site instrumentation. Sample the 8-channel output
    // BEFORE the slice and the 4-channel output AFTER the slice — this is the
    // "communication error" boundary where wrong slicing would silently corrupt output.
    // The slice itself MUST execute regardless of telemetry state (production correctness).
    let varianceBeforeStat: TuberiaTensorStat? = telemetry.map { _ in TuberiaTensorStat.sample(output) }

    // output: [B, H/8, W/8, 8]
    // Discard variance channels (last 4), keep noise prediction (first 4)
    output = output[0..., 0..., 0..., 0..<4]

    if telemetry != nil, let beforeStat = varianceBeforeStat {
      let afterStat = TuberiaTensorStat.sample(output)
      pendingEvents.append(
        .varianceChannelsDiscarded(
          beforeChannels: 8,
          afterChannels: 4,
          beforeStat: beforeStat,
          afterStat: afterStat))
      if beforeStat.hasNaN || beforeStat.hasInf {
        pendingEvents.append(
          .numericalAnomaly(
            phase: "pixart_variance_before",
            kind: beforeStat.hasNaN ? .nan : .inf,
            stepIndex: nil,
            stat: beforeStat))
      }
      if afterStat.hasNaN || afterStat.hasInf {
        pendingEvents.append(
          .numericalAnomaly(
            phase: "pixart_variance_after",
            kind: afterStat.hasNaN ? .nan : .inf,
            stepIndex: nil,
            stat: afterStat))
      }
    }

    // Dispatch a SINGLE Task that awaits all captures sequentially, guaranteeing
    // deterministic event ordering at the reporter actor (Sortie 7a sequence invariant).
    if let telemetry, !pendingEvents.isEmpty {
      let events = pendingEvents  // explicit value capture; no reference-capture ambiguity
      Task {
        for event in events {
          await telemetry.capture(event)
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
    //   - <key>.weight  — U32 packed, shape [outDim, inDim/8]
    //   - <key>.scales  — F16, shape [outDim, numGroups]  (numGroups = inDim/64)
    //   - <key>.biases  — F16, shape [outDim, numGroups]  (zero-point = min value)
    //   These are dequantized at load time: floatWeight = packed_values * scales + biases
    //
    // FP16 safetensors (pixart-sigma-xl-dit-fp16, produced by dequantize_dit_to_fp16.py):
    //   - <key>.weight  — F16, shape [outDim, inDim]
    //   No .scales or .biases keys present. Load directly without dequantization.
    //
    // The additive layer bias (.bias, singular) is always loaded as-is regardless of format.

    // Sortie 3: weight-apply telemetry. ONE lock acquisition per apply(weights:). Events
    // accumulate into pendingEvents and are dispatched in a single Task at the end (mirrors
    // Sortie 5b's forward-pass coalesced pattern).
    let telemetry = currentTelemetry()
    let start = Date()
    var pendingEvents: [PixArtTelemetryEvent] = []

    // Quantization detection — scan keys for .scales/.biases sidecars (int4) or fp16 .weight
    // dtype. This runs unconditionally (used to populate the weight-apply-start payload).
    let weightKeyCount = weights.parameters.count
    let quantization: PixArtTelemetryEvent.PixArtQuantization = {
      var hasSidecar = false
      var hasFP16Weight = false
      for (key, tensor) in weights.parameters {
        if key.hasSuffix(".scales") || key.hasSuffix(".biases") {
          hasSidecar = true
          break
        }
        if key.hasSuffix(".weight") && tensor.dtype == .float16 {
          hasFP16Weight = true
        }
      }
      if hasSidecar { return .int4 }
      if hasFP16Weight { return .fp16 }
      return .unknown
    }()

    if telemetry != nil {
      pendingEvents.append(
        .weightApplyStart(quantization: quantization, weightKeyCount: weightKeyCount))
    }

    var params: [String: MLXArray] = [:]
    var scalesMap: [String: MLXArray] = [:]
    var biasesMap: [String: MLXArray] = [:]

    // First pass: collect scales and biases (quantization zero-points, int4 only)
    for (key, tensor) in weights.parameters {
      if key.hasSuffix(".scales") {
        let base = String(key.dropLast(".scales".count))
        scalesMap[base] = tensor
      } else if key.hasSuffix(".biases") {
        let base = String(key.dropLast(".biases".count))
        biasesMap[base] = tensor
      }
    }

    // Per-branch counters for the weight-apply-complete payload (Q3.2 default: count .weight
    // keys consumed via each path, NOT individual MLX kernel invocations).
    var dequantizedKeys = 0
    var passThroughKeys = 0
    var scalesBiasesSkipped = 0

    // Second pass: dequantize int4 weight tensors or pass through fp16 weights unchanged
    for (key, tensor) in weights.parameters {
      // Skip quantization sidecar keys (handled via scalesMap/biasesMap above)
      if key.hasSuffix(".scales") || key.hasSuffix(".biases") {
        scalesBiasesSkipped += 1
        continue
      }

      let base = String(key.dropLast(".weight".count))
      if key.hasSuffix(".weight"),
        let scales = scalesMap[base],
        let biases = biasesMap[base],
        tensor.dtype == .uint32
      {
        // INT4 path: dequantize packed uint32 to float16
        // floatWeight[i, j] = packed_values[i, j] * scales[i, g] + biases[i, g]
        // where g = j / groupSize
        let floatWeight = dequantized(
          tensor, scales: scales, biases: biases, groupSize: 64, bits: 4)
        params[key] = floatWeight.asType(.float16)
        dequantizedKeys += 1
      } else {
        // FP16 path (or any non-quantized tensor): load directly
        params[key] = tensor
        passThroughKeys += 1
      }
    }

    let mlxParams = MLXNN.ModuleParameters.unflattened(params)
    self.update(parameters: mlxParams)

    self.weights = weights

    // Micro-conditioning status scan — fires once per apply(weights:). Today both flags are
    // false (int4 conversion drops these keys); if a future weight conversion adds them
    // back, this event flips and the host can confirm micro-conditioning is now live.
    if telemetry != nil {
      var sizeEmbedderFound = false
      var arEmbedderFound = false
      for key in weights.parameters.keys {
        if key.contains("sizeEmbedder") { sizeEmbedderFound = true }
        if key.contains("arEmbedder") { arEmbedderFound = true }
      }
      pendingEvents.append(
        .microConditioningStatus(
          present: sizeEmbedderFound || arEmbedderFound,
          sizeEmbedderFound: sizeEmbedderFound,
          arEmbedderFound: arEmbedderFound))

      let durationSeconds = Date().timeIntervalSince(start)
      let sizeMB = Double(estimatedMemoryBytes) / 1_048_576.0
      pendingEvents.append(
        .weightApplyComplete(
          quantization: quantization,
          totalKeys: weightKeyCount,
          dequantizedKeys: dequantizedKeys,
          passThroughKeys: passThroughKeys,
          scalesBiasesSkipped: scalesBiasesSkipped,
          sizeMB: sizeMB,
          durationSeconds: durationSeconds))
    }

    self.isLoaded = true

    // Dispatch all accumulated events in a single Task (mirrors Sortie 5b's pattern).
    if let telemetry, !pendingEvents.isEmpty {
      let events = pendingEvents
      Task {
        for event in events {
          await telemetry.capture(event)
        }
      }
    }
  }

  public func unload() {
    // Sortie 3: capture restoredKeyCount BEFORE clearing weights, then dispatch a single
    // weight-unload event. Single event = no pendingEvents array needed.
    let telemetry = currentTelemetry()
    let restoredKeyCount = weights?.parameters.count ?? 0

    self.weights = nil
    self.isLoaded = false

    if let telemetry {
      Task {
        await telemetry.capture(.weightUnload(restoredKeyCount: restoredKeyCount))
      }
    }
  }
}
