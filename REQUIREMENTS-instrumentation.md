# pixart-swift-mlx — Instrumentation Requirements

**Status:** Draft, awaiting implementation
**Pattern source:** [Vinetas `docs/INSTRUMENTATION_PLAN.md`](https://github.com/intrusive-memory/Vinetas/blob/development/docs/INSTRUMENTATION_PLAN.md) + Produciesta `Docs/TELEMETRY_IMPL_PATTERN.md`
**Host:** Vinetas
**Depends on:** SwiftTuberia ≥ 0.7.0 (for `TuberiaTensorStat` and the `TuberiaTelemetryEvent` boundaries this library nests under)
**Priority:** P0 — alongside flux-2-swift-mlx for math density, smaller surface, simpler implementation

---

## 1. Why instrument pixart-swift-mlx

`PixArtDiT` is the DiT backbone alternative to FLUX.2 in Vinetas. Where `Flux2Pipeline` orchestrates its own end-to-end generation, `PixArtDiT` is **a backbone only** — it plugs into a `DiffusionPipeline` (defined in SwiftTuberia) that drives the loop. This means:

- SwiftTuberia's `backboneForwardStart`/`Complete` events bracket every PixArt forward pass — those events live in the host adapter's view of Tuberia, not here.
- This library's job is to surface what happens **inside** `PixArtDiT.forward(_:)` and at `PixArtDiT.apply(weights:)` (the int4-dequantization boundary).

The instrumentation must surface:

- **Recipe selection.** `PixArtRecipe` (INT4 default, ~300 MB) and `PixArtFP16Recipe` are mutually exclusive. Bug reports about "PixArt looks wrong" almost always involve confusion about which recipe is live. Recipe selection becomes its own event.
- **The INT4 → FP16 dequantization at weight load** (`PixArtDiT.swift:218–241`). Each `.weight` key with `.scales` and `.biases` sidecars is dequantized: `floatWeight = packed * scales + biases`. Telemetry records the count of dequantized vs pass-through tensors and the input/output dtype histogram.
- **The known-omitted micro-conditioning** (`PixArtDiT.swift:151–154`). The int4 safetensors do not include `sizeEmbedder` / `arEmbedder` weights, so the forward pass skips micro-conditions. This is a load-bearing absence — if a future weight conversion adds those keys back, telemetry will record the divergence.
- **The documented silu workaround** (`PixArtDiT.swift:157–159`). `silu(x) = x * sigmoid(x)` is implemented manually because `MLX.silu` uses `compile(shapeless:true)` which can return 0-D tensors under memory pressure. Recording that this code path executed is a forensic data point for the next time that compile-shapeless bug bites someone.
- **DiT forward boundary stats** at the same per-step cadence Tuberia tracks: input latent, output noise-pred. Same `TensorStat` discipline as flux.
- **The variance channel discard** (`PixArtDiT.swift:175`). PixArt produces 8 output channels and keeps the first 4 (noise prediction) discarding the rest (variance). Wrong slicing here would produce silent corruption — record before/after stats.
- **Final layer AdaLN** behavior. The final layer uses 2-param AdaLN with the raw timestep embedding (before t_block); recording this is cheap and answers questions about timestep-conditioning bugs.

What it must NOT surface:
- Per-DiT-block events (28 blocks × 20 steps = 560 events per generation). Anomalies show up in the per-step output `noisePred` stat regardless of which block produced them.
- Per-attention-head events.
- Internal `dequantized(...)` kernel internals — block-aggregated counts are sufficient.

---

## 2. Coexistence with existing surfaces

| Surface | Purpose | Status |
|---|---|---|
| `PixArtDiT.isLoaded` (`PixArtDiT.swift:31`) | Boolean readiness flag | **Keep as-is.** Mirrored in the `weightApplyComplete` event payload. |
| `PixArtDiT.estimatedMemoryBytes` (`PixArtDiT.swift:182`) | Memory cost estimate | **Keep as-is.** Used by Tuberia's `memoryGate`; telemetry events at the boundary include this value. |
| `PixArtDiT.apply(weights:)` / `unload()` | Load/unload seam | **Keep as-is.** Instrumented via emission inside. |
| `PixArtRecipeError` enum | Validation errors | **Keep as-is.** Each throw site fires `errorThrown`. |
| `assert(blocks.count == 28, ...)` (`PixArtDiT.swift:106`) | Init invariant | **Keep as-is.** Telemetry doesn't replace asserts. |

---

## 3. Public types to add

```
Sources/PixArtBackbone/Telemetry/
  PixArtTelemetryEvent.swift
  PixArtTelemetryReporter.swift
```

`TuberiaTensorStat` is imported from SwiftTuberia.

### 3.1 `PixArtTelemetryEvent.swift`

```swift
@preconcurrency import MLX
import Foundation
import Tuberia

public enum PixArtTelemetryEvent: Sendable {

    // --- Recipe lifecycle ---
    case recipeSelected(name: String, version: String, expectedSteps: Int, expectedGuidanceScale: Double, allComponentIDs: [String])
    case recipeValidated(name: String, checksPassed: Int)
    case recipeValidationFailed(name: String, check: String, reason: String)

    // --- DiT init ---
    case ditInitialized(
        hiddenSize: Int,
        depth: Int,
        numHeads: Int,
        patchSize: Int,
        maxTextLength: Int,
        captionChannels: Int,
        peInterpolation: Float,
        baseSize: Int
    )

    // --- Weight load (boundary memory event on complete) ---
    case weightApplyStart(quantization: PixArtQuantization, weightKeyCount: Int)
    case weightApplyComplete(
        quantization: PixArtQuantization,
        totalKeys: Int,
        dequantizedKeys: Int,      // int4 -> fp16 dequantization count
        passThroughKeys: Int,      // already-fp16 keys loaded directly
        scalesBiasesSkipped: Int,  // .scales and .biases sidecar keys consumed
        sizeMB: Double,
        durationSeconds: Double
    )
    case weightUnload(restoredKeyCount: Int)
    case microConditioningStatus(present: Bool, sizeEmbedderFound: Bool, arEmbedderFound: Bool)

    // --- Forward pass (per scheduler step) ---
    case ditForwardStart(
        stepIndex: Int?,  // populated when caller passes it through; nil if standalone test
        batch: Int,
        latentShape: [Int],
        conditioningShape: [Int],
        timestepShape: [Int],
        inputLatentStat: TuberiaTensorStat,
        conditioningStat: TuberiaTensorStat
    )
    case patchEmbedComplete(stat: TuberiaTensorStat, gridH: Int, gridW: Int)
    case captionProjectionComplete(stat: TuberiaTensorStat)
    case timestepEmbeddingComplete(sinusoidalStat: TuberiaTensorStat, projectedStat: TuberiaTensorStat, tBlockStat: TuberiaTensorStat)
    case siluWorkaroundExecuted  // marker that the manual sigmoid path ran (it always does today; absence on a future event would mean MLX.silu got swapped back in)
    case finalLayerComplete(stat: TuberiaTensorStat)  // 8-channel output before variance discard
    case varianceChannelsDiscarded(beforeChannels: Int, afterChannels: Int, beforeStat: TuberiaTensorStat, afterStat: TuberiaTensorStat)
    case ditForwardComplete(
        stepIndex: Int?,
        outputStat: TuberiaTensorStat,  // [B, H/8, W/8, 4] noise prediction
        durationSeconds: Double
    )

    // --- Numerical anomaly side-channel ---
    case numericalAnomaly(phase: String, kind: AnomalyKind, stepIndex: Int?, stat: TuberiaTensorStat)

    // --- Error side-channel ---
    case errorThrown(phase: ErrorPhase, errorDescription: String)

    public enum PixArtQuantization: String, Sendable, Codable {
        case int4         // int4-quantized safetensors (~300 MB)
        case fp16         // dequantized fp16 safetensors (larger, slightly different math)
        case unknown      // weights loaded but format heuristic didn't match either
    }

    public enum AnomalyKind: String, Sendable {
        case nan
        case inf
        case outOfRange
        case zeroLatent
        case shapeMismatch
    }

    public enum ErrorPhase: String, Sendable {
        case ditInit
        case weightApply
        case forwardPass
        case recipeValidation
        case shapeMismatch
        case other
    }
}
```

**On `microConditioningStatus`.** Fires once per `weightApplyComplete` after the load loop scans for the size/ar embedder keys. Today both are `false` (the int4 conversion drops them). If a future PixArt-Sigma 2 lands those keys, this event flips to `true` and the host can confidently say "the math will now include micro-conditioning."

**On `siluWorkaroundExecuted`.** This is a tracer for the documented kernel-substitution at `PixArtDiT.swift:157–159`. It's a single event per forward (not per block). Cost is effectively zero — no tensor sampling, just a marker.

### 3.2 `PixArtTelemetryReporter.swift`

```swift
public protocol PixArtTelemetryReporter: Sendable {
    func capture(_ event: PixArtTelemetryEvent) async
}

public struct NoopPixArtTelemetryReporter: PixArtTelemetryReporter {
    public init() {}
    public func capture(_ event: PixArtTelemetryEvent) async {}
}
```

---

## 4. Injection points

### 4.1 `PixArtDiT` (the `@unchecked Sendable` class)

`PixArtDiT.swift:26` declares `public final class PixArtDiT: Module, Backbone, @unchecked Sendable`. Same lock-based pattern as flux:

```swift
import os.lock

public final class PixArtDiT: Module, Backbone, @unchecked Sendable {
    private let _telemetryLock = OSAllocatedUnfairLock<(any PixArtTelemetryReporter)?>(initialState: nil)

    public func setTelemetry(_ reporter: (any PixArtTelemetryReporter)?) {
        _telemetryLock.withLock { $0 = reporter }
    }

    fileprivate func currentTelemetry() -> (any PixArtTelemetryReporter)? {
        _telemetryLock.withLock { $0 }
    }
}
```

### 4.2 `PixArtRecipe` and `PixArtFP16Recipe` (value-type recipes)

These are `public struct` (`PixArtRecipe.swift:24`, `PixArtFP16Recipe.swift:26`). The standard pattern for value types is a defaulted telemetry parameter on `validate(...)`:

```swift
public func validate(telemetry: (any PixArtTelemetryReporter)? = nil) throws
```

The host wires this in at `DiffusionPipeline.init(recipe:)` time by wrapping the validate call (or calling `validate(telemetry:)` directly through the recipe protocol if Tuberia adds support).

Alternatively, since recipes are constructed in-line at pipeline-init time and validate fires once, the host can call `try await recipe.validate(telemetry: pixartAdapter)` explicitly before `DiffusionPipeline.init`. Either approach is acceptable; the implementation should pick one and document it.

### 4.3 `recipeSelected` emission point

This event fires at the host: when Vinetas's engine router picks PixArt, the Vinetas adapter emits `recipeSelected` directly. No library-side seam is needed for this case — the event still belongs to PixArt's event enum because it's about PixArt's recipes.

---

## 5. Per-event emission spec

| Event | Emission site | Notes |
|---|---|---|
| `recipeSelected` | Host-side, when engine router selects PixArt | Adapter constructs this event in Vinetas based on `EngineRouter` selection. |
| `recipeValidated` / `recipeValidationFailed` | Inside `PixArtRecipe.validate()` / `PixArtFP16Recipe.validate()` (lines 159, 135 respectively) | One pair per pipeline init. |
| `ditInitialized` | End of `PixArtDiT.init(configuration:)` (`PixArtDiT.swift:105`) | Once per pipeline. |
| `weightApplyStart` | Entry of `PixArtDiT.apply(weights:)` (`PixArtDiT.swift:189`) | Quantization detected by scanning for `.scales`/`.biases` keys before the load loop. |
| `weightApplyComplete` | Exit of `apply(weights:)` (`PixArtDiT.swift:247`) | **Memory snapshot.** Carries dequantization counts. |
| `weightUnload` | `unload()` (`PixArtDiT.swift:250`) | One per pipeline lifecycle. |
| `microConditioningStatus` | After the load loop in `apply(weights:)` scans for `sizeEmbedder`/`arEmbedder` keys | Once per `apply(weights:)`. |
| `ditForwardStart` | Start of `forward(_ input: BackboneInput)` (`PixArtDiT.swift:111`) — after extracting input but before patchEmbed | One per scheduler step. |
| `patchEmbedComplete` | After `let patched = patchEmbed(latents)` (`:126`) | One per step. **Cost-critical** — gated by `currentTelemetry()` lookup at function entry, cached locally. |
| `captionProjectionComplete` | After `let y = captionProjection(conditioning)` (`:142`) | One per step. |
| `timestepEmbeddingComplete` | After `let tBlock = tBlockLinear(...)` (`:159`) | One per step. Carries three stats (sinusoidal, projected, tBlock). |
| `siluWorkaroundExecuted` | Inside the block at `:159` immediately after the manual silu math runs | One per step. Cost: zero (no tensor sampling). |
| `finalLayerComplete` | After `var output = finalLayer(x, t: tRaw, ...)` (`:171`) | One per step. Captures the 8-channel output. |
| `varianceChannelsDiscarded` | At `output = output[..., ..., ..., 0..<4]` (`:175`) | One per step. Stats before (8ch) and after (4ch). |
| `ditForwardComplete` | Just before `return output` (`:177`) | One per step. **Brackets the per-step events.** |
| `numericalAnomaly` | Inside `TuberiaTensorStat.sample` post-construction, when hasNaN/hasInf observed | Side-channel, lives next to source event in JSONL. |
| `errorThrown` | All throw sites: `PixArtDiT.swift:61` (init), `apply(weights:)` (if extended to throw), `PixArtRecipe.swift:162, 168, 174`, `PixArtFP16Recipe.swift:138, 144, 150` | Fire immediately before throw. |

### Hot-path discipline

Each scheduler step issues **8 pixart events** when telemetry is on (forwardStart, patchEmbed, captionProj, timestepEmb, silu marker, finalLayer, varianceDiscard, forwardComplete). Of these, 6 carry `TuberiaTensorStat` = 48 MLX reductions. Plus Tuberia's bracketing events with their own stats. Plus flux/pixart's own loop is brackted by Tuberia.

This is a lot. **All 6 stat-carrying events must guard at the function entry with a single `currentTelemetry()` lookup**:

```swift
public func forward(_ input: BackboneInput) throws -> MLXArray {
    let telemetry = currentTelemetry()  // ONE lock acquisition per forward
    // ... math ...
    if let telemetry {
        await telemetry.capture(.ditForwardStart(/* ... */))
    }
    let patched = patchEmbed(latents)
    if let telemetry {
        await telemetry.capture(.patchEmbedComplete(stat: TuberiaTensorStat.sample(patched), /* ... */))
    }
    // ... etc ...
}
```

The lock is fetched once; all six stat samples and eight emissions reuse the captured pointer.

---

## 6. Adapter mapping (Vinetas host side)

`PixArtTelemetryAdapter` at `Vinetas/Telemetry/Adapters/PixArtTelemetryAdapter.swift`:

| Event | Sink phase | Memory snapshot? |
|---|---|---|
| `recipeSelected` | `pixart_recipe_selected` | no |
| `recipeValidated` | `pixart_recipe_validated` | no |
| `recipeValidationFailed` | `pixart_recipe_FAIL_<check>` | no |
| `ditInitialized` | `pixart_dit_init` | no |
| `weightApplyStart` | `pixart_weight_apply_start` | no |
| `weightApplyComplete` | `pixart_weight_load_complete` | **yes** (per INSTRUMENTATION_PLAN §3.1) |
| `weightUnload` | `pixart_weight_unload` | no |
| `microConditioningStatus` | `pixart_microcond_<present/absent>` | no |
| `ditForwardStart` | `pixart_dit_forward_start` | no (stepIndex on Snapshot) |
| `patchEmbedComplete` | `pixart_patch_embed` | no |
| `captionProjectionComplete` | `pixart_caption_proj` | no |
| `timestepEmbeddingComplete` | `pixart_timestep_emb` | no |
| `siluWorkaroundExecuted` | `pixart_silu_workaround` | no |
| `finalLayerComplete` | `pixart_final_layer` | no |
| `varianceChannelsDiscarded` | `pixart_variance_discard` | no |
| `ditForwardComplete` | `pixart_dit_forward_complete` | no |
| `numericalAnomaly` | `pixart_anomaly_<kind>` | no |
| `errorThrown` | `pixart_error_<phase>` | no |

Adapter must switch exhaustively.

---

## 7. Tests

Add to `Tests/PixArtBackboneTests/`:

| Test | Purpose |
|---|---|
| `PixArtTelemetryWeightApplyINT4Tests` | Load int4-quantized weights through `MockReporter`. Assert `weightApplyComplete(quantization: .int4, dequantizedKeys: > 0, passThroughKeys: 0, scalesBiasesSkipped: > 0)`. Assert `microConditioningStatus(present: false, sizeEmbedderFound: false, arEmbedderFound: false)` fires today. |
| `PixArtTelemetryWeightApplyFP16Tests` | Load fp16 weights through `MockReporter`. Assert `weightApplyComplete(quantization: .fp16, dequantizedKeys: 0, passThroughKeys: > 0)`. |
| `PixArtTelemetryForwardSequenceTests` | One forward call through `MockReporter`. Assert event sequence: `ditForwardStart` → `patchEmbedComplete` → `captionProjectionComplete` → `timestepEmbeddingComplete` → `siluWorkaroundExecuted` → `finalLayerComplete` → `varianceChannelsDiscarded` → `ditForwardComplete`. |
| `PixArtTelemetryVarianceDiscardTests` | Assert `varianceChannelsDiscarded.beforeChannels == 8 && .afterChannels == 4`. |
| `PixArtTelemetryNoopOverheadTests` | 20 forward calls with `nil` reporter and 20 with `NoopPixArtTelemetryReporter`. Wall-clock medians within ±2%. |
| `PixArtTelemetryAnomalyTests` | Inject a `patchEmbed` mock that returns a tensor with NaN. Assert `patchEmbedComplete.stat.hasNaN == true` and `numericalAnomaly(phase: "pixart_patch_embed", kind: .nan, ...)` fires alongside. |
| `PixArtTelemetryLockContentionTests` | Concurrent `setTelemetry` toggles + a running forward pass; TSan-clean. |

---

## 8. Out of scope

- Per-DiT-block events (28 blocks × 20 steps = 560 events/gen).
- Per-attention-head events.
- Internal kernel diagnostics (the `dequantized(...)` function internals).
- The 2D sinusoidal position-embedding computation (deterministic from grid dims; logging shape is enough).
- Training instrumentation. PixArt training isn't in Vinetas's scope.

---

## 9. Versioning

**Minor** version bump (additive). Pin floor: `0.7.0` post-release. Must ship AFTER SwiftTuberia ≥ 0.7.0 (for `TuberiaTensorStat`). Can ship in parallel with flux-2-swift-mlx (no cross-dependency between them).

---

## 10. Implementation checklist

- [ ] Add `Sources/PixArtBackbone/Telemetry/PixArtTelemetryEvent.swift` per §3.1
- [ ] Add `Sources/PixArtBackbone/Telemetry/PixArtTelemetryReporter.swift` per §3.2
- [ ] Add `OSAllocatedUnfairLock<(any PixArtTelemetryReporter)?>` + `setTelemetry`/`currentTelemetry` to `PixArtDiT`
- [ ] Wire all emission sites per §5; cache `currentTelemetry()` once per `forward(_:)` call
- [ ] Detect quantization in `apply(weights:)` by scanning for `.scales`/`.biases` keys before the load loop
- [ ] Add `microConditioningStatus` scan inside `apply(weights:)`
- [ ] Add defaulted `telemetry:` parameter to `PixArtRecipe.validate()` and `PixArtFP16Recipe.validate()`
- [ ] Ensure every `throw` site is preceded by `errorThrown` emit
- [ ] Add tests per §7
- [ ] Run baseline overhead test (20 forward calls, ±2% bound)
- [ ] Run TSan on lock-contention test
- [ ] Tag release with `MINOR` bump
