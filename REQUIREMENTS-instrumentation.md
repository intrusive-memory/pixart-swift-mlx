# pixart-swift-mlx — Instrumentation Requirements

**Status:** Shipped — slim boundary-only surface. 6 event cases, live on `main` since v0.7.0.
**Pattern source:** [`flux-2-swift-mlx/AGENTS.md §11`](../flux-2-swift-mlx/AGENTS.md#11-telemetry-chokepoint-convention-cross-library)
**Reference implementation:** [`../flux-2-swift-mlx/REQUIREMENTS-instrumentation.md`](../flux-2-swift-mlx/REQUIREMENTS-instrumentation.md)
**Host:** Vinetas (adapter not yet wired — see §6)
**Depends on:** SwiftTuberia ≥ 0.7.0 (for `TuberiaTensorStat`)

## Design principle

**PixArt is a backbone, not a pipeline.** `PixArtDiT` plugs into SwiftTuberia's `DiffusionPipeline`, which drives the denoising loop, manages the scheduler, owns VAE decode, and handles cancellation. This library does NOT own `pipelineInit`, `schedulerConfigured`, `denoiseLoopStart`/`End`, `vaeDecodeComplete`, or `generationCancelled` — those live at the Tuberia pipeline layer.

What PixArt owns: weight load/unload lifecycle on `PixArtDiT`, recipe validation outcome, and the one numerical chokepoint that matters — the DiT forward-pass output anomaly check.

Instrument **boundaries**, not internals. Zero events on the happy forward path; one anomaly side-channel only when the output is numerically bad.

---

## 1. Why instrument pixart-swift-mlx

`PixArtDiT.forward(_:)` is where NaN, Inf, and zero-latent anomalies originate. A single stat sample at forward exit answers "did the backbone produce bad output this step?" without per-stage overhead. `PixArtDiT.apply(weights:)` is the memory chokepoint. Recipe validation is the first place a mis-configured pipeline is caught.

---

## 2. Coexistence with existing surfaces

| Surface | Status |
|---|---|
| `PixArtDiT.isLoaded` | Keep as-is. Mirrored by `weightLoadComplete` / `weightUnloadComplete`. |
| `PixArtDiT.estimatedMemoryBytes` | Keep as-is. Used by Tuberia's memory gate. |
| `PixArtRecipeError` / `PixArtFP16RecipeError` | Instrumented — every throw fires `recipeValidationFailed` + `errorThrown`. |
| `assert(blocks.count == 28, ...)` | Keep as-is. Telemetry does not replace asserts. |

---

## 3. Public types

```
Sources/PixArtBackbone/Telemetry/
  PixArtTelemetryEvent.swift
  PixArtTelemetryReporter.swift
  README.md
```

### 3.1 `PixArtTelemetryEvent.swift` — 6 cases

```swift
public enum PixArtTelemetryEvent: Sendable {

  // Resource lifecycle
  case weightLoadComplete(component: WeightComponent, paramCount: Int, durationSeconds: Double)
  case weightUnloadComplete

  // Recipe configuration
  case recipeValidated(name: String, checksPassed: Int)
  case recipeValidationFailed(name: String, check: String, reason: String)

  // Side channels
  case numericalAnomaly(phase: AnomalyPhase, kind: AnomalyKind, stat: TuberiaTensorStat)
  case errorThrown(phase: ErrorPhase, errorDescription: String)

  public enum WeightComponent: String, Sendable, Codable {
    case dit  // only the DiT; encoder and VAE are instrumented at the Tuberia layer
  }

  public enum AnomalyPhase: String, Sendable {
    case weightLoad   // reserved — not currently emitted
    case ditForward   // active
  }

  public enum AnomalyKind: String, Sendable {
    case nan
    case inf
    case outOfRange   // |x| > TuberiaTensorStat.defaultOutOfRangeThreshold
    case zeroLatent   // mean ≈ 0 && std ≈ 0
  }

  public enum ErrorPhase: String, Sendable {
    case weightLoad         // reserved
    case forward            // reserved
    case recipeValidation   // active — all current errorThrown sites
    case other
  }
}
```

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

### 4.1 `setTelemetry` seam on `PixArtDiT`

`PixArtDiT` stores the reporter behind an `OSAllocatedUnfairLock` (required because the class is `@unchecked Sendable`):

```swift
// PixArtDiT.swift lines 37–48
private let _telemetryLock = OSAllocatedUnfairLock<(any PixArtTelemetryReporter)?>(initialState: nil)

public func setTelemetry(_ reporter: (any PixArtTelemetryReporter)?) {
  _telemetryLock.withLock { state in state = reporter }
}

fileprivate func currentTelemetry() -> (any PixArtTelemetryReporter)? {
  _telemetryLock.withLock { $0 }
}
```

`PixArtDiT` is the only type that exposes `setTelemetry`. Recipe types take the reporter as a parameter on `validate(telemetry:)`.

### 4.2 `telemetry:` parameter on `PixArtRecipe` / `PixArtFP16Recipe`

`PipelineRecipe` requires sync `validate() throws`. An async sibling was added alongside, preserving the conformance:

```swift
public func validate() throws { /* unchanged */ }

public func validate(telemetry: (any PixArtTelemetryReporter)? = nil) async throws {
  // same guard checks; emits recipeValidated / recipeValidationFailed / errorThrown
}
```

Hosts that want telemetry call `try await recipe.validate(telemetry: reporter)` before `DiffusionPipeline.init`. The sync path is unchanged for hosts that do not.

---

## 5. Per-event emission spec

| Event | Call site | Count |
|---|---|---|
| `weightLoadComplete` | `PixArtDiT.swift:254` — exit of `apply(weights:)` | 1 per `apply` |
| `weightUnloadComplete` | `PixArtDiT.swift:264` — `unload()` | 1 per `unload` |
| `recipeValidated` | `PixArtRecipe.swift:222` / `PixArtFP16Recipe.swift:199` | 1 on success |
| `recipeValidationFailed` + `errorThrown` | `PixArtRecipe.swift:193–219` / `PixArtFP16Recipe.swift:170–196`, paired before each `throw` | 0–1 per failing check |
| `numericalAnomaly` | `PixArtDiT.swift:183` — exit of `forward(_:)`, anomaly classifier only | 0 on healthy; 1 on anomaly |

### Hot-path discipline

The forward pass dispatches **zero events on the happy path** — one `currentTelemetry()` lock acquisition, one `TuberiaTensorStat.sample(output)` at exit, and at most one `Task { await telemetry.capture(event) }`:

```swift
// PixArtDiT.swift:178–185
if let telemetry {
  let outputStat = TuberiaTensorStat.sample(output)
  if let anomaly = anomalyKind(for: outputStat) {
    Task { await telemetry.capture(
      .numericalAnomaly(phase: .ditForward, kind: anomaly, stat: outputStat)) }
  }
}
```

Anomaly classifier (`PixArtDiT.anomalyKind(for:)`, lines 272–278): `nan` → `inf` → `outOfRange` → `zeroLatent` → `nil`.

---

## 6. Adapter mapping (Vinetas host side)

Vinetas adapter pending. When it lands, use an exhaustive switch (no `default:`):

| Event | Sink phase string | Memory snapshot? |
|---|---|---|
| `weightLoadComplete` | `pixart_weight_load_complete` | yes |
| `weightUnloadComplete` | `pixart_weight_unload` | no |
| `recipeValidated` | `pixart_recipe_validated` | no |
| `recipeValidationFailed` | `pixart_recipe_fail_<check>` | no |
| `numericalAnomaly` | `pixart_anomaly_<kind>` | no |
| `errorThrown` | `pixart_error_<phase>` | no |

---

## 7. Tests

| Test file | What it asserts |
|---|---|
| `PixArtTelemetryWeightApplyINT4Tests.swift` | `apply(weights:)` with synthetic INT4 dict emits exactly one `weightLoadComplete(component: .dit)`; sidecar `.scales`/`.biases` do not count toward `paramCount`. |
| `PixArtTelemetryWeightApplyFP16Tests.swift` | Same boundary shape for FP16. `unload()` emits one `weightUnloadComplete`. No events when reporter is nil. |
| `PixArtTelemetryAnomalyTests.swift` | NaN-poisoned input emits `numericalAnomaly(phase: .ditForward, kind: .nan)`. Clean input emits zero events from `forward(_:)`. |
| `PixArtTelemetryLockContentionTests.swift` | Concurrent `setTelemetry` toggles + running forward passes exercise the lock seam. TSan-clean. |

`MockReporter.swift` — `actor`-based test helper. Use a fresh `PixArtDiT` instance per test; shared fixtures are contaminated by concurrent suites.

Tests use swift-testing (`@Suite`, `@Test`, `#expect`). Lock-contention test uses XCTest due to swift-testing + macOS 26.2 SDK constraints.

---

## 8. Out of scope

These appear in the stale draft (archived in `docs/complete/stethoscope-furnace-01/`) but were never built or were cut post-mission:

- Per-stage forward events: `ditForwardStart`, `patchEmbedComplete`, `captionProjectionComplete`, `timestepEmbeddingComplete`, `siluWorkaroundExecuted`, `finalLayerComplete`, `varianceChannelsDiscarded`, `ditForwardComplete`
- Per-DiT-block events (28 blocks × N steps per generation)
- Per-attention-head events
- Internal dequantization kernel detail
- Host-side lifecycle: `recipeSelected`, `ditInitialized`, `weightApplyStart`, `microConditioningStatus`
- Per-step correlation via `BackboneInput.stepIndex` (needs a SwiftTuberia PR)
- Pipeline-level events owned by Tuberia: `pipelineInit`, `schedulerConfigured`, `denoiseLoopStart`, `denoiseLoopEnd`, `vaeDecodeComplete`, `generationCancelled`

Re-add any of the above only when a real incident demands finer localization.

---

## 9. Versioning

**Shipped:** v0.7.0. Pin floor: `0.7.0`. SwiftTuberia floor: `0.7.0`.
Additive only — hosts not calling `setTelemetry(_:)` or `validate(telemetry:)` see no behavior change.
