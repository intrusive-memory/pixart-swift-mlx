# pixart-swift-mlx — Instrumentation Requirements

**Status:** Implemented (slim form) in v0.7.0.
**Shipped surface:** 6 events. See [`Sources/PixArtBackbone/Telemetry/README.md`](../../../Sources/PixArtBackbone/Telemetry/README.md) for the consumer guide.
**Pattern source:** [Vinetas `docs/INSTRUMENTATION_PLAN.md`](https://github.com/intrusive-memory/Vinetas/blob/development/docs/INSTRUMENTATION_PLAN.md) + Produciesta `Docs/TELEMETRY_IMPL_PATTERN.md` + boundary-only convention from `flux-2-swift-mlx/AGENTS.md §11`.
**Host:** Vinetas (integration still TODO — see §11).
**Depends on:** SwiftTuberia ≥ 0.7.0 for `TuberiaTensorStat` (the only Tuberia type still surfaced).
**Mission of record:** OPERATION STETHOSCOPE FURNACE — this document lives in the mission's archive folder alongside [`RECONCILIATION.md`](RECONCILIATION.md), which details what the mission produced vs. what shipped after the post-mission pare-down.

> **What changed from the original draft.** This document was first written as an 18-event blueprint for full per-stage forward-pass instrumentation. The mission produced that surface, but a post-mission pass (`b585120 "updating testing strategery"`) cut it down to a 6-event boundary-only surface before tagging v0.7.0. Sections 3, 5, 7, and 10 below now describe the shipped form. Section 11 catalogs what was removed and the reasoning, so future expansion has a clear starting point.

---

## 1. Why instrument pixart-swift-mlx

`PixArtDiT` is the DiT backbone alternative to FLUX.2 in Vinetas. Where `Flux2Pipeline` orchestrates its own end-to-end generation, `PixArtDiT` is **a backbone only** — it plugs into a `DiffusionPipeline` (defined in SwiftTuberia) that drives the loop. This means:

- SwiftTuberia's `backboneForwardStart`/`Complete` events bracket every PixArt forward pass — those events live in the host adapter's view of Tuberia, not here.
- This library's job is to surface what happens **inside** `PixArtDiT.forward(_:)` and at `PixArtDiT.apply(weights:)` (the int4-dequantization boundary).

The library currently surfaces:

- **Recipe validation outcome** — pass (with check count) or fail (with the specific check and reason). Each recipe type emits its own.
- **Weight-load lifecycle** — the boundary memory event that says "the DiT is now hot" with param count and duration.
- **Weight unload** — symmetric release event.
- **DiT-forward numerical anomaly** — sampled once at forward exit. If the output is NaN, Inf, out-of-range, or zero-latent, emit a single event with the full `TuberiaTensorStat` and the anomaly kind. Silent on healthy output.
- **Thrown errors** — every `throw` in the recipe validation path fires `errorThrown` before throwing.

What it deliberately does NOT surface (and what was cut from the original draft — see §11):

- Per-stage forward-pass events (patchEmbed, captionProjection, timestepEmbedding, finalLayer, varianceDiscard, ditForwardStart/Complete).
- Per-DiT-block events (28 blocks × 20 steps = 560 events per generation).
- Per-attention-head events.
- The `siluWorkaroundExecuted` marker.
- `microConditioningStatus`, `recipeSelected`, `ditInitialized`, `weightApplyStart` — host-side or scoped-out.
- Internal `dequantized(...)` kernel internals.

Anomalies at any of the cut stages show up in the per-forward output stat regardless of which stage produced them. If a real incident demands finer localization, expand the surface then — not before.

---

## 2. Coexistence with existing surfaces

| Surface | Purpose | Status |
|---|---|---|
| `PixArtDiT.isLoaded` | Boolean readiness flag | **Keep as-is.** Mirrored implicitly by the `weightLoadComplete` / `weightUnloadComplete` event pair. |
| `PixArtDiT.estimatedMemoryBytes` | Memory cost estimate | **Keep as-is.** Used by Tuberia's `memoryGate`; the host bracketing events carry this value. |
| `PixArtDiT.apply(weights:)` / `unload()` | Load/unload seam | **Instrumented** — emits `weightLoadComplete(paramCount, durationSeconds)` and `weightUnloadComplete`. |
| `PixArtRecipeError` enum | Validation errors | **Instrumented** — every throw site fires `recipeValidationFailed` + `errorThrown` before throwing. |
| `assert(blocks.count == 28, ...)` | Init invariant | **Keep as-is.** Telemetry does not replace asserts. |

---

## 3. Public types (as shipped in v0.7.0)

```
Sources/PixArtBackbone/Telemetry/
  PixArtTelemetryEvent.swift
  PixArtTelemetryReporter.swift
  README.md                    ← consumer guide
```

`TuberiaTensorStat` is imported from SwiftTuberia.

### 3.1 `PixArtTelemetryEvent.swift` — 6 cases

```swift
import Foundation
@preconcurrency import MLX
import Tuberia

public enum PixArtTelemetryEvent: Sendable {

  // --- Resource lifecycle ---
  case weightLoadComplete(component: WeightComponent, paramCount: Int, durationSeconds: Double)
  case weightUnloadComplete

  // --- Recipe configuration ---
  case recipeValidated(name: String, checksPassed: Int)
  case recipeValidationFailed(name: String, check: String, reason: String)

  // --- Side channels ---
  case numericalAnomaly(phase: AnomalyPhase, kind: AnomalyKind, stat: TuberiaTensorStat)
  case errorThrown(phase: ErrorPhase, errorDescription: String)

  public enum WeightComponent: String, Sendable, Codable {
    case dit
  }

  public enum AnomalyPhase: String, Sendable {
    case weightLoad
    case ditForward
  }

  public enum AnomalyKind: String, Sendable {
    case nan
    case inf
    case outOfRange
    case zeroLatent
  }

  public enum ErrorPhase: String, Sendable {
    case weightLoad
    case forward
    case recipeValidation
    case other
  }
}
```

Notes:
- `WeightComponent` is `dit` only today — encoder and decoder are TuberiaCatalog components instrumented at the Tuberia layer.
- `AnomalyPhase` is `weightLoad` (reserved — not currently emitted) or `ditForward` (the only active phase).
- `AnomalyKind` does not carry `.shapeMismatch` — shape mismatches throw, and the throw fires `errorThrown` instead.
- `ErrorPhase.weightApply` from the original draft was renamed to `weightLoad` and `forwardPass` → `forward` for consistency.

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

Identical to the original draft. The async-capture contract is what forced the hot-path coalescing question that was later mooted by the scope cut (see §11.4).

---

## 4. Injection points (as shipped)

### 4.1 `PixArtDiT` lock seam

`PixArtDiT.swift` declares `public final class PixArtDiT: Module, Backbone, @unchecked Sendable`. The seam matches the flux pattern:

```swift
import os.lock

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
```

`setTelemetry(_:)` is safe to call from any thread, at any point in the DiT's lifecycle. Pass `nil` to detach.

### 4.2 `PixArtRecipe` / `PixArtFP16Recipe` — sibling async-validate

`PipelineRecipe` requires sync `validate() throws`. Replacing the signature would break protocol conformance, so a sibling async method was added alongside instead — keeping the existing sync entrypoint untouched:

```swift
public func validate() throws { /* unchanged; satisfies PipelineRecipe */ }

public func validate(telemetry: (any PixArtTelemetryReporter)? = nil) async throws {
  // same checks, emits recipeValidated / recipeValidationFailed / errorThrown
}
```

Hosts that want telemetry call `try await recipe.validate(telemetry: reporter)` explicitly before `DiffusionPipeline.init`. Hosts that don't can keep calling the sync version — the recipe protocol is unchanged.

### 4.3 What is NOT a library-side seam

- `recipeSelected` — emitted by the host's engine router. Not a PixArt event in the shipped surface; if Vinetas wants to log it, it does so against its own taxonomy.
- `ditInitialized` — cut as redundant with `weightLoadComplete`'s `paramCount` payload.

---

## 5. Per-event emission spec (as shipped)

| Event | Emission site | Notes |
|---|---|---|
| `weightLoadComplete` | Exit of `PixArtDiT.apply(weights:)` | Carries `paramCount` (excluding `.scales`/`.biases` sidecars) and `durationSeconds`. The quantization detail from the original draft is implicit in the param count and is recoverable from the host's component descriptor. |
| `weightUnloadComplete` | `PixArtDiT.unload()` | Bare event, no payload. |
| `recipeValidated` | Final line of `PixArtRecipe.validate(telemetry:)` / `PixArtFP16Recipe.validate(telemetry:)` | Carries recipe name and a count of checks that passed (3 today). |
| `recipeValidationFailed` | Each `guard` failure inside `validate(telemetry:)` before throwing | Names the specific check (`encoder_caption_channels`, `encoder_text_length`, `decoder_latent_channels`). |
| `numericalAnomaly` | Exit of `PixArtDiT.forward(_:)`, only if the sampled output stat triggers an anomaly classifier | `phase: .ditForward`. `kind`: `nan` / `inf` / `outOfRange` / `zeroLatent`. Classifier lives at `PixArtDiT.anomalyKind(for:)`. Zero events on healthy output. |
| `errorThrown` | Paired with each `recipeValidationFailed` immediately before `throw` | `phase: .recipeValidation`. The other `ErrorPhase` cases are reserved for future surfaces (weight-load failures, forward-pass throws). |

### Hot-path discipline (as shipped)

The forward pass takes **zero events on the happy path**. One `currentTelemetry()` lookup at function entry, one sample at exit, and at most one `Task { await reporter.capture(event) }` dispatched if the anomaly classifier fires. This bypassed the coalescing problem the original draft (and the mission) spent significant design effort on — see §11.4.

```swift
public func forward(_ input: BackboneInput) throws -> MLXArray {
  let telemetry = currentTelemetry()  // ONE lock acquisition per forward
  // ... math (no emissions) ...
  if let telemetry {
    let outputStat = TuberiaTensorStat.sample(output)
    if let anomaly = anomalyKind(for: outputStat) {
      let event = PixArtTelemetryEvent.numericalAnomaly(
        phase: .ditForward, kind: anomaly, stat: outputStat)
      Task { await telemetry.capture(event) }
    }
  }
  return output
}
```

---

## 6. Adapter mapping (Vinetas host side)

Host integration is still pending (see §11.1). The shipped event set maps onto a much smaller phase taxonomy than the original draft anticipated:

| Event | Suggested phase string |
|---|---|
| `weightLoadComplete` | `pixart_weight_load_complete` |
| `weightUnloadComplete` | `pixart_weight_unload` |
| `recipeValidated` | `pixart_recipe_validated` |
| `recipeValidationFailed` | `pixart_recipe_fail_<check>` |
| `numericalAnomaly` | `pixart_anomaly_<kind>` |
| `errorThrown` | `pixart_error_<phase>` |

When the adapter lands in Vinetas, it should switch exhaustively over `PixArtTelemetryEvent` to catch new cases at compile time.

---

## 7. Tests (as shipped)

| Test file | Purpose |
|---|---|
| `PixArtTelemetryAnomalyTests.swift` | Inject a forward path that produces NaN; assert `numericalAnomaly(phase: .ditForward, kind: .nan)` fires with a populated stat. |
| `PixArtTelemetryLockContentionTests.swift` | Concurrent `setTelemetry` toggles + a running forward pass; TSan-clean. `test-tsan` Makefile target exercises this. |
| `PixArtTelemetryWeightApplyINT4Tests.swift` | Load int4-quantized weights through `MockReporter`. Assert `weightLoadComplete(component: .dit, paramCount: > 0)`. |
| `PixArtTelemetryWeightApplyFP16Tests.swift` | Load fp16 weights through `MockReporter`. Assert `weightLoadComplete(component: .dit, paramCount: > 0)`. |

`MockReporter.swift` is the shared test helper — actor-based, captures every event in order, supports per-test fresh instances (mandatory; see RECONCILIATION.md / mission brief §1.8 for why a shared fixture is unsafe).

Tests deliberately removed during the pare-down:
- `PixArtTelemetryForwardSequenceTests` — verified the 8-event forward sequence that no longer exists.
- `PixArtTelemetryVarianceDiscardTests` — verified the variance-channel event that no longer exists.
- `PixArtTelemetryNoopOverheadTests` — the overhead concern is moot at one Task per forward (and only on anomaly).

---

## 8. Out of scope

Still out of scope, unchanged from the original draft:
- Per-DiT-block events (28 blocks × 20 steps = 560 events/gen).
- Per-attention-head events.
- Internal kernel diagnostics (the `dequantized(...)` function internals).
- The 2D sinusoidal position-embedding computation (deterministic from grid dims).
- Training instrumentation.

Newly out of scope after the pare-down (see §11):
- Per-stage forward-pass events.
- Recipe-selection, DiT-init, and weight-apply-start lifecycle events.
- Micro-conditioning presence flag as a standalone event.
- The silu-workaround marker.

These can be re-added later if a real incident requires them — the surface is intentionally slim and grows only on demand.

---

## 9. Versioning

**Shipped:** v0.7.0 (minor bump from v0.6.0). Pin floor: `0.7.0`.
**SwiftTuberia floor:** `0.7.0` (for `TuberiaTensorStat`).
**Compatibility:** All additions; no breaking changes. Hosts that don't call `setTelemetry(_:)` or `validate(telemetry:)` see no behavior change.

---

## 10. Implementation status

All v0.7.0 surface items are landed on `main`:

- [x] `Sources/PixArtBackbone/Telemetry/PixArtTelemetryEvent.swift` (6-case slim form per §3.1)
- [x] `Sources/PixArtBackbone/Telemetry/PixArtTelemetryReporter.swift` (protocol + Noop)
- [x] `Sources/PixArtBackbone/Telemetry/README.md` (consumer guide)
- [x] `OSAllocatedUnfairLock<(any PixArtTelemetryReporter)?>` + `setTelemetry`/`currentTelemetry` on `PixArtDiT`
- [x] `weightLoadComplete` / `weightUnloadComplete` emissions in `apply(weights:)` / `unload()`
- [x] Single `numericalAnomaly` emission at `forward(_:)` exit gated by `anomalyKind(for:)`
- [x] `validate(telemetry:)` async sibling on `PixArtRecipe` and `PixArtFP16Recipe`
- [x] `recipeValidationFailed` + `errorThrown` paired with each `throw` in validation
- [x] 4 test suites (anomaly, lock contention, weight-apply INT4, weight-apply FP16)
- [x] `test-tsan` Makefile target
- [x] AGENTS.md Telemetry section linking the consumer guide and reconciliation note

Open / deferred (see §11):

- [ ] Vinetas host adapter (`PixArtTelemetryAdapter.swift` + `setTelemetry`/`validate(telemetry:)` wiring at `DiffusionPipeline.init`)
- [ ] Per-step correlation via `BackboneInput.stepIndex: Int?` (requires a SwiftTuberia PR; out of scope for this repo)
- [ ] Re-expansion plan if a real incident demands finer granularity

---

## 11. Scope cuts and why

The original draft of this document specified 18 event cases and ~8 emission points inside `forward(_:)`. The OPERATION STETHOSCOPE FURNACE mission implemented that surface, and CI passed with 159 tests. After mission completion, a follow-up pass (`b585120`) cut the surface to its current 6-event form before tagging v0.7.0. This section documents what was cut and why, so future expansion has a clear premise.

### 11.1 Vinetas host adapter — out of scope for v0.7.0

The original draft assumed a Vinetas-side `PixArtTelemetryAdapter` would land in lockstep with this library. It didn't. As of v0.7.0, no host actually calls `setTelemetry(...)` or `validate(telemetry:)`, so the surface is technically unobservable in production.

**Carry forward:** A Vinetas-side mission is the natural follow-up. Until that lands, the consumer guide at `Sources/PixArtBackbone/Telemetry/README.md` describes the integration pattern any host can follow.

### 11.2 Per-stage forward-pass events — cut as premature

The original draft specified `ditForwardStart`, `patchEmbedComplete`, `captionProjectionComplete`, `timestepEmbeddingComplete`, `siluWorkaroundExecuted`, `finalLayerComplete`, `varianceChannelsDiscarded`, and `ditForwardComplete` — 8 events per forward call, with 6 of those sampling `TuberiaTensorStat` (= 48 MLX reductions per step, ×20 steps per generation).

**Why cut:** Anomalies at any of those stages surface in the per-forward output stat. The original draft's own §1 noted this and added the events anyway, on the theory that localization mattered. After the mission shipped, the question shifted from "is this useful?" to "is this useful *now*, given there is no Vinetas adapter yet to consume them?" The answer was no — and the convention in `flux-2-swift-mlx/AGENTS.md §11` is "instrument boundaries, not internals" by default. The slim form follows that convention.

**Carry forward:** If a real incident demands per-stage localization (a NaN that appears only after `patchEmbed` but is clean at input, for example), re-introduce the specific stage event that would have caught it. Don't restore the full 8-event sequence speculatively.

### 11.3 Lifecycle events `recipeSelected`, `ditInitialized`, `weightApplyStart`, `microConditioningStatus` — cut as redundant or host-side

- `recipeSelected` belonged to the host's engine router, not this library.
- `ditInitialized` carried information (`hiddenSize`, `depth`, etc.) that is fully described by the recipe's configuration and is derivable by any host that has the recipe reference. No new bits of information per event.
- `weightApplyStart` was a "we are about to load" marker. The completion event with `durationSeconds` is more useful and bookends the same span.
- `microConditioningStatus` was a one-shot flag indicating whether `sizeEmbedder`/`arEmbedder` keys were present. This information is now part of the model's component descriptor and does not change at runtime per load.

**Carry forward:** If any host ever needs to log lifecycle boundaries explicitly, it can do so against its own taxonomy without this library producing the event.

### 11.4 The hot-path coalescing pattern — mooted by the pare-down

The mission ran into a real Swift-concurrency problem: `Backbone.forward(_:)` is sync `throws` but `PixArtTelemetryReporter.capture(_:)` is `async`. The original example code in §5 ("Hot-path discipline") shows `await telemetry.capture(...)` inside `forward(_:)` — which does not compile. The mission's solution was a stack-local `var pendingEvents: [PixArtTelemetryEvent] = []` queue plus a single `Task { for e in events { await telemetry.capture(e) } }` at the end of `forward(_:)` to preserve emission order without spawning 16 independent `Task`s (which have no ordering guarantee).

That pattern is now dead code. With the surface cut to one anomaly event per forward (and only on the unhappy path), there is nothing to coalesce — the forward pass either dispatches zero `Task`s or exactly one.

**Carry forward:** If the surface is ever re-expanded to multiple per-forward events, **bring back the coalesced single-Task pattern** — do not naively spawn one Task per emission. The mission's RECONCILIATION.md preserves the pattern as reusable scaffolding.

### 11.5 The Noop overhead gate — cut as moot

`PixArtTelemetryNoopOverheadTests` measured the wall-clock delta between `nil` reporter and `NoopPixArtTelemetryReporter` across 20 forward calls, with a ±1% target band. MLX lazy-evaluation variance dominated the measurement (the mission reported `delta = -0.60` — noise floor). The test passed but couldn't precisely measure sub-1% overhead.

After the pare-down, the cost of telemetry on a healthy forward pass is one lock acquisition and one stat sample. Both are constant-time. There is nothing meaningful for an overhead gate to measure.

**Carry forward:** If the surface is re-expanded, write the overhead test against the *reporter call itself* (sub-call instrumentation) rather than at the `forward()` boundary — the mission brief §1.11 details the methodology improvement.

### 11.6 `recipeSelected` correlation via `BackboneInput.stepIndex` — deferred to SwiftTuberia

The original draft assumed `BackboneInput.stepIndex: Int?` would be available so events could be correlated to a specific denoising step. SwiftTuberia 0.7.0's `BackboneInput` does not carry that field. The mission emitted `stepIndex: nil` everywhere, then the pare-down removed the field entirely.

**Carry forward:** If per-step correlation is needed, file a SwiftTuberia PR adding the field, then thread it through emissions here. Out of scope for pixart-swift-mlx alone.

### 11.7 Lessons for future expansion

If a real incident motivates expanding this surface:
- Start from the specific event that would have caught the incident — not the full 18-event taxonomy.
- Bring back the **pendingEvents + single-Task coalescing pattern** (do not spawn multiple Tasks per forward).
- Use **fresh `PixArtDiT` instances per telemetry test** — the mission discovered (the hard way) that a shared `BackboneFixture.dit` is contaminated by concurrent test suites running forward passes against it.
- Match the project's actual test framework (**swift-testing**, not XCTest). The original mission plan referenced XCTest APIs; agents adapted in flight.
- Write overhead tests **at the reporter call** (sub-call instrumentation), not at the `forward()` boundary, to avoid MLX lazy-eval variance.

These are codified in [`OPERATION_STETHOSCOPE_FURNACE_01_BRIEF.md`](OPERATION_STETHOSCOPE_FURNACE_01_BRIEF.md) §2 and §5.
