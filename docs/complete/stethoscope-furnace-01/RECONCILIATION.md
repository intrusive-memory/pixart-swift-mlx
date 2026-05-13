# Reconciliation: brief vs. what actually shipped

**Added:** 2026-05-13
**Scope:** Reconciles the `OPERATION_STETHOSCOPE_FURNACE_01_BRIEF.md` description of the telemetry surface against the code that exists on `main` today.

## TL;DR

The brief in this folder describes the surface as it existed at mission completion (final commit `3fe16a4` on `mission/stethoscope-furnace/01`). After the PR merged as `503acc6`, a follow-up commit **`b585120` ("updating testing strategery", 2026-05-13)** slashed the surface significantly. The brief is preserved as-written for historical accuracy, but readers should treat the file inventory in §6 and several discoveries in §1 as **describing an intermediate state, not the current code**.

## What was removed in `b585120`

`b585120` was a `-1398 / +442` change touching the telemetry surface, the DiT, both recipes, and the test suite. The intent was to scope the surface down to "boundary-only" instrumentation per the convention in `flux-2-swift-mlx/AGENTS.md §11`.

### Event cases removed

The `PixArtTelemetryEvent` enum dropped roughly 12 of its 18 cases, leaving 6:

| Status | Case |
|---|---|
| ✅ kept | `weightLoadComplete` |
| ✅ kept | `weightUnloadComplete` |
| ✅ kept | `recipeValidated` |
| ✅ kept | `recipeValidationFailed` |
| ✅ kept | `numericalAnomaly(phase, kind, stat)` |
| ✅ kept | `errorThrown(phase, errorDescription)` |
| ❌ removed | `ditForwardStart` |
| ❌ removed | `ditForwardComplete` |
| ❌ removed | `patchEmbedComplete` |
| ❌ removed | per-block / per-stage events |
| ❌ removed | `varianceDiscard` |
| ❌ removed | `microConditioningStatus` |
| ❌ removed | weight-apply counter events |

### Hot-path pattern removed

The `pendingEvents: [PixArtTelemetryEvent]` stack-local queue + single coalesced `Task` dispatch at the end of `forward(_:)` — celebrated in the brief's §1.4, §1.5, and §5 — **no longer exists**. The current `forward(_:)` emits zero events on the happy path and dispatches at most one `Task { await ... }` if the output sample shows an anomaly. The coalescing problem the brief solved no longer applies because the emission count went from 16 to ≤1 per forward call.

### Tests removed

Two test files were deleted entirely:

- `Tests/PixArtBackboneTests/PixArtTelemetryForwardSequenceTests.swift` — verified event ordering across the forward pass
- `Tests/PixArtBackboneTests/PixArtTelemetryVarianceDiscardTests.swift` — verified the `varianceDiscard` event

Remaining telemetry tests:

- `PixArtTelemetryAnomalyTests.swift`
- `PixArtTelemetryLockContentionTests.swift`
- `PixArtTelemetryWeightApplyFP16Tests.swift`
- `PixArtTelemetryWeightApplyINT4Tests.swift`

The Sortie 9 Noop overhead gate (the test the brief flagged as "imprecise") also no longer exists as a standalone file — the overhead-gate concern is moot at one Task per forward.

## What still matches the brief

- The `PixArtTelemetryReporter` protocol shape (`func capture(_ event: PixArtTelemetryEvent) async`) and `NoopPixArtTelemetryReporter`.
- `PixArtDiT.setTelemetry(_:)` + `OSAllocatedUnfairLock` seam.
- The `validate(telemetry:)` async sibling pattern on both `PixArtRecipe` and `PixArtFP16Recipe` (Sortie 4's protocol-conformance call).
- SwiftTuberia 0.7.0 pin (Sortie 0).
- The fresh-`PixArtDiT`-per-test pattern (Sortie 7a's empirical discovery), still in use in the surviving telemetry test files.

## What the brief got right that's worth carrying forward

The §2 process discoveries are still valid lessons regardless of the post-mission scope cut:

- Plans that modify a public method must verify protocol conformance is preserved (§2.4 → Sortie 4).
- Spec example code must compile against real protocol shapes before the plan ships (§1.4, §1.5).
- Multiple independent `Task`s have no ordering guarantee at the receiving actor (§1.5) — still applies anywhere a future contributor adds back parallel emissions.
- Performance gates at <1% precision require interleaved measurements or sub-call instrumentation (§1.11) — relevant if anyone reintroduces an overhead gate.

## Open Decision #2 status

The brief's §3.2 ("Vinetas host adapter wiring — out of scope but where does it live?") is **still open**. As of this reconciliation date, no host explicitly calls `setTelemetry(...)` or `validate(telemetry:)`. The surface remains unobservable in production until a host wires it in. The consumer-facing guide at `Sources/PixArtBackbone/Telemetry/README.md` (added the same day as this note) documents how a host should do it.

## Why preserve the brief instead of editing it

Mission briefs are historical artifacts of what was learned during a specific iteration. Rewriting §6 to match the current code would erase evidence of the Sortie 5b coalescing pattern, the variance/anomaly test design, and the overhead-gate methodology debate — all useful context if a future mission needs to re-expand the surface. This reconciliation note is the bridge.

## Pointers

- Slim surface commit: `b585120 "updating testing strategery"` (2026-05-13)
- Mission merge: `503acc6 feat(telemetry): PixArt instrumentation surface (OPERATION STETHOSCOPE FURNACE) (#19)`
- Release: `v0.7.0` (squash `53bcd2d`)
- Current public surface: `Sources/PixArtBackbone/Telemetry/`
- Consumer guide: [`Sources/PixArtBackbone/Telemetry/README.md`](../../../Sources/PixArtBackbone/Telemetry/README.md)
- Updated requirements (post-pare-down): [`REQUIREMENTS-instrumentation.md`](REQUIREMENTS-instrumentation.md) — §11 catalogs every scope cut
