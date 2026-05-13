# PixArtBackbone Telemetry

A slim, boundary-only telemetry surface for diagnosing weight-load, recipe-validation, and numerical-anomaly problems in the PixArt-Sigma DiT backbone. Hosts conform a reporter, install it on the DiT, and pass it to recipe validation. The library emits when something happens that the host should know about; it stays silent on the happy path.

## What you get

Six event cases on `PixArtTelemetryEvent`:

| Event | When it fires | Use it to |
|---|---|---|
| `weightLoadComplete(component, paramCount, durationSeconds)` | After `PixArtDiT.apply(weights:)` succeeds | Track load time, confirm param count matches expectations |
| `weightUnloadComplete` | After `PixArtDiT.unload()` | Confirm release for memory accounting |
| `recipeValidated(name, checksPassed)` | When `recipe.validate(telemetry:)` passes all checks | Confirm shape contracts before pipeline init |
| `recipeValidationFailed(name, check, reason)` | A specific shape check failed | Alert: misconfigured pipeline; recipe will also throw |
| `numericalAnomaly(phase, kind, stat)` | DiT output sampled at forward exit is NaN / Inf / out-of-range / zero-latent | The headline diagnostic. Backbone produced bad output; investigate weights, conditioning, or upstream noise |
| `errorThrown(phase, errorDescription)` | Recipe validation throws (paired with `recipeValidationFailed`) | Single-channel sink for all thrown errors |

`AnomalyPhase` is currently always `.ditForward` (or `.weightLoad` if we extend later). `AnomalyKind` is `.nan`, `.inf`, `.outOfRange`, or `.zeroLatent`. `stat` is a `TuberiaTensorStat` from SwiftTuberia carrying min/max/mean/std/hasNaN/hasInf.

## What you do NOT get (deliberately)

- **No per-step, per-block, or per-attention-head events.** A `numericalAnomaly` points at the region; finer instrumentation is added in a follow-up iteration only after a real failure demands it.
- **No happy-path forward-pass events.** Zero events fire when `forward(_:)` produces healthy output. This keeps the hot path quiet and the noise floor at zero.
- **No `stepIndex` correlation.** `BackboneInput` doesn't carry one in SwiftTuberia 0.7.0. Hosts that need per-step correlation should reconstruct it from their denoise-loop wrapper.

If you need finer granularity, file an issue describing the actual failure you're trying to localize — the surface is intentionally slim and grows only on demand.

## How to consume it

### 1. Conform a reporter

```swift
import PixArtBackbone
import Tuberia
import os

actor MyPixArtReporter: PixArtTelemetryReporter {
  private let log = Logger(subsystem: "com.example.app", category: "pixart")

  func capture(_ event: PixArtTelemetryEvent) async {
    switch event {
    case .weightLoadComplete(let component, let paramCount, let durationSeconds):
      log.info("PixArt \(component.rawValue) loaded: \(paramCount) params in \(durationSeconds, format: .fixed(precision: 2))s")

    case .weightUnloadComplete:
      log.info("PixArt weights unloaded")

    case .recipeValidated(let name, let checksPassed):
      log.debug("\(name) passed \(checksPassed) checks")

    case .recipeValidationFailed(let name, let check, let reason):
      log.error("\(name) failed check \(check): \(reason)")

    case .numericalAnomaly(let phase, let kind, let stat):
      log.fault("PixArt anomaly in \(phase.rawValue): \(kind.rawValue) — min=\(stat.min) max=\(stat.max) mean=\(stat.mean) std=\(stat.std)")

    case .errorThrown(let phase, let errorDescription):
      log.error("PixArt error in \(phase.rawValue): \(errorDescription)")
    }
  }
}
```

A reporter must be `Sendable`. Conform an actor or a value type with `@unchecked Sendable` if you need shared mutable state.

### 2. Install on the DiT

```swift
let dit = try PixArtDiT(configuration: backboneConfig)
dit.setTelemetry(MyPixArtReporter())
// pass nil to detach: dit.setTelemetry(nil)
```

`setTelemetry(_:)` is lock-guarded and safe to call from any thread, at any point in the DiT's lifecycle. Install before `apply(weights:)` if you want load timing.

### 3. Pass to recipe validation

```swift
let recipe = PixArtRecipe(...)  // or PixArtFP16Recipe
try await recipe.validate(telemetry: reporter)
let pipeline = try DiffusionPipeline(recipe: recipe, ...)
```

The `validate(telemetry:)` async sibling lives next to the protocol-required sync `validate()`. The sync version still works for callers that haven't wired telemetry yet — it just won't emit events.

### 4. Default to noop in production wiring

```swift
let reporter: any PixArtTelemetryReporter = isDebugBuild
  ? MyPixArtReporter()
  : NoopPixArtTelemetryReporter()
```

`NoopPixArtTelemetryReporter` swallows all events and has effectively zero cost — the test suite gates it as a regression floor.

## Recommended alerting

If you only wire one signal: alert on `numericalAnomaly` with `kind: .nan` or `.inf`. That's the "the backbone produced unusable output" smoke alarm and almost always indicates a real bug (bad weights, wrong dtype path, upstream conditioning corruption).

Treat `.outOfRange` and `.zeroLatent` as warnings — they can occur briefly during normal denoising but a sustained stream of them across multiple forward passes is a problem.

`recipeValidationFailed` always indicates a misconfigured pipeline; the recipe will also throw, so the event is mostly useful for centralized logging.

## Concurrency model

`PixArtTelemetryReporter.capture(_:)` is `async`. The DiT's hot path (`forward(_:)`) is sync `throws`, so emissions are dispatched as `Task { await reporter.capture(event) }` from the call sites. Events are not guaranteed to arrive at the reporter in any particular order relative to each other, and the forward pass does not wait for them. Don't build correctness-critical logic on telemetry ordering.

Recipe validation runs in an `async` context, so its events are ordered sequentially within a single validation call.

## Reference

- Public surface: `PixArtTelemetryEvent.swift`, `PixArtTelemetryReporter.swift`
- Emission sites: `PixArtDiT.swift` (`forward`, `apply(weights:)`, `unload`), `PixArtRecipe.swift`, `PixArtFP16Recipe.swift`
- Mission history: `docs/complete/stethoscope-furnace-01/` (see `RECONCILIATION.md` for what shipped vs what the brief described)
