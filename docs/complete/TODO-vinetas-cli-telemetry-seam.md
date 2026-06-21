# RESOLVED — Process-wide Telemetry Seam for CLI Hosts

**RESOLVED in commit**: see `git log --oneline -1` after the telemetry seam commit on `development`.
**Resolution**: Added `PixArtTelemetry` enum (option 1 from §Background) in
`Sources/PixArtBackbone/Telemetry/PixArtTelemetry.swift`. All `PixArtDiT` emission
sites now fall back to `PixArtTelemetry.current` via `effectiveReporter`. Recipes
(`PixArtRecipe`, `PixArtFP16Recipe`) also fall back in `validate(telemetry:)`.
Tests added in `Tests/PixArtBackboneTests/PixArtProcessWideTelemetryTests.swift`.

---

# TODO — Process-wide Telemetry Seam for CLI Hosts (original)

**Filed by**: SwiftVinetas — OPERATION WIRETAP DARKROOM (2026-05-15)
**Issue surfaced in**: `SwiftVinetas/Sources/VinetasCLICore/Telemetry/CLITelemetryBootstrap.swift`

## Background

The SwiftVinetas CLI host (`vinetas generate --telemetry --model pixart-sigma …`) wants to install a `PixArtTelemetryReporter` adapter so that PixArt events are interleaved into the unified JSONL trace alongside Vinetas, Flux2, Tuberia, and Acervo events.

`PixArtDiT.setTelemetry(_:)` at `Sources/PixArtBackbone/PixArtDiT.swift:40` is **instance-bound** — the DiT instance lives privately inside `SwiftVinetas/Sources/SwiftVinetas/Engine/PixArtEngine.swift` and is instantiated lazily during `loadModel(_:)`. The CLI bootstrap has no reference to that instance, so **no PixArt events are reachable from the CLI today**.

## What would unblock the CLI

Any one of the following would let `CLITelemetryBootstrap.enable(...)` capture PixArt telemetry:

1. **Process-wide reporter shared by all `PixArtDiT` instances.** Add `public static var PixArtTelemetry.shared: (any PixArtTelemetryReporter)?` (or `PixArtDiT.setTelemetryForAllInstances(_:)`), have every emission site read it lazily. Simplest from the CLI's POV; matches `SwiftAcervo.AcervoManager.shared`'s pattern.
2. **Per-instance install API exposed through SwiftVinetas** — would require `PixArtEngine` to grow `public func setPixArtDepReporter(_:)`. Cleaner encapsulation but every host has to do per-instance wiring.

Recommendation: option (1).

## Out of scope for this TODO

- The existing instance-bound `setTelemetry(_:)` is the right primitive — keep it. The ask is an additive process-wide layer.
- Don't change the event enum or the reporter protocol — SwiftVinetas already ships an adapter (`PixArtTelemetryCLIAdapter`) conforming to the existing protocol.

## What's already shipped on the SwiftVinetas side

- `Sources/VinetasCLICore/Telemetry/PixArtEventEncoding.swift` — Encodable shim for all 6 cases of `PixArtTelemetryEvent`.
- `Sources/VinetasCLICore/Telemetry/PixArtTelemetryCLIAdapter.swift` — conforms to `PixArtTelemetryReporter`, writes with `kind: "pixart"`.

When the process-wide seam lands, the CLI just calls `<NewSeam>.setReporter(adapter)` in one line and the integration test's `kinds ⊇ {pixart}` assertion will start passing.
