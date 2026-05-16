# AGENTS.md

This file provides comprehensive documentation for AI agents working with the pixart-swift-mlx codebase.

**Version**: 0.7.2-dev
**Purpose**: Guide AI agents working on pixart-swift-mlx
**Audience**: Claude Code, Gemini, and other AI development assistants

---

## Project Overview

pixart-swift-mlx is a model plugin for SwiftTuberia providing the PixArt-Sigma DiT (Diffusion Transformer) backbone. It contributes only the model-specific delta — the unique neural network architecture, weight key mapping, configuration, and pipeline recipe. All infrastructure comes from SwiftTuberia.

## Architecture

**Pipeline recipe**:
```
T5XXLEncoder (catalog) -> PixArtDiT (this repo) -> SDXLVAEDecoder (catalog) -> ImageRenderer (catalog)
                              ^
                       DPMSolver++ (catalog)
```

**This repo provides**:
- PixArt-Sigma DiT backbone (~28 blocks, 1152 hidden dim, ~600M params)
- Weight key mapping (~200 keys, PyTorch -> MLX safetensors)
- Acervo component descriptors for model registration
- LoRA target layer declarations
- Pipeline recipe assembly
- Weight conversion scripts

**~400 lines of model-specific code total.**

## Dependencies

- `SwiftTuberia` (Tuberia + TuberiaCatalog) — pipeline protocols + shared components
- `SwiftAcervo` (transitive via SwiftTuberia) — model registry

## Platform Requirements

- iOS 26.0+, macOS 26.0+ exclusively
- Swift 6.2+, Xcode 26+
- Apple Silicon only (M1+)
- ~2 GB total (int4 quantized), iPad-viable

## Build and Test

This project uses a Makefile. Available targets:

```bash
make resolve      # Resolve SPM dependencies
make build        # Debug build
make test         # Run Swift unit tests
make test-python  # Run Python conversion script tests
make test-all     # Run all tests (Swift + Python)
make lint         # Format Swift sources with swift-format
make clean        # Remove build artifacts and DerivedData
make help         # Show all targets
```

## Critical Rules for AI Agents

1. NEVER commit directly to `main` — use `development` branch
2. ONLY support iOS 26.0+ and macOS 26.0+ (NEVER add code for older platforms)
3. ALWAYS run `make lint` before committing
4. ALWAYS read files before editing
5. NEVER create files unless necessary
6. Follow agent-specific instructions — see [CLAUDE.md](CLAUDE.md) or [GEMINI.md](GEMINI.md)

## App Group configuration (required)

This package depends on [SwiftAcervo](https://github.com/intrusive-memory/SwiftAcervo) for shared model storage. SwiftAcervo v0.10.0 resolves its App Group ID in this order: `ACERVO_APP_GROUP_ID` env var → `com.apple.security.application-groups` entitlement (macOS only) → `fatalError`. There is **no silent fallback**.

- **Signed UI apps (macOS / iOS)**: declare `com.apple.security.application-groups` with `group.intrusive-memory.models` in your `.entitlements` file. iOS apps additionally need `ACERVO_APP_GROUP_ID=group.intrusive-memory.models` in the launch environment.
- **Scripts, CI jobs, test runners**: export `ACERVO_APP_GROUP_ID=group.intrusive-memory.models` in the shell or job environment. The standard place is `~/.zprofile`:

    ```sh
    export ACERVO_APP_GROUP_ID=group.intrusive-memory.models
    ```

Without this, `Acervo.sharedModelsDirectory` traps with `fatalError`. See [SwiftAcervo's USAGE.md](https://github.com/intrusive-memory/SwiftAcervo/blob/main/USAGE.md) for full details.

## Telemetry

PixArtBackbone ships a **dual-seam** telemetry surface following the cross-library pattern documented in `SwiftVinetas/docs/INSTRUMENTATION_PATTERN.md`. Both seams expose the same `PixArtTelemetryEvent` enum (6 cases: `weightLoadComplete`, `weightUnloadComplete`, `recipeValidated`, `recipeValidationFailed`, `numericalAnomaly`, `errorThrown`); the difference is _where_ the reporter is installed and who wins.

**Instance-bound seam** (`PixArtDiT.setTelemetry(_:)`): scoped to one `PixArtDiT` instance. Ideal for unit tests where you need to assert events from a specific instance in isolation. The instance reporter always takes precedence over the process-wide reporter when both are set.

**Process-wide seam** (`PixArtTelemetry.setReporter(_:)`): a lock-guarded static slot on `public enum PixArtTelemetry`. CLI hosts and other process-wide consumers call this once at startup and receive events from every `PixArtDiT` emission site — even DiT instances that are lazily constructed deep inside another library's actor (e.g., `PixArtEngine` in SwiftVinetas). `PixArtRecipe.validate(telemetry:)` and `PixArtFP16Recipe.validate(telemetry:)` also fall back to `PixArtTelemetry.current` when no explicit `telemetry` argument is supplied, so recipe events reach the process-wide reporter automatically.

Every emission site in `PixArtDiT` resolves through the `effectiveReporter` computed property (`instanceReporter ?? PixArtTelemetry.current`). Emission code never calls either seam directly.

### Process-wide install (CLI host)

```swift
// At process startup (CLITelemetryBootstrap.enable)
PixArtTelemetry.setReporter(pixartAdapter)  // pixartAdapter: PixArtTelemetryCLIAdapter

// During teardown (CLITelemetryBootstrap.finish)
PixArtTelemetry.setReporter(nil)
```

### Per-instance install (unit tests)

```swift
let dit = PixArtDiT(configuration: config)
dit.setTelemetry(MockPixArtReporter())  // instance reporter; wins over process-wide when both set
// ... exercise the instance ...
dit.setTelemetry(nil)  // clean up
```

### Adding a new event case

1. Add the case (and any nested enum values) to `PixArtTelemetryEvent` in `Sources/PixArtBackbone/Telemetry/PixArtTelemetryEvent.swift`. Every case and every associated-value type must be `Sendable`. Do NOT add a `runID` field — run identifiers belong at the host/sink layer.
2. Emit via `await effectiveReporter?.capture(.newCase(...))` at the single canonical emission site inside `PixArtDiT` (or `PixArtRecipe` / `PixArtFP16Recipe` for recipe-phase events).
3. Update the host's `PixArtEventEncoding.swift` (Encodable shim in SwiftVinetas) to handle the new case exhaustively.
4. Add a test in `PixArtProcessWideTelemetryTests.swift` or the existing per-instance test file, covering both the instance-only and process-wide-only scenarios.

### When to add a new event

- Instrument boundaries, not internals. One event per phase entry/exit; not per-block or per-attention-head.
- Each emission site must be the single canonical place for that event — do not emit the same logical event from multiple call sites.
- All associated values must be `Sendable`. Struct payloads are preferred over class references.
- Do not add a `runID` to the event cases themselves; the host attaches run context at the sink layer.
- Prefer start/complete pairs (`operationStart` / `operationComplete`) for durational operations so the host can measure elapsed time.
- Keep the hot path quiet: fire events at boundaries (weight load, recipe validation, anomaly detection), never inside tight inner loops.

### Reference

- [Sources/PixArtBackbone/Telemetry/README.md](Sources/PixArtBackbone/Telemetry/README.md) — Consumer guide: events, install pattern, example adapter, alerting recommendations
- [Sources/PixArtBackbone/Telemetry/PixArtTelemetry.swift](Sources/PixArtBackbone/Telemetry/PixArtTelemetry.swift) — Process-wide seam implementation
- [Sources/PixArtBackbone/Telemetry/PixArtTelemetryEvent.swift](Sources/PixArtBackbone/Telemetry/PixArtTelemetryEvent.swift) — 6-case event enum
- [Tests/PixArtBackboneTests/PixArtProcessWideTelemetryTests.swift](Tests/PixArtBackboneTests/PixArtProcessWideTelemetryTests.swift) — Dual-seam unit tests
- [docs/complete/stethoscope-furnace-01/REQUIREMENTS-instrumentation.md](docs/complete/stethoscope-furnace-01/REQUIREMENTS-instrumentation.md) — Implementation requirements (shipped form)
- [docs/complete/stethoscope-furnace-01/RECONCILIATION.md](docs/complete/stethoscope-furnace-01/RECONCILIATION.md) — What OPERATION STETHOSCOPE FURNACE shipped vs. the original brief
- `SwiftVinetas/docs/INSTRUMENTATION_PATTERN.md` — Canonical cross-library dual-seam pattern

## Documentation Index

- [AGENTS.md](AGENTS.md) — Universal agent documentation (this file)
- [CLAUDE.md](CLAUDE.md) — Claude-specific instructions
- [GEMINI.md](GEMINI.md) — Gemini-specific instructions
- [REQUIREMENTS.md](REQUIREMENTS.md) — Full specification
- [ARCHITECTURE.md](ARCHITECTURE.md) — Detailed architecture notes
- [README.md](README.md) — User-facing documentation
- [Sources/PixArtBackbone/Telemetry/README.md](Sources/PixArtBackbone/Telemetry/README.md) — Telemetry consumer guide
