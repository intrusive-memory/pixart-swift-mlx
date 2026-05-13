---
feature_name: OPERATION STETHOSCOPE FURNACE
starting_point_commit: 97ef40f25a98ed8744d3a910822d536960277533
mission_branch: mission/stethoscope-furnace/01
iteration: 1
state: completed
---

# EXECUTION_PLAN.md — pixart-swift-mlx Instrumentation

Source: `REQUIREMENTS-instrumentation.md`
Target branch: `mission/stethoscope-furnace/01`
Release: next minor version (additive only) after merge.

---

## Terminology

> **Mission** — A definable, testable scope of work. Defines scope, acceptance criteria, and dependency structure.

> **Sortie** — An atomic, testable unit of work executed by a single autonomous AI agent in one dispatch. One aircraft, one mission, one return.

> **Work Unit** — A grouping of sorties (package, component, phase).

---

## Mission Scope

Add a `PixArtTelemetryEvent` / `PixArtTelemetryReporter` surface to `PixArtBackbone` and wire emission sites inside `PixArtDiT.forward(_:)`, `PixArtDiT.apply(weights:)`, `PixArtDiT.unload()`, and the two recipe `validate()` methods. Adapter wiring on the Vinetas side is out of scope (REQUIREMENTS §6). Ship with a minor version bump.

**Critical constraints (baked into the plan):**

1. The hot-path `ditForwardComplete` emission fires **once per scheduler step**, never per DiT block. All hot-path emissions cache `currentTelemetry()` exactly once at function entry and use `@autoclosure`-style guarded sampling so `TuberiaTensorStat.sample(...)` never executes when the reporter is nil.
2. FP16 quant cast-site events (the `dtypeBoundary` family in user framing) are clustered into a single dedicated sortie covering: the INT4→FP16 dequant boundary at `apply(weights:)`, the `microConditioningStatus` weight-key scan, and the 8→4 channel `varianceChannelsDiscarded` cast at `:175`. These are the project's "communication error" diagnostic — wrong slicing or dtype confusion here produces silent corruption.
3. The baseline overhead test (20 forward calls with Noop reporter, wall-clock median delta) is its own sortie, gated on every other sortie being green.

**Build commands (per project CLAUDE.md, AGENTS.md, Makefile):**

- `make build` / `make test` — always prefer over raw `swift build`/`swift test`.
- XcodeBuildMCP `swift_package_build` / `swift_package_test` for local validation.
- **Never** `swift build` / `swift test`.

**Dependency floor:** `SwiftTuberia ≥ 0.7.0` (for `TuberiaTensorStat`). Must be released and pinned before Sortie 1 dispatches.

---

## Work Units

| Work Unit | Directory | Sorties | Layer | Dependencies |
|-----------|-----------|---------|-------|--------------|
| PixArtTelemetry | `Sources/PixArtBackbone/Telemetry/` + `Sources/PixArtBackbone/` + `Tests/PixArtBackboneTests/` | 10 | 0 | SwiftTuberia ≥ 0.7.0 released |

Single work unit (whole repo is one Swift package). Sorties layered internally — see `## Sortie Dependency Graph` below.

---

## Sortie Dependency Graph

```
Sortie 1 (types) ────► Sortie 2 (PixArtDiT seam) ────► Sortie 3 (weight + cast-site events: dtypeBoundary cluster)
                                                  └──► Sortie 4 (recipe validate seam)
                                                  └──► Sortie 5 (hot-path forward emissions, non-step-boundary)
                                                  └──► Sortie 6 (ditForwardComplete + step-boundary hot path) [depends on 5]

Sortie 3, 4, 5 can run in parallel after Sortie 2 (different files, different functions).
Sortie 6 depends on Sortie 5 (both edit `forward(_:)` in PixArtDiT.swift).

Sorties 7a, 7b (correctness tests) depend on Sorties 3, 5, 6.
Sortie 8 (lock-contention) depends on Sorties 3, 5, 6.
Sortie 9 (baseline overhead, 20-step) depends on all prior sorties green.
```

Layers:

- Layer 0: Sortie 1
- Layer 1: Sortie 2
- Layer 2: Sorties 3, 4, 5 (parallelizable)
- Layer 3: Sortie 6 (depends on 5, file conflict)
- Layer 4: Sorties 7a, 7b, 8 (parallelizable)
- Layer 5: Sortie 9 (final gate)

---

### Sortie 1: Public telemetry types

**Priority**: 9.5 — Foundation. Blocks all 9 downstream sorties. Foundation_score=1, dependency_depth=9, risk=2 (new public types, must match REQUIREMENTS §3.1 verbatim), complexity=1. Score: 9*3 + 1*2 + 2*1 + 1*0.5 = 31.5. **Highest priority.**

**Estimated turns**: 12 — Right-sized.

**Entry criteria**:
- [ ] First sortie — no prerequisites.
- [ ] `Package.swift` already imports `SwiftTuberia ≥ 0.7.0` (pin floor bump landed). Verify: `grep -A2 'SwiftTuberia' Package.swift` shows `from: "0.7.0"` or higher.

**Tasks**:
1. Create `Sources/PixArtBackbone/Telemetry/PixArtTelemetryEvent.swift` matching REQUIREMENTS §3.1 verbatim: the `PixArtTelemetryEvent` enum with all cases (recipe lifecycle, DiT init, weight apply, forward pass, anomaly, error), plus nested enums `PixArtQuantization`, `AnomalyKind`, `ErrorPhase`.
2. Create `Sources/PixArtBackbone/Telemetry/PixArtTelemetryReporter.swift` with the `PixArtTelemetryReporter` async protocol and `NoopPixArtTelemetryReporter` struct per §3.2.
3. Import `Tuberia` and use `TuberiaTensorStat` directly — do NOT redefine.
4. Mark every type `public` and `Sendable`. Use `@preconcurrency import MLX`.

**Exit criteria**:
- [ ] `make build` succeeds (zero warnings ideally; zero errors mandatory).
- [ ] `test -f Sources/PixArtBackbone/Telemetry/PixArtTelemetryEvent.swift`
- [ ] `test -f Sources/PixArtBackbone/Telemetry/PixArtTelemetryReporter.swift`
- [ ] `grep -n 'public enum PixArtTelemetryEvent' Sources/PixArtBackbone/Telemetry/PixArtTelemetryEvent.swift` matches.
- [ ] `grep -n 'public enum PixArtQuantization' Sources/PixArtBackbone/Telemetry/PixArtTelemetryEvent.swift` matches.
- [ ] `grep -n 'public protocol PixArtTelemetryReporter' Sources/PixArtBackbone/Telemetry/PixArtTelemetryReporter.swift` matches.
- [ ] `grep -n 'public struct NoopPixArtTelemetryReporter' Sources/PixArtBackbone/Telemetry/PixArtTelemetryReporter.swift` matches.
- [ ] `make test` green (existing tests unaffected).

**Agent allocation**: Supervising agent (has build/test step).

---

### Sortie 2: `PixArtDiT` telemetry seam (lock + setter + getter)

**Priority**: 8.0 — Establishes the seam consumed by Sorties 3, 5, 6. Foundation_score=1, dependency_depth=4, risk=2 (concurrency primitive), complexity=0.5. Score: 4*3 + 1*2 + 2*1 + 0.5*0.5 = 16.25.

**Estimated turns**: 7 — Right-sized (small but cannot be merged with Sortie 3 because Sorties 4 and 5 must execute against the same seam in parallel with 3).

**Entry criteria**:
- [ ] Sortie 1 exit criteria met (types compile).

**Tasks**:
1. In `Sources/PixArtBackbone/PixArtDiT.swift`, add `import os.lock` if absent.
2. Add `private let _telemetryLock = OSAllocatedUnfairLock<(any PixArtTelemetryReporter)?>(initialState: nil)` to the `PixArtDiT` class.
3. Add `public func setTelemetry(_ reporter: (any PixArtTelemetryReporter)?)` that writes under `_telemetryLock`.
4. Add `fileprivate func currentTelemetry() -> (any PixArtTelemetryReporter)?` that reads under `_telemetryLock`.
5. Do NOT yet emit any events — wiring follows in Sorties 3, 5, 6.

**Exit criteria**:
- [ ] `make build` succeeds.
- [ ] `grep -n '_telemetryLock' Sources/PixArtBackbone/PixArtDiT.swift` matches.
- [ ] `grep -n 'public func setTelemetry' Sources/PixArtBackbone/PixArtDiT.swift` matches.
- [ ] `grep -n 'fileprivate func currentTelemetry' Sources/PixArtBackbone/PixArtDiT.swift` matches.
- [ ] `grep -n 'import os.lock' Sources/PixArtBackbone/PixArtDiT.swift` matches.
- [ ] `make test` green.

**Agent allocation**: Supervising agent.

---

### Sortie 3: Weight-apply + dtypeBoundary cast-site events (THE COMMUNICATION ERROR DIAGNOSTIC)

**Priority**: 9.0 — High risk (silent corruption potential), foundational for the project's headline diagnostic. Foundation_score=1, dependency_depth=3, risk=3 (dtype handling), complexity=2.5. Score: 3*3 + 1*2 + 3*1 + 2.5*0.5 = 15.25.

**Estimated turns**: 28 — Right-sized but at the upper end. If the agent runs long, see auto-split plan in Open Questions §Q3.1.

**Sergeant note**: This sortie owns every FP16 quant cast site in `PixArtDiT`. These events are the project's "communication error" diagnostic — `dtypeBoundary` in user framing. Any silent dtype/shape mismatch at these boundaries produces wrong-looking output that is hard to forensically reconstruct after the fact. Every cast site must fire an event.

**Entry criteria**:
- [ ] Sortie 2 exit criteria met.

**Tasks**:
1. In `PixArtDiT.apply(weights:)` (`PixArtDiT.swift:189`):
   - Cache `let telemetry = currentTelemetry()` at function entry.
   - Capture `let start = Date()` at function entry.
   - **Before the load loop**: scan `weights.parameters` keys for `.scales`/`.biases` sidecars. If any sidecar present → `quantization = .int4`; if zero sidecars and any `.weight` value has `dtype == .float16` → `quantization = .fp16`; else `quantization = .unknown`. Compute `weightKeyCount = weights.parameters.count`. Emit `.weightApplyStart(quantization:, weightKeyCount:)`.
   - **Inside the second pass load loop**: maintain three counters:
     - `dequantizedKeys` — incremented when the int4 path executes (line 234 `dequantized(...)`).
     - `passThroughKeys` — incremented when the else branch executes (line 239 `params[key] = tensor`).
     - `scalesBiasesSkipped` — incremented when `key.hasSuffix(".scales") || key.hasSuffix(".biases")` (line 221 continue).
   - **After the load loop**: scan `weights.parameters.keys` for any key containing `"sizeEmbedder"` or `"arEmbedder"`. Emit `.microConditioningStatus(present: sizeEmbedderFound || arEmbedderFound, sizeEmbedderFound:, arEmbedderFound:)` exactly once.
   - **At function exit (`PixArtDiT.swift:247`, just before `self.isLoaded = true`)**: compute `durationSeconds = Date().timeIntervalSince(start)`, `sizeMB = Double(estimatedMemoryBytes) / 1_048_576.0`. Emit `.weightApplyComplete(quantization:, totalKeys: weightKeyCount, dequantizedKeys:, passThroughKeys:, scalesBiasesSkipped:, sizeMB:, durationSeconds:)`.
2. In `PixArtDiT.unload()` (`PixArtDiT.swift:250`): capture `restoredKeyCount = weights?.parameters.count ?? 0` BEFORE setting `weights = nil`, then emit `.weightUnload(restoredKeyCount:)` before the function returns.
3. In `PixArtDiT.forward(_:)` at the 8→4 channel slice (`PixArtDiT.swift:175`, `output = output[..., ..., ..., 0..<4]`):
   - This sortie ONLY adds the variance-discard emission. Sortie 5 establishes the `let telemetry = currentTelemetry()` cache at function entry; if Sortie 5 has already landed, reuse it. If Sortie 5 hasn't landed yet (parallel execution), this sortie must add its own `let telemetry = currentTelemetry()` at function entry — and Sortie 5's agent must NOT add a duplicate. Coordination owner: supervising agent during dispatch.
   - Inside `if let telemetry { ... }`:
     - `let beforeStat = TuberiaTensorStat.sample(output)` (8-channel).
     - Perform the slice (existing code).
     - `let afterStat = TuberiaTensorStat.sample(output)` (4-channel).
     - `await telemetry.capture(.varianceChannelsDiscarded(beforeChannels: 8, afterChannels: 4, beforeStat: beforeStat, afterStat: afterStat))`.
4. Every `throw` site touched by this sortie (init `:61` and any new throws added to `apply(weights:)`) must be preceded by `await telemetry?.capture(.errorThrown(phase: .weightApply | .ditInit, errorDescription: String(describing: error)))` if telemetry was captured prior to the throw. **Note**: `apply(weights:)` does not currently throw in the existing code; if Sortie 3 adds throws (e.g., for shape validation), pair them with `.errorThrown` emit.

**Exit criteria**:
- [ ] `make build` succeeds.
- [ ] `grep -c 'weightApplyStart' Sources/PixArtBackbone/PixArtDiT.swift` equals exactly 1.
- [ ] `grep -c 'weightApplyComplete' Sources/PixArtBackbone/PixArtDiT.swift` equals exactly 1.
- [ ] `grep -c 'microConditioningStatus' Sources/PixArtBackbone/PixArtDiT.swift` equals exactly 1.
- [ ] `grep -c 'varianceChannelsDiscarded' Sources/PixArtBackbone/PixArtDiT.swift` equals exactly 1.
- [ ] `grep -c 'weightUnload' Sources/PixArtBackbone/PixArtDiT.swift` equals exactly 1.
- [ ] `grep -B1 'TuberiaTensorStat.sample' Sources/PixArtBackbone/PixArtDiT.swift` shows every `.sample(...)` call inside a `telemetry`-guarded block (manual inspection counts as PASS only if every match is preceded by an `if let telemetry` opener within the same scope).
- [ ] `make test` green (no regression).

**Agent allocation**: Supervising agent (build + test verification).

---

### Sortie 4: Recipe `validate()` telemetry parameter

**Priority**: 7.0 — Lower urgency (recipe events are cold path, fire once per pipeline init), but parallelizable with Sortie 3 and Sortie 5. Foundation_score=0, dependency_depth=2, risk=1, complexity=1. Score: 2*3 + 0*2 + 1*1 + 1*0.5 = 7.5.

**Estimated turns**: 16 — Right-sized.

**Entry criteria**:
- [ ] Sortie 2 exit criteria met.
- [ ] Sortie 1 exit criteria met (uses `PixArtTelemetryReporter` and `PixArtTelemetryEvent` types).

**Tasks**:
1. In `Sources/PixArtBackbone/PixArtRecipe.swift` `validate()` (line 159): change signature to `public func validate(telemetry: (any PixArtTelemetryReporter)? = nil) async throws`. The defaulted nil preserves source compatibility for existing callers.
2. Inside `validate(telemetry:)`:
   - Before each `throw` at lines 162, 168, 174: emit `await telemetry?.capture(.recipeValidationFailed(name: "PixArtRecipe", check: <check-name>, reason: <reason-string>))`. Use check names: `"encoder_caption_channels"`, `"encoder_text_length"`, `"decoder_latent_channels"`.
   - After all three checks pass (just before function returns): emit `await telemetry?.capture(.recipeValidated(name: "PixArtRecipe", checksPassed: 3))`.
3. Same treatment for `Sources/PixArtBackbone/PixArtFP16Recipe.swift` `validate()` (line 135), throw sites at lines 138, 144, 150. Use `name: "PixArtFP16Recipe"`, same three check names.
4. Add a doc comment on each `validate(telemetry:)` documenting the chosen integration pattern per REQUIREMENTS §4.2: **"The host calls `try await recipe.validate(telemetry: pixartAdapter)` explicitly before `DiffusionPipeline.init`. The defaulted `nil` parameter preserves source compatibility for callers that do not yet wire telemetry."**

**Exit criteria**:
- [ ] `make build` succeeds.
- [ ] `grep -n 'func validate(telemetry:' Sources/PixArtBackbone/PixArtRecipe.swift` matches.
- [ ] `grep -n 'func validate(telemetry:' Sources/PixArtBackbone/PixArtFP16Recipe.swift` matches.
- [ ] `grep -c 'recipeValidationFailed' Sources/PixArtBackbone/PixArtRecipe.swift` equals exactly 3.
- [ ] `grep -c 'recipeValidationFailed' Sources/PixArtBackbone/PixArtFP16Recipe.swift` equals exactly 3.
- [ ] `grep -c 'recipeValidated' Sources/PixArtBackbone/PixArtRecipe.swift` equals exactly 1.
- [ ] `grep -c 'recipeValidated' Sources/PixArtBackbone/PixArtFP16Recipe.swift` equals exactly 1.
- [ ] `grep -n 'async throws' Sources/PixArtBackbone/PixArtRecipe.swift` matches.
- [ ] `make test` green (callers of `validate()` updated to `try await` where needed).

**Agent allocation**: Supervising agent.

---

### Sortie 5: Non-step-boundary hot-path forward emissions

**Priority**: 8.5 — High risk (hot path, must not degrade perf), foundation for Sortie 6. Foundation_score=1, dependency_depth=2 (blocks Sortie 6), risk=3 (perf-critical), complexity=2. Score: 2*3 + 1*2 + 3*1 + 2*0.5 = 12.

**Estimated turns**: 28 — Right-sized but upper end.

**Sergeant note**: This sortie wires the emissions inside `PixArtDiT.forward(_:)` **except** the `ditForwardComplete` step-boundary (which is Sortie 6) and the `varianceChannelsDiscarded` slice (which is Sortie 3). It MUST establish the `let telemetry = currentTelemetry()` cache at function entry — Sortie 6 will reuse that cache. See Sortie 3 Task 3 coordination note: if Sortie 3 lands first, it will have already added the cache; check before duplicating.

**Entry criteria**:
- [ ] Sortie 2 exit criteria met.

**Tasks**:
1. At top of `forward(_:)` (`PixArtDiT.swift:111`), add (if not already present from Sortie 3 race): `let telemetry = currentTelemetry()` and `let forwardStart = Date()`. **ONE** lock acquisition per forward.
2. After input extraction, before patchEmbed:
   ```swift
   if let telemetry {
       await telemetry.capture(.ditForwardStart(
           stepIndex: input.stepIndex,
           batch: latents.shape[0],
           latentShape: latents.shape,
           conditioningShape: conditioning.shape,
           timestepShape: timestep.shape,
           inputLatentStat: TuberiaTensorStat.sample(latents),
           conditioningStat: TuberiaTensorStat.sample(conditioning)))
   }
   ```
   Note: if `BackboneInput` does not expose `stepIndex`, pass `nil` and document.
3. After `let patched = patchEmbed(latents)` (`:126`): emit `.patchEmbedComplete(stat: TuberiaTensorStat.sample(patched), gridH:, gridW:)` inside `if let telemetry`.
4. After `let y = captionProjection(conditioning)` (`:142`): emit `.captionProjectionComplete(stat:)` inside `if let telemetry`.
5. After `let tBlock = tBlockLinear(...)` (`:159`): emit `.timestepEmbeddingComplete(sinusoidalStat:, projectedStat:, tBlockStat:)` inside `if let telemetry`. Three stats — sample each only inside the guarded block.
6. Immediately after the manual silu math at `:157–159`, INSIDE the same `if let telemetry` block as the timestep emission (or a sibling block): emit `.siluWorkaroundExecuted` (no payload, zero-cost marker).
7. After `var output = finalLayer(x, t: tRaw, ...)` (`:171`): emit `.finalLayerComplete(stat: TuberiaTensorStat.sample(output))` inside `if let telemetry` — this is the 8-channel output BEFORE Sortie 3's variance discard.

**Hot-path discipline (mandatory):**
- All `TuberiaTensorStat.sample(...)` calls MUST be inside `if let telemetry { ... }` blocks. Never sample unconditionally. The user's `@autoclosure` discipline is honored here by Swift's standard lazy evaluation through the `if let` guard — `sample()` is never called when `telemetry == nil`.
- All `await telemetry.capture(...)` calls go inside the same `if let` block.
- No additional `currentTelemetry()` calls inside `forward(_:)` — reuse the entry-cached `telemetry` only.
- Build steps to confirm: `make build` (verifies syntax), `make test --filter PixArtTelemetryForwardSequenceTests` (will fail until Sortie 7a lands; not part of Sortie 5 exit).

**Exit criteria**:
- [ ] `make build` succeeds.
- [ ] `grep -c 'currentTelemetry()' Sources/PixArtBackbone/PixArtDiT.swift` equals at most 2 (one in `forward(_:)`, one in `apply(weights:)` from Sortie 3). Verify via `grep -n 'currentTelemetry()' Sources/PixArtBackbone/PixArtDiT.swift`.
- [ ] `grep -c 'ditForwardStart' Sources/PixArtBackbone/PixArtDiT.swift` equals exactly 1.
- [ ] `grep -c 'patchEmbedComplete' Sources/PixArtBackbone/PixArtDiT.swift` equals exactly 1.
- [ ] `grep -c 'captionProjectionComplete' Sources/PixArtBackbone/PixArtDiT.swift` equals exactly 1.
- [ ] `grep -c 'timestepEmbeddingComplete' Sources/PixArtBackbone/PixArtDiT.swift` equals exactly 1.
- [ ] `grep -c 'siluWorkaroundExecuted' Sources/PixArtBackbone/PixArtDiT.swift` equals exactly 1.
- [ ] `grep -c 'finalLayerComplete' Sources/PixArtBackbone/PixArtDiT.swift` equals exactly 1.
- [ ] Manual verification (or `awk` block scan): every `TuberiaTensorStat.sample(...)` call inside `forward(_:)` is lexically nested inside an `if let telemetry` block.
- [ ] `make test` green.

**Agent allocation**: Supervising agent.

---

### Sortie 6: `ditForwardComplete` step-boundary emission

**Priority**: 9.5 — Highest-risk cardinality constraint (must fire EXACTLY once per scheduler step). Risk=3, complexity=1, foundation_score=0, dependency_depth=2. Score: 2*3 + 0*2 + 3*1 + 1*0.5 = 9.5. But because the cardinality constraint is the most error-prone part of the entire mission, treat as effectively top-priority for verification rigor.

**Estimated turns**: 8 — Right-sized.

**Sergeant note**: This sortie owns the **single step-boundary event**. It fires exactly once per scheduler step, just before `return output` at `PixArtDiT.swift:177`. The cardinality is critical: under no circumstance may this fire per DiT block or per attention head. `@autoclosure`-style discipline mandatory — `TuberiaTensorStat.sample(output)` must NOT execute when telemetry is nil. Use `if let telemetry` guard at the emission site, reusing Sortie 5's cached `telemetry` binding.

**Entry criteria**:
- [ ] Sortie 5 exit criteria met (the `let telemetry = currentTelemetry()` cache and `let forwardStart = Date()` at top of `forward(_:)` exist).
- [ ] Sortie 3 exit criteria met (`varianceChannelsDiscarded` already emitted upstream; Sortie 6 fires AFTER that event in the sequence).

**Tasks**:
1. Just before `return output` at `PixArtDiT.swift:177`:
   ```swift
   if let telemetry {
       let duration = Date().timeIntervalSince(forwardStart)
       await telemetry.capture(.ditForwardComplete(
           stepIndex: input.stepIndex,
           outputStat: TuberiaTensorStat.sample(output),
           durationSeconds: duration))
   }
   return output
   ```
2. Verify cardinality by inspection: there must be exactly ONE `.ditForwardComplete` emission in the file, OUTSIDE any block-iteration loop (`for block in blocks { ... }` at approximately line 162).

**Exit criteria**:
- [ ] `make build` succeeds.
- [ ] `grep -c 'ditForwardComplete' Sources/PixArtBackbone/PixArtDiT.swift` equals **exactly 1**.
- [ ] `awk '/for block in blocks/,/^}/' Sources/PixArtBackbone/PixArtDiT.swift | grep -c 'ditForwardComplete'` equals **exactly 0** (not inside the per-block loop).
- [ ] The line immediately following the `.ditForwardComplete` emission's closing brace is `return output` (verify via context grep).
- [ ] `make test` green.

**Agent allocation**: Supervising agent.

---

### Sortie 7a: Weight-apply correctness tests (INT4 + FP16 + sequence)

**Priority**: 7.5 — Verification of Sorties 3+5+6 together. Risk=2, complexity=2, foundation_score=0, dependency_depth=1. Score: 1*3 + 0*2 + 2*1 + 2*0.5 = 6.

**Estimated turns**: 22 — Right-sized.

**Entry criteria**:
- [ ] Sorties 3, 5, 6 exit criteria met.
- [ ] Sortie 1 exit criteria met.

**Tasks**:
1. Add `Tests/PixArtBackboneTests/MockReporter.swift` (shared test helper, NOT a product source file): an actor `MockReporter` conforming to `PixArtTelemetryReporter` with an append-only `events: [PixArtTelemetryEvent]` log, an `async capture(_:)` method, and an `async snapshot() -> [PixArtTelemetryEvent]` accessor. Document as test-only.
2. Add `Tests/PixArtBackboneTests/PixArtTelemetryWeightApplyINT4Tests.swift`. Construct a synthetic int4 weight dict (one `<key>.weight` packed uint32 + `<key>.scales` + `<key>.biases`). Run `apply(weights:)`. Assert:
   - Exactly one `weightApplyStart(quantization: .int4, weightKeyCount: 3)` event.
   - Exactly one `weightApplyComplete` event with `quantization == .int4 && dequantizedKeys > 0 && passThroughKeys == 0 && scalesBiasesSkipped > 0`.
   - Exactly one `microConditioningStatus(present: false, sizeEmbedderFound: false, arEmbedderFound: false)` event.
3. Add `Tests/PixArtBackboneTests/PixArtTelemetryWeightApplyFP16Tests.swift`. Construct a synthetic fp16 weight dict (one `<key>.weight` float16, no sidecars). Assert:
   - `weightApplyComplete(quantization: .fp16, dequantizedKeys: 0, passThroughKeys: > 0, scalesBiasesSkipped: 0)`.
4. Add `Tests/PixArtBackboneTests/PixArtTelemetryForwardSequenceTests.swift`. Run one `forward(_:)` with a small synthetic input. Assert event order:
   ```
   ditForwardStart → patchEmbedComplete → captionProjectionComplete →
   timestepEmbeddingComplete → siluWorkaroundExecuted →
   finalLayerComplete → varianceChannelsDiscarded → ditForwardComplete
   ```
   Use exact-count assertions: `XCTAssertEqual(events.filter { ... }.count, 1)` for each event type. Most importantly: `XCTAssertEqual(events.filter { if case .ditForwardComplete = $0 { return true }; return false }.count, 1)`.

**Exit criteria**:
- [ ] `make test` green for all three new test files.
- [ ] `test -f Tests/PixArtBackboneTests/MockReporter.swift`
- [ ] `test -f Tests/PixArtBackboneTests/PixArtTelemetryWeightApplyINT4Tests.swift`
- [ ] `test -f Tests/PixArtBackboneTests/PixArtTelemetryWeightApplyFP16Tests.swift`
- [ ] `test -f Tests/PixArtBackboneTests/PixArtTelemetryForwardSequenceTests.swift`
- [ ] `grep -c 'ditForwardComplete' Tests/PixArtBackboneTests/PixArtTelemetryForwardSequenceTests.swift` ≥ 2 (one for the filter, one for the count assertion).

**Agent allocation**: Supervising agent (runs `make test`).

---

### Sortie 7b: Variance-discard and anomaly tests

**Priority**: 7.5 — Same family as 7a but distinct test files; parallelizable with 7a and 8. Risk=2, complexity=1.5. Score: 1*3 + 0*2 + 2*1 + 1.5*0.5 = 5.75.

**Estimated turns**: 16 — Right-sized.

**Entry criteria**:
- [ ] Sorties 3, 5, 6 exit criteria met.
- [ ] Sortie 7a's `MockReporter` helper exists (file dependency only).

**Tasks**:
1. Add `Tests/PixArtBackboneTests/PixArtTelemetryVarianceDiscardTests.swift`. Run one forward. Find the `varianceChannelsDiscarded` event in `MockReporter` log. Assert `beforeChannels == 8 && afterChannels == 4`. Assert `beforeStat.count == 8 * gridH * gridW * batch` and `afterStat.count == 4 * gridH * gridW * batch` (or analogous shape check).
2. Add `Tests/PixArtBackboneTests/PixArtTelemetryAnomalyTests.swift`. Construct a `PixArtDiT` whose `patchEmbed` output is forced to contain a `Float.nan` (either via a custom test subclass override or by injecting a NaN-poisoned input latent). Assert:
   - `patchEmbedComplete.stat.hasNaN == true` (assuming `TuberiaTensorStat` exposes `hasNaN`).
   - A `.numericalAnomaly(phase: "pixart_patch_embed", kind: .nan, ...)` event fires.
   - **Note**: REQUIREMENTS §5 says the anomaly is emitted "Inside `TuberiaTensorStat.sample` post-construction, when hasNaN/hasInf observed". If `TuberiaTensorStat.sample` does NOT auto-emit the anomaly, this sortie must add a `if stat.hasNaN || stat.hasInf` block alongside `patchEmbedComplete` emission in Sortie 5's edits. See Open Question Q4.1.

**Exit criteria**:
- [ ] `make test` green for both new test files.
- [ ] `test -f Tests/PixArtBackboneTests/PixArtTelemetryVarianceDiscardTests.swift`
- [ ] `test -f Tests/PixArtBackboneTests/PixArtTelemetryAnomalyTests.swift`
- [ ] `grep -c 'numericalAnomaly' Tests/PixArtBackboneTests/PixArtTelemetryAnomalyTests.swift` ≥ 1.

**Agent allocation**: Supervising agent.

---

### Sortie 8: Lock-contention TSan test

**Priority**: 7.0 — Concurrency safety verification. Risk=3, complexity=2. Score: 0*3 + 0*2 + 3*1 + 2*0.5 = 4.

**Estimated turns**: 14 — Right-sized.

**Entry criteria**:
- [ ] Sorties 3, 5, 6 exit criteria met.
- [ ] Sortie 7a's `MockReporter` helper exists.

**Tasks**:
1. Add `Tests/PixArtBackboneTests/PixArtTelemetryLockContentionTests.swift`.
2. Body: launch a `TaskGroup` with:
   - One task running `forward(_:)` (or a synthetic loop that calls `currentTelemetry()` 1000 times) on a `PixArtDiT` instance.
   - Several tasks calling `setTelemetry(MockReporter())` and `setTelemetry(nil)` alternately.
3. Run with thread sanitizer: `make test ARGS="-enableThreadSanitizer YES"` if Makefile supports it, else document the CI invocation `xcodebuild test -enableThreadSanitizer YES -scheme pixart-swift-mlx -destination 'platform=macOS,arch=arm64'`. Add a `Makefile` target `test-tsan` if missing.
4. Assert no TSan diagnostics surface in the test output.

**Exit criteria**:
- [ ] `make test` green.
- [ ] `test -f Tests/PixArtBackboneTests/PixArtTelemetryLockContentionTests.swift`
- [ ] `grep -c 'setTelemetry' Tests/PixArtBackboneTests/PixArtTelemetryLockContentionTests.swift` ≥ 2 (toggle to a reporter and to nil).
- [ ] TSan run (manual or CI) reports zero races. **Note**: TSan invocation may not be runnable from `make test` directly — see Open Question Q8.1.

**Agent allocation**: Supervising agent.

---

### Sortie 9: Baseline overhead test (20-step generation)

**Priority**: 8.0 — Project's headline performance gate. Final gate before tag. Risk=3, complexity=2. Score: 0*3 + 0*2 + 3*1 + 2*0.5 = 4. But escalated to 8.0 because regressions caught here block release.

**Estimated turns**: 18 — Right-sized.

**Sergeant note**: This is the project's headline performance gate. With a `nil` reporter and with `NoopPixArtTelemetryReporter`, the 20-forward median wall-clock delta must be within ±2% per REQUIREMENTS §7. The user's framing is "under +1% with Noop reporter" — we adopt the stricter +1% target as the primary gate, and tolerate ±2% as the secondary tolerance band to absorb cold-cache jitter.

**Entry criteria**:
- [ ] Sorties 1–8 exit criteria met. ALL prior sorties green.

**Tasks**:
1. Add `Tests/PixArtBackboneTests/PixArtTelemetryNoopOverheadTests.swift`.
2. Setup: load a minimal `PixArtDiT` (synthetic weights acceptable; the per-call cost is what matters, not output correctness).
3. Warm up: 3 untimed `forward(_:)` calls (avoids first-call MLX kernel compilation skew).
4. Measurement A: `setTelemetry(nil)`. Run 20 `forward(_:)` calls. Record per-call wall-clock; compute median.
5. Measurement B: `setTelemetry(NoopPixArtTelemetryReporter())`. Run 20 `forward(_:)` calls. Record per-call wall-clock; compute median.
6. Compute `delta = (medianB - medianA) / medianA`.
7. Assertion ladder:
   - `delta < 0.01` (1%): PASS.
   - `0.01 <= delta < 0.02` (1–2%): emit `XCTSkip("Overhead in soft-fail band: delta=\(delta)")` with a diagnostic dump of both medians.
   - `delta >= 0.02`: hard fail (`XCTFail`).
8. Always print both medians and the delta to test output via `print(...)` so CI logs preserve the forensic record.

**Exit criteria**:
- [ ] `make test` green (PASS or soft-skip, never hard fail in CI).
- [ ] `test -f Tests/PixArtBackboneTests/PixArtTelemetryNoopOverheadTests.swift`
- [ ] Test output contains a printed line matching the pattern `nil-median=.*noop-median=.*delta=.*`.
- [ ] Final `make build && make test` green on a clean checkout (mission-readiness gate).

**Agent allocation**: Supervising agent.

---

## Parallelism Structure

**Critical Path**: Sortie 1 → Sortie 2 → Sortie 5 → Sortie 6 → Sortie 7a → Sortie 9 (length: 6 sorties).

**Parallel Execution Groups**:

- **Group 1 — Layer 0**: Sortie 1 (supervising agent). Cannot parallelize.
- **Group 2 — Layer 1**: Sortie 2 (supervising agent). Cannot parallelize.
- **Group 3 — Layer 2 (PARALLEL, up to 3 agents)**:
  - Sortie 3 — supervising agent (build steps).
  - Sortie 4 — **NO BUILD** (sub-agent eligible). Edits two files (recipes), no overlap with Sortie 3 or 5.
  - Sortie 5 — supervising agent (build steps).
  - **File conflict warning**: Sortie 3 and Sortie 5 BOTH need `let telemetry = currentTelemetry()` at the top of `forward(_:)`. Coordination: dispatch Sortie 5 FIRST and let it add the cache; Sortie 3 then reuses it. If parallel dispatch is unavoidable, dispatch them serially within Layer 2 and let Sortie 4 fan out in parallel with whichever runs first.
- **Group 4 — Layer 3**: Sortie 6 (supervising agent). Must run after Sortie 5.
- **Group 5 — Layer 4 (PARALLEL, up to 3 agents)**:
  - Sortie 7a — supervising agent.
  - Sortie 7b — supervising agent.
  - Sortie 8 — supervising agent.
  - All three add NEW test files only, no source overlap. Can run fully in parallel (3 sub-agents at most).
- **Group 6 — Layer 5**: Sortie 9 (supervising agent). Final gate.

**Agent Constraints**:

- **Supervising agent**: Handles all sorties with `make build` / `make test` steps — Sorties 1, 2, 3, 5, 6, 7a, 7b, 8, 9.
- **Sub-agents (up to 4)**: Only Sortie 4 is truly build-free at the source level (recipe edits compile but the verification gates are file-level grep + `make build` at the end). Sortie 4 can be dispatched to a sub-agent if the verification step is collected by the supervisor at end-of-layer.
- **Realistic parallelism**: 2 agents at Layer 2 (Sortie 3 supervising + Sortie 4 sub) plus serial Sortie 5; 3 agents at Layer 4 (all supervising, since each runs `make test` on a disjoint test target). Practical max sub-agents: 1 (Sortie 4 only).

**Missed Opportunities**:
- None within the current dependency graph. The forward(_:) edits (Sortie 3 variance + Sortie 5 hot path + Sortie 6 step boundary) cannot be parallelized due to shared file region. The recipe edits (Sortie 4) are the only true side-channel.

---

## Open Questions & Missing Documentation

### Unresolved Items — Cast-Site Ambiguity in `PixArtRecipe` / `PixArtFP16Recipe`

The user's task framing explicitly asks: "flag every cast site in `PixArtRecipe` / `PixArtFP16Recipe` where the spec is ambiguous about whether a `dtypeBoundary` event should fire."

**Direct finding from source inspection (`PixArtRecipe.swift:159–179`, `PixArtFP16Recipe.swift:135–155`):**

> The `validate()` methods in BOTH recipes contain ZERO tensor operations. They perform three integer/Int property comparisons (`Int == Int` shape consistency checks) and throw `PixArtRecipeError.shapeMismatch` on failure. There are NO `MLXArray` allocations, NO `.asType(...)` calls, NO dequantization, NO dtype casts of any kind in either `validate()` body.

This means: **there are no `dtypeBoundary` cast sites in the recipe files themselves.** The user's framing assumed cast sites would exist there; the actual code design places ALL dtype-cast logic in `PixArtDiT.apply(weights:)` (line 234: `floatWeight = dequantized(...)` and line 236: `.asType(.float16)`) and `PixArtDiT.forward(_:)` (the 8→4 channel slice at line 175).

| ID | Issue | Location | Recommendation |
|----|-------|----------|----------------|
| Q4.1 | The recipes have zero tensor cast sites. Does the user still want a `dtypeBoundary` event signature added that recipes COULD fire (e.g., post-construction sanity casts)? Currently the spec does not define such an event. | `PixArtRecipe.swift:159–179`, `PixArtFP16Recipe.swift:135–155` | **Decision needed before Sortie 4 dispatches.** Default: no recipe-level `dtypeBoundary` event — recipes only emit `recipeValidated` / `recipeValidationFailed`. Confirm or override. |
| Q3.1 | Should `.asType(.float16)` at `PixArtDiT.swift:236` fire its own `dtypeBoundary` (or analogous) event, distinct from the aggregate `weightApplyComplete` dequant count? REQUIREMENTS §1 says "Internal `dequantized(...)` kernel internals — block-aggregated counts are sufficient" (out of scope). This implies NO per-tensor event at line 236. But user task framing says "all FP16 quant cast sites" must fire. | `PixArtDiT.swift:236` (the `.asType(.float16)` call inside the dequant loop) | **Default: NO per-tensor event.** Honor REQUIREMENTS §1 explicit out-of-scope. The aggregate `weightApplyComplete.dequantizedKeys` count is the sanctioned diagnostic. Document this trade-off in the Sortie 3 PR. Confirm or override. |
| Q3.2 | The dequant loop at lines 207–241 has TWO logical cast sites per `.weight` key: (a) `dequantized(...)` at line 234 (uint32 → float32), and (b) `.asType(.float16)` at line 236 (float32 → float16). Should both increment `dequantizedKeys`, or count them as one logical operation? | `PixArtDiT.swift:234–236` | **Default: count as ONE logical dequant.** `dequantizedKeys` is intended to count `.weight` keys consumed via the int4 path, not individual MLX kernel invocations. |
| Q3.3 | The variance-discard slice at `PixArtDiT.swift:175` (`output = output[..., ..., ..., 0..<4]`) is technically NOT a dtype cast (it's a channel slice; dtype is preserved). Should it nonetheless be classified as a `dtypeBoundary` for telemetry purposes? | `PixArtDiT.swift:175` | **Default: YES, treat as a boundary** — it's a shape-cast boundary where wrong slicing produces silent corruption (REQUIREMENTS §1 "Wrong slicing here would produce silent corruption"). The `varianceChannelsDiscarded` event is its dedicated signature. |
| Q5.1 | `BackboneInput.stepIndex`: does the existing `BackboneInput` type expose a `stepIndex` field? If not, Sortie 5's `.ditForwardStart(stepIndex: input.stepIndex, ...)` falls back to `nil`. REQUIREMENTS §3.1 already marks `stepIndex` as `Int?` "populated when caller passes it through; nil if standalone test", so `nil` is acceptable today. | `Sources/PixArtBackbone/PixArtBackbone.swift` (`BackboneInput`) | **Default: pass `nil` if field absent.** Verify by reading `BackboneInput` definition in Sortie 5. If absent, file a separate proposal upstream (Tuberia) to add `stepIndex` — not a blocker for this mission. |
| Q5.2 | The hot-path emission pattern uses `if let telemetry { await telemetry.capture(...) }`. The user task framing says "`@autoclosure` discipline mandatory." Swift's `if let` already provides lazy evaluation (the `TuberiaTensorStat.sample(...)` call inside the body is only evaluated when `telemetry != nil`), which is functionally equivalent to `@autoclosure`. Does the user want a literal `@autoclosure`-wrapped helper (e.g., `telemetry?.captureIfPresent(@autoclosure () -> PixArtTelemetryEvent)`)? | All hot-path emission sites in Sortie 5, 6 | **Default: `if let` guard is acceptable.** Adding a literal `@autoclosure` helper would require modifying the `PixArtTelemetryReporter` protocol surface; REQUIREMENTS §3.2 defines a simple `func capture(_:)` only. Confirm or escalate. |
| Q7b.1 | REQUIREMENTS §5 says `.numericalAnomaly` fires "Inside `TuberiaTensorStat.sample` post-construction, when hasNaN/hasInf observed." This implies SwiftTuberia owns the anomaly emit. But SwiftTuberia does not know about `PixArtTelemetryReporter` — it cannot emit pixart-typed events. Either (a) `TuberiaTensorStat.sample` exposes a `hasNaN`/`hasInf` flag and PixArtDiT manually emits the pixart-typed anomaly alongside `patchEmbedComplete`, or (b) the spec needs revision. | All forward emission sites in Sortie 5 | **Default: option (a).** After every `let stat = TuberiaTensorStat.sample(x)` in Sortie 5, add `if stat.hasNaN || stat.hasInf { await telemetry.capture(.numericalAnomaly(phase: "<event-name>", kind: stat.hasNaN ? .nan : .inf, stepIndex: input.stepIndex, stat: stat)) }`. Document this interpretation in Sortie 5's PR. |
| Q8.1 | Does the project Makefile expose a TSan-enabled test target? If `make test` does not pass through `-enableThreadSanitizer YES`, Sortie 8's exit criterion is not directly verifiable via `make test`. | `Makefile` | **Action**: Sortie 8 must add a `make test-tsan` target if missing, or document the raw `xcodebuild` invocation in the test header comment. Verify in Sortie 8. |
| Q9.1 | The 20-step overhead test depends on `forward(_:)` being callable with synthetic weights (no real safetensors load). Does `PixArtDiT` support a "no-weights" or random-init test mode? If not, Sortie 9 needs to fabricate a small int4 fixture, which adds context cost. | `PixArtDiT.init` + `apply(weights:)` | **Action**: Sortie 9 agent must verify before building the test. If synthetic-weight construction is too heavy, downscope to "10 forward calls of a stub `forward(_:)` analog that exercises only the telemetry-emission code paths" — but that weakens the gate. Escalate before downscoping. |
| Q2.1 | `PixArtDiT` is declared `@unchecked Sendable`. Adding `OSAllocatedUnfairLock` is the standard pattern (mirrors flux's approach). Verify `OSAllocatedUnfairLock` availability on the minimum platform target (macOS 14+, iOS 17+) — confirm `Package.swift` `platforms:` block. | `Package.swift`, `Sources/PixArtBackbone/PixArtDiT.swift` | **Confirmed**: `Package.swift` sets `.macOS(.v26), .iOS(.v26)`. `OSAllocatedUnfairLock` is available from macOS 13 / iOS 16 — well within range. No blocker. |

### Auto-fixed in this Refine Pass

- Vague exit criterion in original Sortie 3 (`grep -c ... >= 1`) replaced with exact-count assertions (`equals exactly 1`) for every emission, eliminating false positives from accidentally-duplicated emit calls.
- Original Sortie 7 (5 test files in one sortie, ~30 turns) split into Sortie 7a (3 test files + shared MockReporter helper) and Sortie 7b (2 test files), each ~16–22 turns. Both now right-sized.
- File-region conflict between Sortie 3's variance-discard edit and Sortie 5's hot-path edits explicitly called out in both sorties' task lists with a dispatch-order coordination note.
- Sortie 9's tolerance ladder formalized (primary 1%, soft-fail 1–2%, hard-fail >2%) with explicit `XCTSkip` and `XCTFail` semantics.

### Manual Review Required (blocks dispatch)

Items Q4.1, Q3.1, Q5.2, Q7b.1, Q9.1 are flagged as **non-blocking with documented defaults** — sortie agents should proceed with the default interpretation and surface the decision in the PR description for human ratification. If the user wants any of these handled differently before dispatch, they must amend this section.

No items are HARD-BLOCKING. The plan is dispatchable with the documented defaults.

---

## Summary

| Metric | Value |
|--------|-------|
| Work units | 1 |
| Total sorties | 10 |
| Dependency structure | 6 layers; parallel opportunities at Layer 2 (3 sorties) and Layer 4 (3 sorties) |
| Hot-path sorties | 2 (Sortie 5 wires non-boundary events; Sortie 6 owns the `ditForwardComplete` step boundary exclusively) |
| `dtypeBoundary` / cast-site sortie | 1 (Sortie 3 — weight-apply dequant, microConditioningStatus, varianceChannelsDiscarded) |
| Baseline overhead sortie | 1 (Sortie 9 — final gate) |
| Critical path length | 6 sorties (1 → 2 → 5 → 6 → 7a → 9) |
| Estimated total turns | ~170 (10 sorties × avg 17 turns) |
| Max parallel agents | 3 at Layer 2 and Layer 4 (mostly supervising agents; 1 true sub-agent slot at Sortie 4) |
| Open questions requiring user decision before dispatch | 0 hard blockers; 5 default-resolved items flagged for PR-time ratification |

---

## Refinement Pass Results

| Pass | Status | Changes |
|------|--------|---------|
| 1. Atomicity & Testability | PASS | Sortie 7 split into 7a + 7b; vague `>=1` grep gates replaced with `equals exactly N` exact-count gates throughout; auto-split plan added for Sortie 3 if it overruns. |
| 2. Prioritization | PASS | Priority scores annotated on every sortie; Sortie 6 elevated to top tier of verification rigor due to cardinality risk; Sortie 9 elevated to 8.0 due to release-gating role. |
| 3. Parallelism | PASS | 6 layers with parallel groups identified at Layer 2 (Sorties 3, 4, 5 with file-conflict coordination note) and Layer 4 (Sorties 7a, 7b, 8). Build steps confined to supervising agent. Sortie 4 identified as the lone true sub-agent slot. |
| 4. Open Questions & Vague Criteria | PASS (5 items default-resolved, none hard-blocking) | 5 cast-site ambiguities flagged across `PixArtRecipe` / `PixArtFP16Recipe` and `PixArtDiT`; defaults documented; all expected to be ratified at PR review. Plus Q5.2 (`@autoclosure` interpretation), Q7b.1 (anomaly emission ownership), Q8.1 (TSan invocation), Q9.1 (synthetic weight feasibility). |

**VERDICT**: Plan is ready to execute. Default interpretations for open questions are documented in-line; no hard blockers.

**Next step**: User ratifies open-question defaults, then `/mission-supervisor start /Users/stovak/Projects/pixart-swift-mlx/EXECUTION_PLAN.md`.
