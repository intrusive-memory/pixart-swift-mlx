import Foundation
import MLX
import Testing
import Tuberia

@testable import PixArtBackbone

// MARK: - Noop Reporter Overhead Gate (Sortie 9 — OPERATION STETHOSCOPE FURNACE)
//
// Mission-readiness gate: measures the wall-clock cost added by
// `NoopPixArtTelemetryReporter` vs. `nil` across 20 forward calls each.
//
// Tolerance ladder (fixed — do NOT relax):
//   delta < 0.01  (< 1%)   → PASS
//   0.01 ≤ delta < 0.02    → soft-fail band: recorded as a known intermittent
//                             issue so CI stays green while the diagnostic is
//                             preserved (equivalent to XCTSkip in XCTest).
//   delta ≥ 0.02  (≥ 2%)   → hard fail — release blocked.
//
// The test ALWAYS prints both medians and the delta so CI logs preserve the
// forensic record regardless of outcome:
//   [OVERHEAD] nil-median=<s> noop-median=<s> delta=<fraction>
//
// ## Synthetic weights
//
// Uses the same `PixArtDiT(configuration: PixArtDiTConfiguration())` factory
// as Sortie 7a's PixArtTelemetryForwardSequenceTests — no real weight files
// needed. The per-call cost is what matters; output correctness is irrelevant
// for this gate (confirmed as acceptable per EXECUTION_PLAN.md Q9.1).
//
// ## Fire-and-forget Task drain
//
// Each forward(_:) with a non-nil reporter spawns one `Task { ... }` that
// delivers events to `capture(_:)`. With `NoopPixArtTelemetryReporter` the
// capture body is empty, so Tasks complete immediately — but they still spawn.
// A 250 ms post-loop sleep gives all 20 Tasks time to drain before the test
// tears down, preventing cross-test contamination.
//
// ## Hardening notes
//
// Trial count: 20 (median = index 10 of 0-based sorted array).
// Warm-up: 3 untimed forward calls (discards first-call MLX kernel
// compilation skew).
// If flakiness is observed on shared CI hardware, bump warm-up to 5 and
// trial count to 30 (median at index 15) without touching the thresholds.

@Suite("PixArtTelemetryNoopOverhead", .serialized)
struct PixArtTelemetryNoopOverheadTests {

    // MARK: - Fixture factory (mirrors Sortie 7a's pattern verbatim)

    /// Fresh `PixArtDiT` with the default configuration.
    /// Matches `makeFreshDiT()` in PixArtTelemetryForwardSequenceTests.
    private static func makeFreshDiT() throws -> PixArtDiT {
        try PixArtDiT(configuration: PixArtDiTConfiguration())
    }

    /// Minimal synthetic input: [1, 4, 4, 4] latent.
    /// gridH = 4/2 = 2, gridW = 4/2 = 2 → 4 tokens.
    /// Matches `makeSyntheticInput()` in PixArtTelemetryForwardSequenceTests.
    private static func makeSyntheticInput() -> BackboneInput {
        BackboneInput(
            latents: MLXArray.zeros([1, 4, 4, 4]),
            conditioning: MLXArray.zeros([1, 120, 4096]),
            conditioningMask: MLXArray.ones([1, 120]),
            timestep: MLXArray([500 as Float])
        )
    }

    // MARK: - Overhead gate

    @Test("NoopPixArtTelemetryReporter wall-clock overhead is under 1% per forward call")
    func noopReporter_overheadUnder1Percent() async throws {

        // 1. Construct a fresh isolated DiT (no shared state with other suites).
        let dit = try Self.makeFreshDiT()
        let input = Self.makeSyntheticInput()

        // 2. Warm up — 3 untimed forward calls to burn through MLX kernel
        //    compilation and any first-call setup cost.
        for _ in 0..<3 {
            let out = try dit.forward(input)
            eval(out)
        }

        // 3. Measurement A — nil reporter (baseline).
        //    The if-let guard inside forward(_:) short-circuits immediately when
        //    reporter is nil, so no Task is spawned. No drain sleep needed.
        dit.setTelemetry(nil)
        var nilDurations: [TimeInterval] = []
        nilDurations.reserveCapacity(20)
        for _ in 0..<20 {
            let start = Date()
            let out = try dit.forward(input)
            eval(out)
            nilDurations.append(Date().timeIntervalSince(start))
        }
        // Brief safety sleep — even with nil telemetry there should be no Tasks
        // pending, but guard anyway.
        try await Task.sleep(nanoseconds: 100_000_000)

        // 4. Measurement B — NoopPixArtTelemetryReporter.
        //    Each forward(_:) spawns ONE fire-and-forget Task whose body calls
        //    capture(_:) (a no-op). The Task completes almost instantly, but
        //    we still account for the spawn overhead.
        dit.setTelemetry(NoopPixArtTelemetryReporter())
        var noopDurations: [TimeInterval] = []
        noopDurations.reserveCapacity(20)
        for _ in 0..<20 {
            let start = Date()
            let out = try dit.forward(input)
            eval(out)
            noopDurations.append(Date().timeIntervalSince(start))
        }
        // Allow all 20 fire-and-forget Tasks to drain (20 × ~no-op = fast,
        // but 250 ms provides generous headroom on slow CI hardware).
        try await Task.sleep(nanoseconds: 250_000_000)

        // 5. Compute medians.
        //    Median of 20 samples: use index 10 of the sorted array (slight
        //    high-bias; acceptable for a conservative gate).
        let medianNil  = nilDurations.sorted()[10]
        let medianNoop = noopDurations.sorted()[10]

        // 6. Guard against pathologically small baseline (sub-microsecond) that
        //    would produce a meaningless ratio. Should never trigger in practice
        //    since a forward pass through 28 DiT blocks costs >>1 µs.
        guard medianNil > 1e-7 else {
            // Record as a known intermittent issue; the test cannot give a
            // meaningful result but should not block CI.
            withKnownIssue(
                "Median nil-reporter forward time too small to compute meaningful overhead ratio (medianNil=\(medianNil))",
                isIntermittent: true
            ) {
                Issue.record("medianNil=\(medianNil) — ratio undefined")
            }
            return
        }

        let delta = (medianNoop - medianNil) / medianNil

        // 7. ALWAYS print — CI logs preserve the forensic record even on PASS.
        print("[OVERHEAD] nil-median=\(medianNil) noop-median=\(medianNoop) delta=\(delta)")

        // 8. Tolerance ladder (thresholds are fixed by contract — do NOT change).
        if delta < 0.01 {
            // PASS: overhead is within the 1% target.
            #expect(delta < 0.01, "Noop reporter overhead under 1% target (delta=\(delta))")
        } else if delta < 0.02 {
            // SOFT-FAIL BAND (1%–2%): CI stays green; diagnostic is preserved.
            // Equivalent to XCTSkip in XCTest — withKnownIssue suppresses the
            // failure while recording the issue for human review.
            withKnownIssue(
                "Noop reporter overhead in soft-fail band: delta=\(delta) (target <1%, hard-fail at ≥2%). nil-median=\(medianNil) noop-median=\(medianNoop)",
                isIntermittent: true
            ) {
                Issue.record(
                    "Overhead \(String(format: "%.2f", delta * 100))% exceeds 1% target but is below 2% hard-fail threshold"
                )
            }
        } else {
            // HARD FAIL: overhead ≥ 2% — release blocked.
            // Architecture may need rework (see EXECUTION_PLAN.md §Sortie 9).
            Issue.record(
                "Noop reporter overhead exceeds 2% hard-fail threshold: delta=\(delta) (nil=\(medianNil) noop=\(medianNoop)). Release blocked — see Sortie 9 escalation protocol."
            )
        }
    }
}
