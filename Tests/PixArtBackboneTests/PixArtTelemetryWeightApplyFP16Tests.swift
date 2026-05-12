import Foundation
import MLX
import Testing
import Tuberia

@testable import PixArtBackbone

// MARK: - FP16 Weight-Apply Telemetry Tests
//
// Verifies that `PixArtDiT.apply(weights:)` emits the correct telemetry events
// when given a synthetic FP16 weight dictionary (no INT4 sidecars).
//
// An FP16 weight dictionary contains:
//   - <key>.weight  — float16, shape [outDim, inDim]
//   (No .scales or .biases sidecar keys present.)
//
// Expected event:
//   - weightApplyComplete(quantization: .fp16, dequantizedKeys: 0,
//                         passThroughKeys: > 0, scalesBiasesSkipped: 0)
//
// After `apply(weights:)` returns the events are dispatched via a fire-and-forget
// Task.  Tests sleep 100 ms (Strategy A) before snapshotting the reporter log.

@Suite("PixArtTelemetryWeightApplyFP16", .serialized)
struct PixArtTelemetryWeightApplyFP16Tests {

    // MARK: - Synthetic fixture helpers

    /// Build a minimal FP16 weight dict: one <key>.weight at float16, no sidecars.
    private static func makeFP16Params(key: String = "foo") -> Tuberia.ModuleParameters {
        let weight = MLXArray([1.0, 2.0, 3.0, 4.0] as [Float]).asType(.float16)
        return Tuberia.ModuleParameters(parameters: [
            "\(key).weight": weight,
        ])
    }

    // MARK: - weightApplyStart quantization classification

    @Test("FP16 apply emits weightApplyStart with quantization=.fp16")
    func fp16ApplyEmitsWeightApplyStartWithFP16Quantization() async throws {
        let dit = try PixArtDiT(configuration: PixArtDiTConfiguration())
        let reporter = MockReporter()
        dit.setTelemetry(reporter)

        let params = Self.makeFP16Params()
        try dit.apply(weights: params)

        try await Task.sleep(nanoseconds: 100_000_000)
        let events = await reporter.snapshot()

        let startEvents = events.filter { event -> Bool in
            if case .weightApplyStart(let q, _) = event {
                return q == .fp16
            }
            return false
        }
        #expect(
            startEvents.count == 1,
            "Expected exactly one weightApplyStart(.fp16); got \(startEvents.count) in \(events)"
        )
    }

    // MARK: - weightApplyComplete

    @Test("FP16 apply emits weightApplyComplete with dequantizedKeys=0, passThroughKeys>0, scalesBiasesSkipped=0")
    func fp16ApplyEmitsWeightApplyComplete() async throws {
        let dit = try PixArtDiT(configuration: PixArtDiTConfiguration())
        let reporter = MockReporter()
        dit.setTelemetry(reporter)

        let params = Self.makeFP16Params()
        try dit.apply(weights: params)

        try await Task.sleep(nanoseconds: 100_000_000)
        let events = await reporter.snapshot()

        let completeEvents = events.filter { event -> Bool in
            if case .weightApplyComplete(
                let q, _, let dequantized, let passThrough, let skipped, _, _) = event {
                return q == .fp16
                    && dequantized == 0
                    && passThrough > 0
                    && skipped == 0
            }
            return false
        }
        #expect(
            completeEvents.count == 1,
            "Expected exactly one weightApplyComplete(.fp16, dequantizedKeys:0, passThroughKeys:>0, scalesBiasesSkipped:0); got \(completeEvents.count) in \(events)"
        )
    }

    // MARK: - No telemetry when reporter is nil

    @Test("FP16 apply emits no events when reporter is nil (no crash, no leak)")
    func fp16ApplyNoEventsWhenReporterNil() async throws {
        let dit = try PixArtDiT(configuration: PixArtDiTConfiguration())
        // Do NOT call setTelemetry — reporter stays nil.

        let params = Self.makeFP16Params()
        // Must not throw and must not crash.
        try dit.apply(weights: params)
        #expect(dit.isLoaded == true)
    }
}
