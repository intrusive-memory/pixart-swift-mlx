import Foundation
import MLX
import Testing
import Tuberia

@testable import PixArtBackbone

// MARK: - INT4 Weight-Apply Telemetry Tests
//
// Verifies that `PixArtDiT.apply(weights:)` emits the correct telemetry events
// when given a synthetic INT4-quantized weight dictionary.
//
// An INT4 weight dictionary contains three keys per logical weight:
//   - <key>.weight  — uint32 packed, shape [outDim, inDim/8]
//   - <key>.scales  — float16, shape [outDim, numGroups]
//   - <key>.biases  — float16, shape [outDim, numGroups]
//
// Expected events (in dispatch order):
//   1. weightApplyStart(quantization: .int4, weightKeyCount: 3)
//   2. microConditioningStatus(present: false, …)
//   3. weightApplyComplete(quantization: .int4, …)
//
// After `apply(weights:)` returns the events are dispatched via a fire-and-forget
// Task.  Tests sleep 100 ms (Strategy A) before snapshotting the reporter log.

@Suite("PixArtTelemetryWeightApplyINT4", .serialized)
struct PixArtTelemetryWeightApplyINT4Tests {

    // MARK: - Synthetic fixture helpers

    /// Build a minimal but valid INT4-quantized triple using MLX.quantized.
    /// shape [16, 64]: 16 outDim, 1 group of 64 (groupSize=64, bits=4).
    private static func makeINT4Params(key: String = "foo") -> Tuberia.ModuleParameters {
        let outDim = 16
        let inDim = 64
        var values = [Float]()
        values.reserveCapacity(outDim * inDim)
        for i in 0..<(outDim * inDim) {
            values.append(Float(i % 32) / 16.0 - 1.0)
        }
        let w = MLXArray(values).reshaped(outDim, inDim).asType(.float16)
        let quant = quantized(w, groupSize: 64, bits: 4)
        let biases = quant.biases!
        return Tuberia.ModuleParameters(parameters: [
            "\(key).weight": quant.wq,
            "\(key).scales": quant.scales,
            "\(key).biases": biases,
        ])
    }

    // MARK: - weightApplyStart

    @Test("INT4 apply emits weightApplyStart with quantization=.int4 and weightKeyCount=3")
    func int4ApplyEmitsWeightApplyStart() async throws {
        let dit = try PixArtDiT(configuration: PixArtDiTConfiguration())
        let reporter = MockReporter()
        dit.setTelemetry(reporter)

        let params = Self.makeINT4Params()
        try dit.apply(weights: params)

        // Strategy A: wait 100 ms for the fire-and-forget Task to deliver events.
        try await Task.sleep(nanoseconds: 100_000_000)
        let events = await reporter.snapshot()

        let startEvents = events.filter { event -> Bool in
            if case .weightApplyStart(let q, let count) = event {
                return q == .int4 && count == 3
            }
            return false
        }
        #expect(startEvents.count == 1, "Expected exactly one weightApplyStart(.int4, 3); got \(startEvents.count) in \(events)")
    }

    // MARK: - weightApplyComplete

    @Test("INT4 apply emits weightApplyComplete with correct quantization and counters")
    func int4ApplyEmitsWeightApplyComplete() async throws {
        let dit = try PixArtDiT(configuration: PixArtDiTConfiguration())
        let reporter = MockReporter()
        dit.setTelemetry(reporter)

        let params = Self.makeINT4Params()
        try dit.apply(weights: params)

        try await Task.sleep(nanoseconds: 100_000_000)
        let events = await reporter.snapshot()

        let completeEvents = events.filter { event -> Bool in
            if case .weightApplyComplete(
                let q, _, let dequantized, let passThrough, let skipped, _, _) = event {
                return q == .int4
                    && dequantized > 0
                    && passThrough == 0
                    && skipped > 0
            }
            return false
        }
        #expect(
            completeEvents.count == 1,
            "Expected exactly one weightApplyComplete(.int4, dequantizedKeys>0, passThroughKeys=0, scalesBiasesSkipped>0); got \(completeEvents.count) in \(events)"
        )
    }

    // MARK: - microConditioningStatus

    @Test("INT4 apply emits microConditioningStatus(present:false, sizeEmbedderFound:false, arEmbedderFound:false)")
    func int4ApplyEmitsMicroConditioningStatus() async throws {
        let dit = try PixArtDiT(configuration: PixArtDiTConfiguration())
        let reporter = MockReporter()
        dit.setTelemetry(reporter)

        let params = Self.makeINT4Params()
        try dit.apply(weights: params)

        try await Task.sleep(nanoseconds: 100_000_000)
        let events = await reporter.snapshot()

        let statusEvents = events.filter { event -> Bool in
            if case .microConditioningStatus(
                let present, let sizeFound, let arFound) = event {
                return !present && !sizeFound && !arFound
            }
            return false
        }
        #expect(
            statusEvents.count == 1,
            "Expected exactly one microConditioningStatus(present:false, sizeEmbedderFound:false, arEmbedderFound:false); got \(statusEvents.count) in \(events)"
        )
    }

    // MARK: - Event cardinality (all three in one apply call)

    @Test("INT4 apply emits exactly one each of start, complete, and microConditioningStatus")
    func int4ApplyEventCardinality() async throws {
        let dit = try PixArtDiT(configuration: PixArtDiTConfiguration())
        let reporter = MockReporter()
        dit.setTelemetry(reporter)

        let params = Self.makeINT4Params()
        try dit.apply(weights: params)

        try await Task.sleep(nanoseconds: 100_000_000)
        let events = await reporter.snapshot()

        let startCount = events.filter { if case .weightApplyStart = $0 { return true }; return false }.count
        let completeCount = events.filter { if case .weightApplyComplete = $0 { return true }; return false }.count
        let statusCount = events.filter { if case .microConditioningStatus = $0 { return true }; return false }.count

        #expect(startCount == 1, "Expected 1 weightApplyStart, got \(startCount)")
        #expect(completeCount == 1, "Expected 1 weightApplyComplete, got \(completeCount)")
        #expect(statusCount == 1, "Expected 1 microConditioningStatus, got \(statusCount)")
    }
}
