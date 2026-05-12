import Foundation
import MLX
import Testing
import Tuberia

@testable import PixArtBackbone

// MARK: - Variance-Discard Telemetry Tests
//
// Verifies that `PixArtDiT.forward(_:)` emits exactly one `.varianceChannelsDiscarded`
// event when the 8-channel output is sliced to 4 channels, and that the event carries
// the expected `beforeChannels`, `afterChannels`, and stat shapes.
//
// Shape analysis for the synthetic [1, 4, 4, 4] input with patchSize=2, outChannels=8:
//
//   latents input: [B=1, spatialH=4, spatialW=4, inChannels=4]
//   gridH = spatialH / patchSize = 4 / 2 = 2
//   gridW = spatialW / patchSize = 4 / 2 = 2
//
//   FinalLayer output (before slice):
//     [B, gridH * patchSize, gridW * patchSize, outChannels]
//     = [1, 4, 4, 8]                     ← beforeStat.shape
//
//   After slice (variance channels discarded):
//     output[..., 0..<4] → [1, 4, 4, 4]  ← afterStat.shape
//
// Both stat shapes are verified in the assertions below.
//
// IMPORTANT: A fresh `PixArtDiT` instance is used per test (NOT `BackboneFixture.dit`)
// to avoid event contamination from concurrent `forward(_:)` calls in other test suites
// (BackboneForwardTests is not `.serialized` with this suite).
//
// After `forward(_:)` returns, events are dispatched via a fire-and-forget Task.
// Tests sleep 100 ms (Strategy A) before snapshotting the reporter log.

@Suite("PixArtTelemetryVarianceDiscard", .serialized)
struct PixArtTelemetryVarianceDiscardTests {

    // MARK: - Fixture helpers

    /// Minimal latent: [1, 4, 4, 4] — 4×4 spatial, 4 channels.
    /// gridH = 4/2 = 2, gridW = 4/2 = 2. Matches Sortie 7a's pattern.
    private static func makeSyntheticInput() -> BackboneInput {
        BackboneInput(
            latents: MLXArray.zeros([1, 4, 4, 4]),
            conditioning: MLXArray.zeros([1, 120, 4096]),
            conditioningMask: MLXArray.ones([1, 120]),
            timestep: MLXArray([500 as Float])
        )
    }

    /// Creates an isolated PixArtDiT. Each test gets its own instance so
    /// telemetry events from other suites cannot contaminate the reporter log.
    private static func makeFreshDiT() throws -> PixArtDiT {
        try PixArtDiT(configuration: PixArtDiTConfiguration())
    }

    // MARK: - Event presence and cardinality

    @Test("varianceChannelsDiscarded fires exactly once per forward call")
    func varianceChannelsDiscardedFiresExactlyOnce() async throws {
        let dit = try Self.makeFreshDiT()
        let reporter = MockReporter()
        dit.setTelemetry(reporter)

        let input = Self.makeSyntheticInput()
        let output = try dit.forward(input)
        eval(output)

        // Strategy A: 100 ms to allow fire-and-forget Task to deliver events.
        try await Task.sleep(nanoseconds: 100_000_000)
        let events = await reporter.snapshot()

        let discardEvents = events.filter { event -> Bool in
            if case .varianceChannelsDiscarded = event { return true }
            return false
        }
        #expect(
            discardEvents.count == 1,
            "Expected exactly one .varianceChannelsDiscarded event; got \(discardEvents.count) in \(events.count) total events"
        )
    }

    // MARK: - beforeChannels and afterChannels

    @Test("varianceChannelsDiscarded carries beforeChannels=8 and afterChannels=4")
    func varianceChannelsDiscardedChannelCounts() async throws {
        let dit = try Self.makeFreshDiT()
        let reporter = MockReporter()
        dit.setTelemetry(reporter)

        let input = Self.makeSyntheticInput()
        let output = try dit.forward(input)
        eval(output)

        try await Task.sleep(nanoseconds: 100_000_000)
        let events = await reporter.snapshot()

        let matchingEvents = events.filter { event -> Bool in
            if case .varianceChannelsDiscarded(let before, let after, _, _) = event {
                return before == 8 && after == 4
            }
            return false
        }
        #expect(
            matchingEvents.count == 1,
            "Expected .varianceChannelsDiscarded(beforeChannels:8, afterChannels:4); got \(matchingEvents.count) matching events"
        )
    }

    // MARK: - beforeStat shape

    @Test("varianceChannelsDiscarded.beforeStat has shape [1, 4, 4, 8] (8-channel pre-slice)")
    func varianceChannelsDiscardedBeforeStatShape() async throws {
        let dit = try Self.makeFreshDiT()
        let reporter = MockReporter()
        dit.setTelemetry(reporter)

        let input = Self.makeSyntheticInput()
        let output = try dit.forward(input)
        eval(output)

        try await Task.sleep(nanoseconds: 100_000_000)
        let events = await reporter.snapshot()

        // Extract the first (and only) varianceChannelsDiscarded event.
        var beforeStat: TuberiaTensorStat? = nil
        for event in events {
            if case .varianceChannelsDiscarded(_, _, let b, _) = event {
                beforeStat = b
                break
            }
        }

        #expect(beforeStat != nil, "No .varianceChannelsDiscarded event found in log")

        if let stat = beforeStat {
            // FinalLayer unpatchify output for [1,4,4,4] latent with patchSize=2, outChannels=8:
            //   [B=1, gridH*patchSize=4, gridW*patchSize=4, outChannels=8] = [1, 4, 4, 8]
            #expect(
                stat.shape == [1, 4, 4, 8],
                "beforeStat.shape should be [1, 4, 4, 8] (8-channel pre-slice); got \(stat.shape)"
            )
        }
    }

    // MARK: - afterStat shape

    @Test("varianceChannelsDiscarded.afterStat has shape [1, 4, 4, 4] (4-channel post-slice)")
    func varianceChannelsDiscardedAfterStatShape() async throws {
        let dit = try Self.makeFreshDiT()
        let reporter = MockReporter()
        dit.setTelemetry(reporter)

        let input = Self.makeSyntheticInput()
        let output = try dit.forward(input)
        eval(output)

        try await Task.sleep(nanoseconds: 100_000_000)
        let events = await reporter.snapshot()

        var afterStat: TuberiaTensorStat? = nil
        for event in events {
            if case .varianceChannelsDiscarded(_, _, _, let a) = event {
                afterStat = a
                break
            }
        }

        #expect(afterStat != nil, "No .varianceChannelsDiscarded event found in log")

        if let stat = afterStat {
            // After slice: [1, 4, 4, 4] (first 4 channels only)
            #expect(
                stat.shape == [1, 4, 4, 4],
                "afterStat.shape should be [1, 4, 4, 4] (4-channel post-slice); got \(stat.shape)"
            )
        }
    }

    // MARK: - Relative stat relationship

    @Test("beforeStat has more elements than afterStat (8-channel vs 4-channel)")
    func varianceChannelsDiscardedBeforeHasMoreElementsThanAfter() async throws {
        let dit = try Self.makeFreshDiT()
        let reporter = MockReporter()
        dit.setTelemetry(reporter)

        let input = Self.makeSyntheticInput()
        let output = try dit.forward(input)
        eval(output)

        try await Task.sleep(nanoseconds: 100_000_000)
        let events = await reporter.snapshot()

        var foundEvent = false
        for event in events {
            if case .varianceChannelsDiscarded(let before, let after, let beforeStat, let afterStat) = event {
                foundEvent = true

                let beforeCount = beforeStat.shape.reduce(1, *)
                let afterCount = afterStat.shape.reduce(1, *)

                #expect(
                    before == 8,
                    "beforeChannels should be 8; got \(before)"
                )
                #expect(
                    after == 4,
                    "afterChannels should be 4; got \(after)"
                )
                // The 8-channel tensor has exactly twice as many elements as the 4-channel tensor.
                #expect(
                    beforeCount == afterCount * 2,
                    "beforeStat element count (\(beforeCount)) should be 2× afterStat count (\(afterCount))"
                )
                break
            }
        }

        #expect(foundEvent, "No .varianceChannelsDiscarded event found in log")
    }
}
