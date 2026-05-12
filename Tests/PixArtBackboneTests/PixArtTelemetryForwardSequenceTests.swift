import Foundation
import MLX
import Testing
import Tuberia

@testable import PixArtBackbone

// MARK: - Forward-Pass Telemetry Sequence Tests
//
// Verifies that a single `PixArtDiT.forward(_:)` call produces the correct
// event sequence and that each event fires exactly once.
//
// Expected event sequence (in order):
//   1. ditForwardStart
//   2. patchEmbedComplete
//   3. captionProjectionComplete
//   4. timestepEmbeddingComplete
//   5. siluWorkaroundExecuted
//   6. finalLayerComplete
//   7. varianceChannelsDiscarded
//   8. ditForwardComplete
//
// The most important invariant: `ditForwardComplete` fires EXACTLY ONCE per
// scheduler step — never per DiT block (there are 28 of them).
//
// After `forward(_:)` returns, events are dispatched via a fire-and-forget
// Task.  Tests sleep 100 ms (Strategy A) before snapshotting the reporter log.

@Suite("PixArtTelemetryForwardSequence", .serialized)
struct PixArtTelemetryForwardSequenceTests {

    // MARK: - Shared synthetic input

    /// Minimal latent: [1, 4, 4, 4] — 4×4 spatial, 4 channels.
    /// gridH = 4/2 = 2, gridW = 4/2 = 2 → 4 tokens through all 28 DiT blocks.
    /// This matches the fixture pattern used in BackboneForwardTests.
    private static func makeSyntheticInput() -> BackboneInput {
        BackboneInput(
            latents: MLXArray.zeros([1, 4, 4, 4]),
            conditioning: MLXArray.zeros([1, 120, 4096]),
            conditioningMask: MLXArray.ones([1, 120]),
            timestep: MLXArray([500 as Float])
        )
    }

        // MARK: - Isolated DiT fixture for telemetry tests
    //
    // IMPORTANT: These tests use an isolated PixArtDiT instance (NOT BackboneFixture.dit)
    // to avoid event contamination from other test suites that also call forward() on
    // the shared BackboneFixture.dit.  BackboneForwardTests is not .serialized with this
    // suite, so if we shared the fixture, concurrent forward() calls from other suites
    // would install extra events into our MockReporter via the shared telemetry lock.
    //
    // Cost: each test in this suite pays a PixArtDiT init (~600M params allocated).
    // Mitigation: we use the same minimal configuration as BackboneFixture (patchSize=2,
    // depth=28) and the minimal [1,4,4,4] latent so the forward pass is fast.

    private static func makeFreshDiT() throws -> PixArtDiT {
        try PixArtDiT(configuration: PixArtDiTConfiguration())
    }

    // MARK: - ditForwardComplete cardinality (the headline invariant)

    @Test("ditForwardComplete fires exactly once per forward call (not per DiT block)")
    func ditForwardCompleteCardinalityIsOne() async throws {
        let dit = try Self.makeFreshDiT()
        let reporter = MockReporter()
        dit.setTelemetry(reporter)

        let input = Self.makeSyntheticInput()
        let output = try dit.forward(input)
        eval(output)

        // Strategy A: 100 ms to allow fire-and-forget Task to deliver events.
        try await Task.sleep(nanoseconds: 100_000_000)
        let events = await reporter.snapshot()

        let ditForwardCompleteCount = events.filter { event -> Bool in
            if case .ditForwardComplete = event { return true }
            return false
        }.count
        #expect(
            ditForwardCompleteCount == 1,
            "ditForwardComplete must fire EXACTLY once per scheduler step (not per DiT block); got \(ditForwardCompleteCount) in \(events.count) total events"
        )
    }

    // MARK: - Individual event cardinality (each fires exactly once)

    @Test("Each forward-pass event fires exactly once per forward call")
    func eachForwardEventFiresExactlyOnce() async throws {
        let dit = try Self.makeFreshDiT()
        let reporter = MockReporter()
        dit.setTelemetry(reporter)

        let input = Self.makeSyntheticInput()
        let output = try dit.forward(input)
        eval(output)

        try await Task.sleep(nanoseconds: 100_000_000)
        let events = await reporter.snapshot()

        let startCount = events.filter { if case .ditForwardStart = $0 { return true }; return false }.count
        let patchCount = events.filter { if case .patchEmbedComplete = $0 { return true }; return false }.count
        let captionCount = events.filter { if case .captionProjectionComplete = $0 { return true }; return false }.count
        let timestepCount = events.filter { if case .timestepEmbeddingComplete = $0 { return true }; return false }.count
        let siluCount = events.filter { if case .siluWorkaroundExecuted = $0 { return true }; return false }.count
        let finalCount = events.filter { if case .finalLayerComplete = $0 { return true }; return false }.count
        let varianceCount = events.filter { if case .varianceChannelsDiscarded = $0 { return true }; return false }.count
        let completeCount = events.filter { if case .ditForwardComplete = $0 { return true }; return false }.count

        #expect(startCount == 1, "Expected 1 ditForwardStart, got \(startCount)")
        #expect(patchCount == 1, "Expected 1 patchEmbedComplete, got \(patchCount)")
        #expect(captionCount == 1, "Expected 1 captionProjectionComplete, got \(captionCount)")
        #expect(timestepCount == 1, "Expected 1 timestepEmbeddingComplete, got \(timestepCount)")
        #expect(siluCount == 1, "Expected 1 siluWorkaroundExecuted, got \(siluCount)")
        #expect(finalCount == 1, "Expected 1 finalLayerComplete, got \(finalCount)")
        #expect(varianceCount == 1, "Expected 1 varianceChannelsDiscarded, got \(varianceCount)")
        #expect(completeCount == 1, "Expected 1 ditForwardComplete, got \(completeCount)")
    }

    // MARK: - Event ordering (the eight-event sequence)

    @Test("Forward-pass events arrive in the canonical sequence")
    func forwardEventOrdering() async throws {
        let dit = try Self.makeFreshDiT()
        let reporter = MockReporter()
        dit.setTelemetry(reporter)

        let input = Self.makeSyntheticInput()
        let output = try dit.forward(input)
        eval(output)

        try await Task.sleep(nanoseconds: 100_000_000)
        let events = await reporter.snapshot()

        // Locate each event type by its first occurrence in the log.
        let idxStart = events.firstIndex { if case .ditForwardStart = $0 { return true }; return false }
        let idxPatch = events.firstIndex { if case .patchEmbedComplete = $0 { return true }; return false }
        let idxCaption = events.firstIndex { if case .captionProjectionComplete = $0 { return true }; return false }
        let idxTimestep = events.firstIndex { if case .timestepEmbeddingComplete = $0 { return true }; return false }
        let idxSilu = events.firstIndex { if case .siluWorkaroundExecuted = $0 { return true }; return false }
        let idxFinal = events.firstIndex { if case .finalLayerComplete = $0 { return true }; return false }
        let idxVariance = events.firstIndex { if case .varianceChannelsDiscarded = $0 { return true }; return false }
        let idxComplete = events.firstIndex { if case .ditForwardComplete = $0 { return true }; return false }

        // All eight events must be present.
        #expect(idxStart != nil, "ditForwardStart not found in event log")
        #expect(idxPatch != nil, "patchEmbedComplete not found in event log")
        #expect(idxCaption != nil, "captionProjectionComplete not found in event log")
        #expect(idxTimestep != nil, "timestepEmbeddingComplete not found in event log")
        #expect(idxSilu != nil, "siluWorkaroundExecuted not found in event log")
        #expect(idxFinal != nil, "finalLayerComplete not found in event log")
        #expect(idxVariance != nil, "varianceChannelsDiscarded not found in event log")
        #expect(idxComplete != nil, "ditForwardComplete not found in event log")

        guard let s = idxStart, let p = idxPatch, let c = idxCaption,
              let ts = idxTimestep, let si = idxSilu, let fi = idxFinal,
              let v = idxVariance, let done = idxComplete else {
            return  // Already reported missing events above via #expect.
        }

        // Verify canonical ordering:
        // ditForwardStart → patchEmbedComplete → captionProjectionComplete →
        // timestepEmbeddingComplete → siluWorkaroundExecuted →
        // finalLayerComplete → varianceChannelsDiscarded → ditForwardComplete
        #expect(s < p, "ditForwardStart must precede patchEmbedComplete (indices: \(s), \(p))")
        #expect(p < c, "patchEmbedComplete must precede captionProjectionComplete (indices: \(p), \(c))")
        #expect(c < ts, "captionProjectionComplete must precede timestepEmbeddingComplete (indices: \(c), \(ts))")
        #expect(ts < si, "timestepEmbeddingComplete must precede siluWorkaroundExecuted (indices: \(ts), \(si))")
        #expect(si < fi, "siluWorkaroundExecuted must precede finalLayerComplete (indices: \(si), \(fi))")
        #expect(fi < v, "finalLayerComplete must precede varianceChannelsDiscarded (indices: \(fi), \(v))")
        #expect(v < done, "varianceChannelsDiscarded must precede ditForwardComplete (indices: \(v), \(done))")
    }

    // MARK: - No telemetry when reporter is nil (no crash)

    @Test("forward produces no events and does not crash when reporter is nil")
    func forwardNoEventsWhenReporterNil() throws {
        // Use a fresh isolated DiT (not BackboneFixture.dit) to avoid installing or
        // clearing state on the shared fixture.
        let dit = try Self.makeFreshDiT()
        // No setTelemetry call — reporter stays nil by default.

        let input = Self.makeSyntheticInput()
        let output = try dit.forward(input)
        eval(output)

        // If we get here without crashing, the guard against nil-reporter is working.
        #expect(output.shape == [1, 4, 4, 4])
    }
}
