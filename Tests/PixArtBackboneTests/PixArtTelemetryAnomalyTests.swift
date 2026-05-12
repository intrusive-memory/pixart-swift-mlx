import Foundation
import MLX
import Testing
import Tuberia

@testable import PixArtBackbone

// MARK: - Numerical Anomaly Telemetry Tests
//
// Verifies that `PixArtDiT.forward(_:)` emits `.numericalAnomaly` events when a
// NaN-poisoned input latent is supplied, and that no anomaly events fire when the
// input is clean.
//
// ## Approach A — NaN-poisoned input latent
//
// The forward path samples the input latent at the `ditForwardStart` emission site:
//
//   let inputLatentStat = TuberiaTensorStat.sample(latents)
//   if inputLatentStat.hasNaN || inputLatentStat.hasInf {
//       pendingEvents.append(.numericalAnomaly(
//           phase: "pixart_dit_forward_start_input_latent",
//           kind: inputLatentStat.hasNaN ? .nan : .inf,
//           stepIndex: nil,
//           stat: inputLatentStat))
//   }
//
// Injecting `Float.nan` into the latent tensor is sufficient to trigger this path
// without any subclassing or source modification.
//
// Phase string reference: `PixArtDiT.swift`, the `ditForwardStart` emission block.
//
// IMPORTANT: A fresh `PixArtDiT` instance is used per test (NOT `BackboneFixture.dit`)
// to avoid event contamination from concurrent `forward(_:)` calls in other test suites.
//
// After `forward(_:)` returns, events are dispatched via a fire-and-forget Task.
// Tests sleep 100 ms (Strategy A) before snapshotting the reporter log.

@Suite("PixArtTelemetryAnomaly", .serialized)
struct PixArtTelemetryAnomalyTests {

    // MARK: - Fixture helpers

    /// Normal (clean) synthetic input: all zeros, shape [1, 4, 4, 4].
    private static func makeCleanInput() -> BackboneInput {
        BackboneInput(
            latents: MLXArray.zeros([1, 4, 4, 4]),
            conditioning: MLXArray.zeros([1, 120, 4096]),
            conditioningMask: MLXArray.ones([1, 120]),
            timestep: MLXArray([500 as Float])
        )
    }

    /// NaN-poisoned latent input: a [1, 4, 4, 4] tensor where one entry is Float.nan.
    ///
    /// Construction: start with a zeros array and scatter one NaN via concatenation so
    /// `TuberiaTensorStat.sample(latents)` returns `hasNaN == true`. This causes
    /// `PixArtDiT.forward(_:)` to emit `.numericalAnomaly(phase: "pixart_dit_forward_start_input_latent", kind: .nan, ...)`.
    private static func makeNaNPoisonedInput() -> BackboneInput {
        // Build a flat [64] float32 array with one NaN at index 0.
        let nanValue: Float = .nan
        var values = [Float](repeating: 0.0, count: 1 * 4 * 4 * 4)
        values[0] = nanValue
        let poisonedLatents = MLXArray(values).reshaped([1, 4, 4, 4])
        eval(poisonedLatents)

        return BackboneInput(
            latents: poisonedLatents,
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

    // MARK: - NaN anomaly: at least one event fires

    @Test("NaN-poisoned latent causes at least one .numericalAnomaly event")
    func nanPoisonedLatentEmitsNumericalAnomaly() async throws {
        let dit = try Self.makeFreshDiT()
        let reporter = MockReporter()
        dit.setTelemetry(reporter)

        let input = Self.makeNaNPoisonedInput()
        let output = try dit.forward(input)
        eval(output)

        // Strategy A: 100 ms to allow fire-and-forget Task to deliver events.
        try await Task.sleep(nanoseconds: 100_000_000)
        let events = await reporter.snapshot()

        let anomalyEvents = events.filter { event -> Bool in
            if case .numericalAnomaly = event { return true }
            return false
        }
        #expect(
            anomalyEvents.count >= 1,
            "Expected at least one .numericalAnomaly event for NaN-poisoned input; got \(anomalyEvents.count) in \(events.count) total events"
        )
    }

    // MARK: - NaN anomaly: kind is .nan

    @Test("NaN-poisoned latent anomaly event has kind == .nan")
    func nanPoisonedLatentAnomalyKindIsNaN() async throws {
        let dit = try Self.makeFreshDiT()
        let reporter = MockReporter()
        dit.setTelemetry(reporter)

        let input = Self.makeNaNPoisonedInput()
        let output = try dit.forward(input)
        eval(output)

        try await Task.sleep(nanoseconds: 100_000_000)
        let events = await reporter.snapshot()

        let nanAnomalyEvents = events.filter { event -> Bool in
            if case .numericalAnomaly(_, let kind, _, _) = event {
                return kind == .nan
            }
            return false
        }
        #expect(
            nanAnomalyEvents.count >= 1,
            "Expected at least one .numericalAnomaly with kind=.nan; got \(nanAnomalyEvents.count)"
        )
    }

    // MARK: - NaN anomaly: phase matches the input-latent emission site

    @Test("NaN anomaly event for input latent uses phase 'pixart_dit_forward_start_input_latent'")
    func nanPoisonedLatentAnomalyPhaseMatchesInputLatent() async throws {
        let dit = try Self.makeFreshDiT()
        let reporter = MockReporter()
        dit.setTelemetry(reporter)

        let input = Self.makeNaNPoisonedInput()
        let output = try dit.forward(input)
        eval(output)

        try await Task.sleep(nanoseconds: 100_000_000)
        let events = await reporter.snapshot()

        // The input-latent sampling in PixArtDiT.swift emits this exact phase string.
        // See the `ditForwardStart` emission block in PixArtDiT.forward(_:).
        let expectedPhase = "pixart_dit_forward_start_input_latent"
        let matchingEvents = events.filter { event -> Bool in
            if case .numericalAnomaly(let phase, let kind, _, _) = event {
                return phase == expectedPhase && kind == .nan
            }
            return false
        }
        #expect(
            matchingEvents.count >= 1,
            "Expected at least one .numericalAnomaly(phase: \"\(expectedPhase)\", kind: .nan); got \(matchingEvents.count)"
        )
    }

    // MARK: - NaN anomaly: stat.hasNaN is true

    @Test("NaN anomaly event carries a stat with hasNaN == true")
    func nanPoisonedLatentAnomalyStatHasNaN() async throws {
        let dit = try Self.makeFreshDiT()
        let reporter = MockReporter()
        dit.setTelemetry(reporter)

        let input = Self.makeNaNPoisonedInput()
        let output = try dit.forward(input)
        eval(output)

        try await Task.sleep(nanoseconds: 100_000_000)
        let events = await reporter.snapshot()

        var foundStatWithNaN = false
        for event in events {
            if case .numericalAnomaly(_, let kind, _, let stat) = event,
               kind == .nan,
               stat.hasNaN == true
            {
                foundStatWithNaN = true
                break
            }
        }
        #expect(
            foundStatWithNaN,
            "Expected a .numericalAnomaly event with kind=.nan and stat.hasNaN==true"
        )
    }

    // MARK: - Positive case: clean input emits zero anomaly events

    @Test("Clean input latent emits zero .numericalAnomaly events")
    func cleanInputEmitsNoNumericalAnomalyEvents() async throws {
        let dit = try Self.makeFreshDiT()
        let reporter = MockReporter()
        dit.setTelemetry(reporter)

        let input = Self.makeCleanInput()
        let output = try dit.forward(input)
        eval(output)

        // Strategy A: 100 ms to allow fire-and-forget Task to deliver events.
        try await Task.sleep(nanoseconds: 100_000_000)
        let events = await reporter.snapshot()

        let anomalyEvents = events.filter { event -> Bool in
            if case .numericalAnomaly = event { return true }
            return false
        }
        #expect(
            anomalyEvents.count == 0,
            "Expected zero .numericalAnomaly events for a clean input; got \(anomalyEvents.count) in \(events.count) total events"
        )
    }
}
