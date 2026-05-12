import Foundation
@testable import PixArtBackbone

/// Test-only reporter that captures every PixArtTelemetryEvent into an
/// append-only log. Not part of the public API.
///
/// The append-only log is implemented as an actor so concurrent capture
/// calls from the fire-and-forget Task in `PixArtDiT.forward(_:)` and
/// `PixArtDiT.apply(weights:)` are serialized without data races.
///
/// ## Waiting for fire-and-forget dispatch
///
/// Both `forward(_:)` and `apply(weights:)` dispatch telemetry events via a
/// single unstructured `Task { ... }` at the end of the synchronous call.
/// After the synchronous call returns the Task may not yet have delivered all
/// events to this actor.  Tests use Strategy A: a fixed 100 ms sleep before
/// snapshotting the log.  Bump to 250 ms if flakiness is observed on slow CI
/// hardware.
///
///     try await Task.sleep(nanoseconds: 100_000_000)
///     let events = await reporter.snapshot()
actor MockReporter: PixArtTelemetryReporter {
    private(set) var events: [PixArtTelemetryEvent] = []

    func capture(_ event: PixArtTelemetryEvent) async {
        events.append(event)
    }

    func snapshot() async -> [PixArtTelemetryEvent] {
        events
    }

    func clear() async {
        events.removeAll()
    }
}
