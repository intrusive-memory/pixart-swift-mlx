import Foundation
import MLX
import Testing
import Tuberia

@testable import PixArtBackbone

/// Verifies that `PixArtDiT.forward(_:)` emits `numericalAnomaly(phase: .ditForward)`
/// when the forward output is numerically bad, and emits nothing when the output is
/// clean.
///
/// In the slim surface there is **no happy-path forward event** — the only signal
/// PixArt sends from the forward path is a side-channel anomaly. A NaN injected into
/// the input latent propagates through the model and shows up in the output stat,
/// where the exit-time check catches it.
///
/// A fresh `PixArtDiT` instance per test prevents event contamination from other
/// suites that exercise the forward path concurrently.
@Suite("PixArtTelemetryAnomaly", .serialized)
struct PixArtTelemetryAnomalyTests {

  private static func makeCleanInput() -> BackboneInput {
    BackboneInput(
      latents: MLXArray.zeros([1, 4, 4, 4]),
      conditioning: MLXArray.zeros([1, 120, 4096]),
      conditioningMask: MLXArray.ones([1, 120]),
      timestep: MLXArray([500 as Float])
    )
  }

  private static func makeNaNPoisonedInput() -> BackboneInput {
    var values = [Float](repeating: 0.0, count: 1 * 4 * 4 * 4)
    values[0] = .nan
    let poisonedLatents = MLXArray(values).reshaped([1, 4, 4, 4])
    eval(poisonedLatents)

    return BackboneInput(
      latents: poisonedLatents,
      conditioning: MLXArray.zeros([1, 120, 4096]),
      conditioningMask: MLXArray.ones([1, 120]),
      timestep: MLXArray([500 as Float])
    )
  }

  private static func makeFreshDiT() throws -> PixArtDiT {
    try PixArtDiT(configuration: PixArtDiTConfiguration())
  }

  @Test("NaN-poisoned latent emits exactly one numericalAnomaly(phase: .ditForward, kind: .nan)")
  func nanPoisonedLatentEmitsAnomaly() async throws {
    let dit = try Self.makeFreshDiT()
    let reporter = MockReporter()
    dit.setTelemetry(reporter)

    let output = try dit.forward(Self.makeNaNPoisonedInput())
    eval(output)

    try await Task.sleep(nanoseconds: 100_000_000)
    let events = await reporter.snapshot()

    let anomalies = events.compactMap {
      event -> (
        PixArtTelemetryEvent.AnomalyPhase, PixArtTelemetryEvent.AnomalyKind, TuberiaTensorStat
      )? in
      if case .numericalAnomaly(let phase, let kind, let stat) = event {
        return (phase, kind, stat)
      }
      return nil
    }
    #expect(
      anomalies.count == 1,
      "Expected exactly one numericalAnomaly; got \(anomalies.count) in \(events)")
    if let (phase, kind, stat) = anomalies.first {
      #expect(phase == .ditForward)
      #expect(kind == .nan)
      #expect(stat.hasNaN == true)
    }
  }

  @Test("Clean input emits zero events from forward (boundary-only surface)")
  func cleanInputEmitsNoEvents() async throws {
    let dit = try Self.makeFreshDiT()
    let reporter = MockReporter()
    dit.setTelemetry(reporter)

    let output = try dit.forward(Self.makeCleanInput())
    eval(output)

    try await Task.sleep(nanoseconds: 100_000_000)
    let events = await reporter.snapshot()

    #expect(
      events.isEmpty,
      "Expected forward(_:) with clean input to emit zero events; got \(events.count): \(events)")
  }
}
