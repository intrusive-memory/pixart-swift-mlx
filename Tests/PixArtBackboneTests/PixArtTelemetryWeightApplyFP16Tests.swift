import Foundation
import MLX
import Testing
import Tuberia

@testable import PixArtBackbone

/// Verifies `PixArtDiT.apply(weights:)` emits a single boundary event when given a
/// synthetic FP16 weight dictionary (no INT4 sidecars).
///
/// In the slim surface there is no quantization counter event — `weightLoadComplete`
/// just carries component + paramCount + duration. Quantization variants of the
/// same fixture are expected to produce the same boundary event shape.
@Suite("PixArtTelemetryWeightApplyFP16", .serialized)
struct PixArtTelemetryWeightApplyFP16Tests {

  private static func makeFP16Params(key: String = "foo") -> Tuberia.ModuleParameters {
    let weight = MLXArray([1.0, 2.0, 3.0, 4.0] as [Float]).asType(.float16)
    return Tuberia.ModuleParameters(parameters: [
      "\(key).weight": weight
    ])
  }

  @Test("FP16 apply emits exactly one weightLoadComplete(component: .dit) with paramCount==1")
  func fp16ApplyEmitsSingleWeightLoadComplete() async throws {
    let dit = try PixArtDiT(configuration: PixArtDiTConfiguration())
    let reporter = MockReporter()
    dit.setTelemetry(reporter)

    try dit.apply(weights: Self.makeFP16Params())

    try await Task.sleep(nanoseconds: 100_000_000)
    let events = await reporter.snapshot()

    let completeEvents = events.compactMap {
      event -> (PixArtTelemetryEvent.WeightComponent, Int)? in
      if case .weightLoadComplete(let component, let paramCount, _) = event {
        return (component, paramCount)
      }
      return nil
    }
    #expect(
      completeEvents.count == 1,
      "Expected exactly one weightLoadComplete; got \(completeEvents.count) in \(events)")
    if let (component, paramCount) = completeEvents.first {
      #expect(component == .dit)
      #expect(paramCount == 1)
    }
  }

  @Test("FP16 apply emits no events when reporter is nil (no crash)")
  func fp16ApplyNoEventsWhenReporterNil() async throws {
    let dit = try PixArtDiT(configuration: PixArtDiTConfiguration())
    try dit.apply(weights: Self.makeFP16Params())
    #expect(dit.isLoaded == true)
  }

  @Test("unload() emits exactly one weightUnloadComplete")
  func unloadEmitsWeightUnloadComplete() async throws {
    let dit = try PixArtDiT(configuration: PixArtDiTConfiguration())
    let reporter = MockReporter()
    dit.setTelemetry(reporter)

    try dit.apply(weights: Self.makeFP16Params())
    try await Task.sleep(nanoseconds: 100_000_000)
    await reporter.clear()

    dit.unload()
    try await Task.sleep(nanoseconds: 100_000_000)
    let events = await reporter.snapshot()

    let unloadCount = events.filter {
      if case .weightUnloadComplete = $0 { return true }
      return false
    }.count
    #expect(unloadCount == 1, "Expected one weightUnloadComplete; got \(unloadCount) in \(events)")
    #expect(dit.isLoaded == false)
  }
}
