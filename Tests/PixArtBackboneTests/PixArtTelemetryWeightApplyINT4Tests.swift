import Foundation
import MLX
import Testing
import Tuberia

@testable import PixArtBackbone

/// Verifies `PixArtDiT.apply(weights:)` emits a single boundary event when given a
/// synthetic INT4-quantized weight dictionary.
///
/// INT4 dictionary shape: `<key>.weight` (uint32 packed) + `<key>.scales` (fp16) +
/// `<key>.biases` (fp16). Only the `.weight` key counts toward `paramCount`; the
/// sidecars are consumed silently during dequantization.
@Suite("PixArtTelemetryWeightApplyINT4", .serialized)
struct PixArtTelemetryWeightApplyINT4Tests {

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

  @Test("INT4 apply emits exactly one weightLoadComplete(component: .dit) with paramCount==1")
  func int4ApplyEmitsSingleWeightLoadComplete() async throws {
    let dit = try PixArtDiT(configuration: PixArtDiTConfiguration())
    let reporter = MockReporter()
    dit.setTelemetry(reporter)

    let params = Self.makeINT4Params()
    try dit.apply(weights: params)

    try await Task.sleep(nanoseconds: 100_000_000)
    let events = await reporter.snapshot()

    let completeEvents = events.compactMap {
      event -> (PixArtTelemetryEvent.WeightComponent, Int, Double)? in
      if case .weightLoadComplete(let component, let paramCount, let duration) = event {
        return (component, paramCount, duration)
      }
      return nil
    }
    #expect(
      completeEvents.count == 1,
      "Expected exactly one weightLoadComplete; got \(completeEvents.count) in \(events)")
    if let (component, paramCount, duration) = completeEvents.first {
      #expect(component == .dit)
      #expect(paramCount == 1, "Sidecar .scales/.biases must not count toward paramCount")
      #expect(duration >= 0)
    }
  }

  @Test("INT4 apply emits no other event cases")
  func int4ApplyEmitsNothingElse() async throws {
    let dit = try PixArtDiT(configuration: PixArtDiTConfiguration())
    let reporter = MockReporter()
    dit.setTelemetry(reporter)

    try dit.apply(weights: Self.makeINT4Params())

    try await Task.sleep(nanoseconds: 100_000_000)
    let events = await reporter.snapshot()

    // Boundary-only surface: a successful apply emits one event, nothing else.
    #expect(
      events.count == 1,
      "Expected exactly one event from apply(weights:); got \(events.count): \(events)")
  }
}
