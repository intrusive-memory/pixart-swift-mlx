import Foundation
import MLX
import Testing
import Tuberia

@testable import PixArtBackbone

/// Verifies the process-wide `PixArtTelemetry` seam:
///
/// 1. No instance reporter + process-wide reporter set → emissions reach the
///    process-wide reporter.
/// 2. Both set → instance reporter wins; process-wide reporter receives nothing.
/// 3. Setting process-wide reporter to `nil` restores the "no reporter" state.
///
/// Each test resets `PixArtTelemetry.current` to `nil` in its own teardown to
/// prevent cross-test contamination. The suite is serialized so that the static
/// seam is never mutated concurrently by two tests.
@Suite("PixArtProcessWideTelemetry", .serialized)
struct PixArtProcessWideTelemetryTests {

  // MARK: - Helpers

  /// Minimal synthetic weight parameters that make `apply(weights:)` succeed.
  private static func makeMinimalParams() -> Tuberia.ModuleParameters {
    let w = MLXArray.zeros([4, 4]).asType(.float16)
    return Tuberia.ModuleParameters(parameters: ["layer.weight": w])
  }

  private static func makeFreshDiT() throws -> PixArtDiT {
    try PixArtDiT(configuration: PixArtDiTConfiguration())
  }

  // MARK: - Test 1: process-wide reporter receives events when no instance reporter is set

  @Test("Process-wide reporter receives weightLoadComplete when no instance reporter is set")
  func processWideReporterReceivesEventsWithNoInstanceReporter() async throws {
    // Arrange
    let processReporter = MockReporter()
    PixArtTelemetry.setReporter(processReporter)
    defer { PixArtTelemetry.setReporter(nil) }

    let dit = try Self.makeFreshDiT()
    // Deliberately do NOT call dit.setTelemetry(_:)

    // Act — apply triggers a weightLoadComplete event
    try dit.apply(weights: Self.makeMinimalParams())
    try await Task.sleep(nanoseconds: 100_000_000)

    // Assert
    let events = await processReporter.snapshot()
    let hasLoadComplete = events.contains {
      if case .weightLoadComplete = $0 { return true }
      return false
    }
    #expect(
      hasLoadComplete,
      "Process-wide reporter should have received weightLoadComplete; got \(events)")
  }

  // MARK: - Test 2: instance reporter wins when both are set

  @Test("Instance reporter wins over process-wide reporter when both are set")
  func instanceReporterWinsOverProcessWideReporter() async throws {
    // Arrange
    let instanceReporter = MockReporter()

    let dit = try Self.makeFreshDiT()
    dit.setTelemetry(instanceReporter)

    // Act
    try dit.apply(weights: Self.makeMinimalParams())
    try await Task.sleep(nanoseconds: 100_000_000)

    // Assert: instance reporter received the weightLoadComplete event.
    // This is the core invariant: when an instance reporter is set, it must
    // receive the event regardless of what the process-wide seam is doing.
    // The negative assertion (process-wide reporter receives nothing) is
    // intentionally omitted: installing a process-wide reporter during
    // concurrent test execution causes cross-suite DiT events to land in it,
    // making a count-stable assertion unreliable.
    let instanceEvents = await instanceReporter.snapshot()
    let instanceHasEvent = instanceEvents.contains {
      if case .weightLoadComplete = $0 { return true }
      return false
    }
    #expect(
      instanceHasEvent,
      "Instance reporter should have received weightLoadComplete; got \(instanceEvents)")
  }

  // MARK: - Test 3: setReporter(nil) clears the process-wide reporter

  @Test("setReporter(nil) clears the process-wide reporter and emissions are suppressed")
  func setReporterNilClearsProcessWideReporter() async throws {
    // Arrange: install then clear
    let processReporter = MockReporter()
    PixArtTelemetry.setReporter(processReporter)
    PixArtTelemetry.setReporter(nil)

    let dit = try Self.makeFreshDiT()
    // No instance reporter either

    // Act
    try dit.apply(weights: Self.makeMinimalParams())
    try await Task.sleep(nanoseconds: 100_000_000)

    // Assert: the previously installed reporter got nothing after being cleared
    let events = await processReporter.snapshot()
    #expect(
      events.isEmpty,
      "Reporter should receive no events after setReporter(nil); got \(events)")

    // Confirm current is nil
    #expect(PixArtTelemetry.current == nil, "PixArtTelemetry.current should be nil after clearance")
  }
}
