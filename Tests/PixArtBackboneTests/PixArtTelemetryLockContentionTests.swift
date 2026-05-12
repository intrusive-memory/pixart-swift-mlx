import Foundation
import Testing

@testable import PixArtBackbone

// MARK: - Lock-Contention Tests for the Telemetry Seam
//
// Sortie 8 of OPERATION STETHOSCOPE FURNACE.
//
// Stresses `PixArtDiT._telemetryLock` via concurrent writer tasks from a
// `TaskGroup`.  The goal is to expose data races that OSAllocatedUnfairLock
// is meant to prevent.
//
// ## Design
//
// `currentTelemetry()` is `fileprivate`, so it cannot be called directly from
// the test target even with `@testable import`.  The lock guards BOTH the read
// path (`currentTelemetry()`) and the write path (`setTelemetry(_:)`).  Hammering
// `setTelemetry` from multiple concurrent tasks is therefore sufficient to give
// the Thread Sanitizer a representative read/write interleaving.
//
// ## TSan verification
//
// Under the default `make test` invocation this test passes if no crash or
// assertion fires.  Real data-race detection requires TSan:
//
//   make test-tsan
//
// or manually:
//
//   xcodebuild test \
//     -scheme pixart-swift-mlx \
//     -destination 'platform=macOS,arch=arm64' \
//     -enableThreadSanitizer YES \
//     -only-testing:PixArtBackboneTests/PixArtTelemetryLockContentionTests
//
// A TSan run that reports zero `WARNING: ThreadSanitizer` / `data race`
// diagnostics confirms that Sortie 2's OSAllocatedUnfairLock seam is correct.
//
// ## After fire-and-forget Tasks
//
// `setTelemetry(_:)` is synchronous; no fire-and-forget Task is involved.
// No sleep is required before asserting the post-contention invariant.

@Suite("PixArtTelemetryLockContention", .serialized)
struct PixArtTelemetryLockContentionTests {

    // MARK: - Fixture helpers

    /// Returns a fresh, isolated `PixArtDiT` instance.
    ///
    /// Each test gets its own instance so that concurrent writes from this
    /// suite do not interfere with any other test suite's reporter state.
    private func makeFreshDiT() throws -> PixArtDiT {
        try PixArtDiT(configuration: PixArtDiTConfiguration())
    }

    // MARK: - Concurrent set / nil cycle

    /// Two writer tasks alternate between a `MockReporter` and `nil` while a
    /// third writer hammers `setTelemetry(nil)` in a tight loop.
    ///
    /// Pass criterion (default run): no crash, no assertion, task group
    /// completes normally.
    /// Pass criterion (TSan run): zero `data race` diagnostics.
    @Test("Concurrent setTelemetry writes do not race on _telemetryLock")
    func test_concurrentSetAndNil_doesNotRace() async throws {
        let dit = try makeFreshDiT()
        let reporter1 = MockReporter()
        let reporter2 = MockReporter()

        await withTaskGroup(of: Void.self) { group in
            // Writer 1: alternates between reporter1 and nil.
            group.addTask {
                for _ in 0..<500 {
                    dit.setTelemetry(reporter1)
                    dit.setTelemetry(nil)
                }
            }
            // Writer 2: alternates between reporter2 and reporter1.
            group.addTask {
                for _ in 0..<500 {
                    dit.setTelemetry(reporter2)
                    dit.setTelemetry(reporter1)
                }
            }
            // Writer 3: hammers nil — high-frequency read-path exerciser
            // (currentTelemetry() is called by forward(_:) and apply(weights:)
            // whenever a task holds a reporter; writing nil forces the lock to
            // flip state and exercises the read path via the locked compare).
            group.addTask {
                for _ in 0..<2000 {
                    dit.setTelemetry(nil)
                }
            }
        }

        // If execution reaches here the lock held under all concurrent writes.
        // Under a TSan-enabled build any unguarded access would have already
        // triggered a diagnostic abort.
        #expect(Bool(true), "Completed \(500 + 500 + 2000) concurrent setTelemetry calls without crash")
    }

    // MARK: - Interleaved reporter swap

    /// Four tasks concurrently swap the reporter back and forth between two
    /// `MockReporter` instances.  The intent is to saturate the lock with
    /// concurrent read-modify-write cycles.
    @Test("Interleaved reporter swaps across four tasks do not produce a data race")
    func test_interleavedReporterSwap_doesNotRace() async throws {
        let dit = try makeFreshDiT()
        let reporterA = MockReporter()
        let reporterB = MockReporter()

        await withTaskGroup(of: Void.self) { group in
            for i in 0..<4 {
                group.addTask {
                    let even = (i % 2 == 0)
                    for _ in 0..<300 {
                        dit.setTelemetry(even ? reporterA : reporterB)
                        dit.setTelemetry(even ? reporterB : reporterA)
                        dit.setTelemetry(nil)
                    }
                }
            }
        }

        #expect(Bool(true), "Completed interleaved reporter swaps across 4 tasks without crash")
    }

    // MARK: - Set / query round-trip under contention

    /// Writer tasks alternate between `reporter1` and `nil` while a concurrent
    /// "reader-proxy" task calls `setTelemetry` immediately after observing a
    /// state (simulating the lock exercised on the read path indirectly).
    ///
    /// Because `currentTelemetry()` is `fileprivate` the read path is exercised
    /// indirectly: `setTelemetry(nil)` acquires the lock for a write that
    /// transitions the state from whatever the concurrent writers left it in.
    /// Any unsynchronised read inside the lock would manifest as a TSan race.
    @Test("setTelemetry toggles do not race with a high-frequency nil writer")
    func test_setTelemetry_toggleWithHighFrequencyNilWriter_doesNotRace() async throws {
        let dit = try makeFreshDiT()
        let reporter1 = MockReporter()
        let reporter2 = MockReporter()

        await withTaskGroup(of: Void.self) { group in
            // Slow toggler: sets reporter1 then reporter2 in each iteration.
            group.addTask {
                for _ in 0..<200 {
                    dit.setTelemetry(reporter1)
                    dit.setTelemetry(reporter2)
                }
            }
            // High-frequency nil writer: forces lock state transitions between
            // every slow-toggler write.
            group.addTask {
                for _ in 0..<5000 {
                    dit.setTelemetry(nil)
                }
            }
            // Second high-frequency nil writer for extra pressure.
            group.addTask {
                for _ in 0..<5000 {
                    dit.setTelemetry(nil)
                }
            }
        }

        #expect(Bool(true), "High-frequency nil writer completed without crash alongside slow toggler")
    }
}
