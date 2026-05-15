import os.lock

/// Process-wide telemetry seam for PixArt DiT backbone.
///
/// Hosts that cannot hold a direct reference to a `PixArtDiT` instance — such as
/// a CLI bootstrap that installs a reporter before the engine lazily creates its
/// DiT — can call `PixArtTelemetry.setReporter(_:)` once at process startup.
/// Every subsequent `PixArtDiT` emission site falls back to this reporter when
/// the instance has no reporter installed via `setTelemetry(_:)`.
///
/// ## Precedence
///
/// Instance reporter (installed via `PixArtDiT.setTelemetry(_:)`) always takes
/// precedence over the process-wide reporter.  Both can be active simultaneously;
/// only the winning reporter receives each event.
///
/// ## Thread safety
///
/// `setReporter(_:)` and `current` are both guarded by an `OSAllocatedUnfairLock`
/// and are safe to call from any thread or task concurrently.
///
/// ## Lifetime
///
/// Call `PixArtTelemetry.setReporter(nil)` to clear the process-wide reporter.
/// This is the expected call from a CLI host's `bootstrap.finish()` teardown.
///
/// ## Example
///
/// ```swift
/// // At process startup (CLI bootstrap)
/// PixArtTelemetry.setReporter(myAdapter)
///
/// // Later, in teardown
/// PixArtTelemetry.setReporter(nil)
/// ```
public enum PixArtTelemetry {

  private static let _lock = OSAllocatedUnfairLock<(any PixArtTelemetryReporter)?>(
    initialState: nil)

  /// Installs or clears the process-wide telemetry reporter.
  ///
  /// Pass `nil` to clear the reporter (e.g. during teardown).
  /// Safe to call from any thread.
  public static func setReporter(_ reporter: (any PixArtTelemetryReporter)?) {
    _lock.withLock { $0 = reporter }
  }

  /// The currently installed process-wide reporter, or `nil` if none is set.
  ///
  /// All `PixArtDiT` emission sites consult this when the instance has no
  /// reporter installed.  Safe to call from any thread.
  public static var current: (any PixArtTelemetryReporter)? {
    _lock.withLock { $0 }
  }
}
