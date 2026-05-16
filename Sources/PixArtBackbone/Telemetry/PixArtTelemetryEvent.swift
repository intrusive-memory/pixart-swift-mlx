import Foundation
@preconcurrency import MLX
import Tuberia

/// Slim boundary-only telemetry surface for the PixArt DiT backbone.
///
/// Follows the cross-library chokepoint convention documented in
/// `flux-2-swift-mlx/AGENTS.md §11`: instrument boundaries, not internals.
/// Per-step / per-block / per-attention-head detail is **deferred** — a
/// `numericalAnomaly` or `errorThrown` here points the agent at the region;
/// finer instrumentation is added in a follow-up iteration only after a real
/// failure demands it.
public enum PixArtTelemetryEvent: Sendable {

  // --- Resource lifecycle ---
  case weightLoadComplete(component: WeightComponent, paramCount: Int, durationSeconds: Double)
  case weightUnloadComplete

  // --- Recipe configuration ---
  case recipeValidated(name: String, checksPassed: Int)
  case recipeValidationFailed(name: String, check: String, reason: String)

  // --- Forward-pass output statistics ---
  /// Emitted unconditionally at the exit of `PixArtDiT.forward(_:)` carrying the
  /// sampled output tensor statistics. Pairs with the anomaly-gated
  /// `numericalAnomaly(phase: .ditForward, ...)` — when the backbone is healthy
  /// this is the only forward-pass event you see; when it is unhealthy you get
  /// both. Downstream pipelines use the per-step `stat` history to attribute
  /// quality regressions (e.g. color cast, saturation clipping) to either the
  /// conditioning origin (bias visible from step 0) or the denoise loop
  /// (bias accumulating over steps).
  case backboneForwardComplete(stat: TuberiaTensorStat)

  // --- Side channels ---
  case numericalAnomaly(phase: AnomalyPhase, kind: AnomalyKind, stat: TuberiaTensorStat)
  case errorThrown(phase: ErrorPhase, errorDescription: String)

  public enum WeightComponent: String, Sendable, Codable {
    case dit
  }

  public enum AnomalyPhase: String, Sendable {
    case weightLoad
    case ditForward
  }

  public enum AnomalyKind: String, Sendable {
    case nan
    case inf
    case outOfRange
    case zeroLatent
  }

  public enum ErrorPhase: String, Sendable {
    case weightLoad
    case forward
    case recipeValidation
    case other
  }
}
