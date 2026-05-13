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
