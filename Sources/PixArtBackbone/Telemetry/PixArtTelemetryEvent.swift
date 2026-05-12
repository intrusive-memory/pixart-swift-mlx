@preconcurrency import MLX
import Foundation
import Tuberia

public enum PixArtTelemetryEvent: Sendable {

    // --- Recipe lifecycle ---
    case recipeSelected(name: String, version: String, expectedSteps: Int, expectedGuidanceScale: Double, allComponentIDs: [String])
    case recipeValidated(name: String, checksPassed: Int)
    case recipeValidationFailed(name: String, check: String, reason: String)

    // --- DiT init ---
    case ditInitialized(
        hiddenSize: Int,
        depth: Int,
        numHeads: Int,
        patchSize: Int,
        maxTextLength: Int,
        captionChannels: Int,
        peInterpolation: Float,
        baseSize: Int
    )

    // --- Weight load (boundary memory event on complete) ---
    case weightApplyStart(quantization: PixArtQuantization, weightKeyCount: Int)
    case weightApplyComplete(
        quantization: PixArtQuantization,
        totalKeys: Int,
        dequantizedKeys: Int,      // int4 -> fp16 dequantization count
        passThroughKeys: Int,      // already-fp16 keys loaded directly
        scalesBiasesSkipped: Int,  // .scales and .biases sidecar keys consumed
        sizeMB: Double,
        durationSeconds: Double
    )
    case weightUnload(restoredKeyCount: Int)
    case microConditioningStatus(present: Bool, sizeEmbedderFound: Bool, arEmbedderFound: Bool)

    // --- Forward pass (per scheduler step) ---
    case ditForwardStart(
        stepIndex: Int?,  // populated when caller passes it through; nil if standalone test
        batch: Int,
        latentShape: [Int],
        conditioningShape: [Int],
        timestepShape: [Int],
        inputLatentStat: TuberiaTensorStat,
        conditioningStat: TuberiaTensorStat
    )
    case patchEmbedComplete(stat: TuberiaTensorStat, gridH: Int, gridW: Int)
    case captionProjectionComplete(stat: TuberiaTensorStat)
    case timestepEmbeddingComplete(sinusoidalStat: TuberiaTensorStat, projectedStat: TuberiaTensorStat, tBlockStat: TuberiaTensorStat)
    case siluWorkaroundExecuted  // marker that the manual sigmoid path ran (it always does today; absence on a future event would mean MLX.silu got swapped back in)
    case finalLayerComplete(stat: TuberiaTensorStat)  // 8-channel output before variance discard
    case varianceChannelsDiscarded(beforeChannels: Int, afterChannels: Int, beforeStat: TuberiaTensorStat, afterStat: TuberiaTensorStat)
    case ditForwardComplete(
        stepIndex: Int?,
        outputStat: TuberiaTensorStat,  // [B, H/8, W/8, 4] noise prediction
        durationSeconds: Double
    )

    // --- Numerical anomaly side-channel ---
    case numericalAnomaly(phase: String, kind: AnomalyKind, stepIndex: Int?, stat: TuberiaTensorStat)

    // --- Error side-channel ---
    case errorThrown(phase: ErrorPhase, errorDescription: String)

    public enum PixArtQuantization: String, Sendable, Codable {
        case int4         // int4-quantized safetensors (~300 MB)
        case fp16         // dequantized fp16 safetensors (larger, slightly different math)
        case unknown      // weights loaded but format heuristic didn't match either
    }

    public enum AnomalyKind: String, Sendable {
        case nan
        case inf
        case outOfRange
        case zeroLatent
        case shapeMismatch
    }

    public enum ErrorPhase: String, Sendable {
        case ditInit
        case weightApply
        case forwardPass
        case recipeValidation
        case shapeMismatch
        case other
    }
}
