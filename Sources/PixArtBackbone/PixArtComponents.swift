import TuberiaCatalog

// MARK: - PixArt Component Registration

/// Registers PixArt-Sigma model component descriptors with TuberiaCatalog's
/// `CatalogRegistration.shared` registry.
///
/// Registers three components:
/// - PixArt-Sigma XL DiT (int4) — ~300 MB, owned by this package
/// - T5-XXL Encoder (int4) — ~1.2 GB, catalog component (re-registered for safety)
/// - SDXL VAE Decoder (fp16) — ~160 MB, catalog component (re-registered for safety)
///
/// T5-XXL and SDXL VAE are also registered by TuberiaCatalog at import time via
/// `CatalogRegistration.shared.ensureRegistered()`. Re-registration is idempotent:
/// `CatalogRegistration.register()` deduplicates by component ID.
///
/// Usage:
/// ```swift
/// _ = PixArtComponents.registered  // trigger once-only registration
/// ```
///
/// Swift guarantees the `registered` static `let` initializer is thread-safe and
/// executes exactly once regardless of concurrent callers.
public enum PixArtComponents {

  /// Trigger component registration. Evaluating this property registers all three
  /// PixArt-Sigma components. Safe to call from any thread at any time.
  ///
  /// Returns `true` always; the Bool return type enables the `_ = PixArtComponents.registered`
  /// idiom without a discardable-result warning.
  public static let registered: Bool = {
    let registry = CatalogRegistration.shared

    // PixArt-Sigma XL DiT (int4) — package-owned component
    // Weights created by convert_pixart_weights.py (Sortie 6)
    registry.register(
      ComponentDescriptor(
        componentId: "pixart-sigma-xl-dit-int4",
        componentType: .backbone,
        huggingFaceRepo: "intrusive-memory/pixart-sigma-xl-dit-int4-mlx",
        filePatterns: ["*.safetensors", "config.json"],
        estimatedSizeBytes: 314_572_800,  // ~300 MB int4
        sha256Checksums: nil  // Populated after weight conversion
      ))

    // PixArt-Sigma XL DiT (fp16) — mixed-precision test component
    // Weights produced by scripts/dequantize_dit_to_fp16.py from the int4 safetensors.
    // Used to isolate whether int4 quantization errors cause the blue/cyan mosaic artifact.
    // HuggingFace repo is a placeholder; for local testing weights live at:
    //   /tmp/vinetas-test-models/pixart-sigma-xl-dit-fp16/
    registry.register(
      ComponentDescriptor(
        componentId: "pixart-sigma-xl-dit-fp16",
        componentType: .backbone,
        huggingFaceRepo: "intrusive-memory/pixart-sigma-xl-dit-fp16-mlx",
        filePatterns: ["*.safetensors", "config.json"],
        estimatedSizeBytes: 1_258_291_200,  // ~1.2 GB fp16
        sha256Checksums: nil
      ))

    // T5-XXL Encoder (int4) — catalog component, authoritative in SwiftTubería.
    // Re-registered here for safety; CatalogRegistration deduplicates by ID.
    registry.register(
      ComponentDescriptor(
        componentId: "t5-xxl-encoder-int4",
        componentType: .encoder,
        huggingFaceRepo: "intrusive-memory/t5-xxl-int4-mlx",
        filePatterns: ["*.safetensors", "tokenizer.json", "tokenizer_config.json", "config.json"],
        estimatedSizeBytes: 1_288_490_188,  // ~1.2 GB int4
        sha256Checksums: nil
      ))

    // SDXL VAE Decoder (fp16) — catalog component, authoritative in SwiftTubería.
    // Re-registered here for safety; CatalogRegistration deduplicates by ID.
    // VAE is not int4-quantized: Conv2d layers do not benefit from weight-only quantization.
    registry.register(
      ComponentDescriptor(
        componentId: "sdxl-vae-decoder-fp16",
        componentType: .decoder,
        huggingFaceRepo: "intrusive-memory/sdxl-vae-fp16-mlx",
        filePatterns: ["*.safetensors", "config.json"],
        estimatedSizeBytes: 167_772_160,  // ~160 MB fp16
        sha256Checksums: nil
      ))

    return true
  }()
}
