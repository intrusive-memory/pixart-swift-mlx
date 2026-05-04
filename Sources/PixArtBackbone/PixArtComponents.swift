import SwiftAcervo
import TuberiaCatalog

// MARK: - Acervo Component Descriptors (v2 API)

private let pixArtSigmaXLDiTInt4Descriptor = SwiftAcervo.ComponentDescriptor(
  id: "pixart-sigma-xl-dit-int4",
  type: .backbone,
  displayName: "PixArt-Sigma XL DiT (int4)",
  repoId: "intrusive-memory/pixart-sigma-xl-dit-int4-mlx",
  minimumMemoryBytes: 800_000_000,
  metadata: [
    "component_role": "backbone",
    "quantization": "int4",
    "architecture": "DiT-XL",
  ]
)

private let pixArtSigmaXLDiTFP16Descriptor = SwiftAcervo.ComponentDescriptor(
  id: "pixart-sigma-xl-dit-fp16",
  type: .backbone,
  displayName: "PixArt-Sigma XL DiT (fp16)",
  repoId: "intrusive-memory/pixart-sigma-xl-dit-fp16-mlx",
  minimumMemoryBytes: 2_500_000_000,
  metadata: [
    "component_role": "backbone",
    "quantization": "fp16",
    "architecture": "DiT-XL",
  ]
)

// MARK: - PixArtComponents

/// Registers PixArt-Sigma backbone component descriptors with the SwiftAcervo v2
/// Component Registry.
///
/// T5-XXL encoder and SDXL VAE decoder are registered automatically by
/// `TuberiaCatalog` on module load; this type only owns the PixArt-specific
/// backbone descriptors (int4 production + fp16 validation variants).
///
/// Usage:
/// ```swift
/// _ = PixArtComponents.registered  // trigger once-only registration
/// ```
public enum PixArtComponents {

  /// Trigger component registration. Evaluating this property registers PixArt
  /// backbone components with Acervo and ensures TuberiaCatalog's T5+VAE
  /// registrations have also fired. Safe to call from any thread at any time.
  public static let registered: Bool = {
    CatalogRegistration.shared.ensureRegistered()
    Acervo.register([
      pixArtSigmaXLDiTInt4Descriptor,
      pixArtSigmaXLDiTFP16Descriptor,
    ])
    return true
  }()
}
