import SwiftAcervo
import Testing
import TuberiaCatalog

@testable import PixArtBackbone

@Suite("PixArtComponents")
struct ComponentRegistrationTests {

  @Test("Registration succeeds")
  func registration() {
    #expect(PixArtComponents.registered == true)
  }

  @Test("DiT component descriptor is non-nil after registration")
  func ditComponentRegistered() {
    _ = PixArtComponents.registered
    #expect(CatalogRegistration.shared.descriptor(for: "pixart-sigma-xl-dit-int4") != nil)
  }

  @Test("T5 encoder component descriptor is non-nil after registration")
  func encoderComponentRegistered() {
    _ = PixArtComponents.registered
    #expect(CatalogRegistration.shared.descriptor(for: "t5-xxl-encoder-int4") != nil)
  }

  @Test("SDXL VAE decoder component descriptor is non-nil after registration")
  func decoderComponentRegistered() {
    _ = PixArtComponents.registered
    #expect(CatalogRegistration.shared.descriptor(for: "sdxl-vae-decoder-fp16") != nil)
  }

  @Test("allComponentIds from recipe are all 3 expected IDs")
  func allThreeComponentIds() {
    let recipe = PixArtRecipe()
    let ids = recipe.allComponentIds

    // Verify all 3 component IDs are present
    #expect(ids.contains("t5-xxl-encoder-int4"))
    #expect(ids.contains("pixart-sigma-xl-dit-int4"))
    #expect(ids.contains("sdxl-vae-decoder-fp16"))
    #expect(ids.count == 3)
  }

  // MARK: - FP16 backbone descriptor (R2.8)

  @Test("FP16 DiT backbone descriptor is registered")
  func fp16DiTComponentRegistered() {
    _ = PixArtComponents.registered
    #expect(Acervo.component("pixart-sigma-xl-dit-fp16") != nil)
  }

  // MARK: - Descriptor metadata field pins (R2.8)

  @Test("int4 DiT descriptor has expected repoId and metadata")
  func int4DescriptorFields() throws {
    _ = PixArtComponents.registered
    let descriptor = try #require(Acervo.component("pixart-sigma-xl-dit-int4"))
    #expect(descriptor.repoId == "intrusive-memory/pixart-sigma-xl-dit-int4-mlx")
    #expect(descriptor.minimumMemoryBytes == 800_000_000)
    #expect(descriptor.metadata["component_role"] == "backbone")
    #expect(descriptor.metadata["quantization"] == "int4")
    #expect(descriptor.metadata["architecture"] == "DiT-XL")
  }

  @Test("fp16 DiT descriptor has expected repoId and metadata")
  func fp16DescriptorFields() throws {
    _ = PixArtComponents.registered
    let descriptor = try #require(Acervo.component("pixart-sigma-xl-dit-fp16"))
    #expect(descriptor.repoId == "intrusive-memory/pixart-sigma-xl-dit-fp16-mlx")
    #expect(descriptor.minimumMemoryBytes == 2_500_000_000)
    #expect(descriptor.metadata["component_role"] == "backbone")
    #expect(descriptor.metadata["quantization"] == "fp16")
    #expect(descriptor.metadata["architecture"] == "DiT-XL")
  }

  // MARK: - Idempotency (R2.8)

  @Test("Repeated calls to PixArtComponents.registered are idempotent")
  func registeredIsIdempotent() {
    let first = PixArtComponents.registered
    let second = PixArtComponents.registered
    #expect(first == true)
    #expect(second == true)
    #expect(Acervo.component("pixart-sigma-xl-dit-int4") != nil)
    #expect(Acervo.component("pixart-sigma-xl-dit-fp16") != nil)
  }
}
