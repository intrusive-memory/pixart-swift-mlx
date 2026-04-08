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

  @Test("DiT component ID matches recipe")
  func ditComponentIdMatchesRecipe() throws {
    let recipe = PixArtRecipe()
    let ids = recipe.allComponentIds
    #expect(ids.contains("pixart-sigma-xl-dit-int4"))
  }

  @Test("Encoder component ID matches recipe")
  func encoderComponentIdMatchesRecipe() {
    let recipe = PixArtRecipe()
    #expect(recipe.encoderConfig.componentId == "t5-xxl-encoder-int4")
  }

  @Test("Decoder component ID matches recipe")
  func decoderComponentIdMatchesRecipe() {
    let recipe = PixArtRecipe()
    #expect(recipe.decoderConfig.componentId == "sdxl-vae-decoder-fp16")
  }
}
