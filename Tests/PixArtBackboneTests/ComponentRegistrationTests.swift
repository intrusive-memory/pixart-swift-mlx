import Testing

@testable import PixArtBackbone

@Suite("PixArtComponents")
struct ComponentRegistrationTests {

  @Test("Registration succeeds")
  func registration() {
    #expect(PixArtComponents.registered == true)
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
