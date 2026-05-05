import Testing
import Tuberia

@testable import PixArtBackbone

@Suite("PixArtRecipe")
struct RecipeTests {

  @Test("Default generation parameters")
  func generationDefaults() {
    #expect(PixArtRecipe.defaultSteps == 20)
    #expect(PixArtRecipe.defaultGuidanceScale == 4.5)
  }

  @Test("Recipe validation passes with default configuration")
  func validationPasses() throws {
    let recipe = PixArtRecipe()
    try recipe.validate()
  }

  @Test("Recipe is text-to-image only")
  func textToImageOnly() {
    let recipe = PixArtRecipe()
    #expect(recipe.supportsImageToImage == false)
  }

  @Test("Unconditional embedding uses empty prompt")
  func unconditionalStrategy() {
    let recipe = PixArtRecipe()
    if case .emptyPrompt = recipe.unconditionalEmbeddingStrategy {
      // expected
    } else {
      Issue.record("Expected .emptyPrompt, got different strategy")
    }
  }

  @Test("Scheduler beta schedule: linear betaStart=0.0001, betaEnd=0.02")
  func schedulerBetaSchedule() {
    let recipe = PixArtRecipe()
    if case .linear(let betaStart, let betaEnd) = recipe.schedulerConfig.betaSchedule {
      #expect(abs(betaStart - 0.0001) < 1e-7)
      #expect(abs(betaEnd - 0.02) < 1e-7)
    } else {
      Issue.record("Expected .linear beta schedule, got a different schedule type")
    }
  }

  @Test(
    "allComponentIds is exactly [t5-xxl-encoder-int4, pixart-sigma-xl-dit-int4, sdxl-vae-decoder-fp16]"
  )
  func componentIdsExactOrder() {
    let recipe = PixArtRecipe()
    #expect(
      recipe.allComponentIds == [
        "t5-xxl-encoder-int4",
        "pixart-sigma-xl-dit-int4",
        "sdxl-vae-decoder-fp16",
      ])
  }

  @Test("Shape contract: encoder embeddingDim matches backbone captionChannels")
  func encoderBackboneContract() {
    let recipe = PixArtRecipe()
    #expect(recipe.encoderConfig.embeddingDim == recipe.backboneConfig.captionChannels)
  }

  @Test("Shape contract: encoder maxSequenceLength matches backbone maxTextLength")
  func sequenceLengthContract() {
    let recipe = PixArtRecipe()
    #expect(recipe.encoderConfig.maxSequenceLength == recipe.backboneConfig.maxTextLength)
  }

  @Test("Shape contract: decoder latentChannels is 4")
  func decoderLatentChannels() {
    let recipe = PixArtRecipe()
    #expect(recipe.decoderConfig.latentChannels == 4)
  }

  @Test("Encoder componentId is t5-xxl-encoder-int4")
  func encoderComponentId() {
    let recipe = PixArtRecipe()
    #expect(recipe.encoderConfig.componentId == "t5-xxl-encoder-int4")
  }

  @Test("Decoder componentId is sdxl-vae-decoder-fp16")
  func decoderComponentId() {
    let recipe = PixArtRecipe()
    #expect(recipe.decoderConfig.componentId == "sdxl-vae-decoder-fp16")
  }

  @Test("Decoder scalingFactor matches SDXL VAE spec")
  func decoderScalingFactor() {
    let recipe = PixArtRecipe()
    #expect(abs(recipe.decoderConfig.scalingFactor - 0.13025) < 1e-6)
  }

  @Test("Backbone config hiddenSize is 1152")
  func backboneHiddenSize() {
    let recipe = PixArtRecipe()
    #expect(recipe.backboneConfig.hiddenSize == 1152)
  }

  @Test("Backbone config depth is 28")
  func backboneDepth() {
    let recipe = PixArtRecipe()
    #expect(recipe.backboneConfig.depth == 28)
  }

  // MARK: - componentIdFor regression pins (R2.7)
  //
  // The override for `componentIdFor` exists because the default zip-based mapping
  // mis-aligns roles when `allComponentIds` skips weightless roles (scheduler, renderer).
  // Bug fixed in PR #10 (commit 32ce4c3); these tests prevent silent reintroduction.

  @Test("componentIdFor maps .encoder to t5-xxl-encoder-int4")
  func componentIdForEncoder() {
    let recipe = PixArtRecipe()
    #expect(recipe.componentIdFor[.encoder] == "t5-xxl-encoder-int4")
  }

  @Test("componentIdFor maps .backbone to pixart-sigma-xl-dit-int4")
  func componentIdForBackbone() {
    let recipe = PixArtRecipe()
    #expect(recipe.componentIdFor[.backbone] == "pixart-sigma-xl-dit-int4")
  }

  @Test("componentIdFor maps .decoder to sdxl-vae-decoder-fp16")
  func componentIdForDecoder() {
    let recipe = PixArtRecipe()
    #expect(recipe.componentIdFor[.decoder] == "sdxl-vae-decoder-fp16")
  }

  @Test("componentIdFor does not map weightless roles")
  func componentIdForWeightlessRoles() {
    let recipe = PixArtRecipe()
    #expect(recipe.componentIdFor[.scheduler] == nil)
    #expect(recipe.componentIdFor[.renderer] == nil)
  }

  // MARK: - Scheduler config field pins (R2.7)

  @Test("Scheduler predictionType is .epsilon")
  func schedulerPredictionType() {
    let recipe = PixArtRecipe()
    #expect(recipe.schedulerConfig.predictionType == .epsilon)
  }

  @Test("Scheduler solverOrder is 2")
  func schedulerSolverOrder() {
    let recipe = PixArtRecipe()
    #expect(recipe.schedulerConfig.solverOrder == 2)
  }

  @Test("Scheduler trainTimesteps is 1000")
  func schedulerTrainTimesteps() {
    let recipe = PixArtRecipe()
    #expect(recipe.schedulerConfig.trainTimesteps == 1000)
  }

  // MARK: - Quantization (R2.7)

  @Test("quantizationFor returns .asStored for every role")
  func quantizationAsStoredForAllRoles() {
    let recipe = PixArtRecipe()
    for role in PipelineRole.allCases {
      if case .asStored = recipe.quantizationFor(role) {
        // expected
      } else {
        Issue.record("Expected .asStored for role \(role), got different config")
      }
    }
  }
}
