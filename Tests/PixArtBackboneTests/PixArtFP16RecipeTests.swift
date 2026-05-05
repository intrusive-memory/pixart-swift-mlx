import Testing
import Tuberia

@testable import PixArtBackbone

@Suite("PixArtFP16Recipe")
struct PixArtFP16RecipeTests {

  // MARK: - Default Generation Parameters

  @Test("Default generation parameters match int4 recipe")
  func generationDefaults() {
    #expect(PixArtFP16Recipe.defaultSteps == PixArtRecipe.defaultSteps)
    #expect(PixArtFP16Recipe.defaultGuidanceScale == PixArtRecipe.defaultGuidanceScale)
    #expect(PixArtFP16Recipe.defaultSteps == 20)
    #expect(PixArtFP16Recipe.defaultGuidanceScale == 4.5)
  }

  // MARK: - Validation

  @Test("Recipe validation passes with default configuration")
  func validationPasses() throws {
    let recipe = PixArtFP16Recipe()
    try recipe.validate()
  }

  // MARK: - Encoder

  @Test("Encoder componentId is t5-xxl-encoder-int4 (encoder stays int4)")
  func encoderComponentId() {
    let recipe = PixArtFP16Recipe()
    #expect(recipe.encoderConfig.componentId == "t5-xxl-encoder-int4")
  }

  @Test("Encoder maxSequenceLength is 120")
  func encoderMaxSequenceLength() {
    let recipe = PixArtFP16Recipe()
    #expect(recipe.encoderConfig.maxSequenceLength == 120)
  }

  @Test("Encoder embeddingDim is 4096")
  func encoderEmbeddingDim() {
    let recipe = PixArtFP16Recipe()
    #expect(recipe.encoderConfig.embeddingDim == 4096)
  }

  // MARK: - Scheduler — divergence pin (R2.2)
  //
  // PixArtRecipe uses .linear; PixArtFP16Recipe currently uses .scaledLinear.
  // The doc comment claims "identical to int4 recipe" but the schedules differ.
  // This test pins whichever is currently in source. If you change either recipe's
  // schedule, update both this test and the doc comments — the divergence has to be
  // an intentional, documented choice rather than a silent drift.

  @Test("Scheduler beta schedule is currently .scaledLinear (diverges from int4 .linear)")
  func schedulerBetaScheduleDivergence() {
    let recipe = PixArtFP16Recipe()
    if case .scaledLinear(let betaStart, let betaEnd) = recipe.schedulerConfig.betaSchedule {
      #expect(abs(betaStart - 0.0001) < 1e-7)
      #expect(abs(betaEnd - 0.02) < 1e-7)
    } else {
      let message = "Expected .scaledLinear schedule (current source). If you've changed the schedule, update this test AND the doc comment in PixArtFP16Recipe.swift line 61."
      Issue.record(Comment(rawValue: message))
    }
  }

  @Test("Scheduler predictionType is .epsilon")
  func schedulerPredictionType() {
    let recipe = PixArtFP16Recipe()
    #expect(recipe.schedulerConfig.predictionType == .epsilon)
  }

  @Test("Scheduler solverOrder is 2")
  func schedulerSolverOrder() {
    let recipe = PixArtFP16Recipe()
    #expect(recipe.schedulerConfig.solverOrder == 2)
  }

  @Test("Scheduler trainTimesteps is 1000")
  func schedulerTrainTimesteps() {
    let recipe = PixArtFP16Recipe()
    #expect(recipe.schedulerConfig.trainTimesteps == 1000)
  }

  // MARK: - Backbone

  @Test("Backbone config uses default PixArtDiTConfiguration")
  func backboneConfigDefault() {
    let recipe = PixArtFP16Recipe()
    #expect(recipe.backboneConfig.hiddenSize == 1152)
    #expect(recipe.backboneConfig.depth == 28)
    #expect(recipe.backboneConfig.captionChannels == 4096)
    #expect(recipe.backboneConfig.maxTextLength == 120)
  }

  // MARK: - Decoder

  @Test("Decoder componentId is sdxl-vae-decoder-fp16")
  func decoderComponentId() {
    let recipe = PixArtFP16Recipe()
    #expect(recipe.decoderConfig.componentId == "sdxl-vae-decoder-fp16")
  }

  @Test("Decoder latentChannels is 4")
  func decoderLatentChannels() {
    let recipe = PixArtFP16Recipe()
    #expect(recipe.decoderConfig.latentChannels == 4)
  }

  @Test("Decoder scalingFactor matches SDXL VAE spec")
  func decoderScalingFactor() {
    let recipe = PixArtFP16Recipe()
    #expect(abs(recipe.decoderConfig.scalingFactor - 0.13025) < 1e-6)
  }

  // MARK: - Recipe properties

  @Test("Recipe is text-to-image only")
  func textToImageOnly() {
    let recipe = PixArtFP16Recipe()
    #expect(recipe.supportsImageToImage == false)
  }

  @Test("Unconditional embedding uses empty prompt")
  func unconditionalStrategy() {
    let recipe = PixArtFP16Recipe()
    if case .emptyPrompt = recipe.unconditionalEmbeddingStrategy {
      // expected
    } else {
      Issue.record("Expected .emptyPrompt, got different strategy")
    }
  }

  // MARK: - allComponentIds — fp16 NOT int4 (R2.2 critical)
  //
  // Silent regression risk: if the FP16 recipe pulls int4 weights, the entire
  // diagnostic harness for "is int4 quantization the source of artifacts?" is
  // invalidated without any visible failure. Pin both the inclusion of fp16 AND
  // the exclusion of int4 explicitly.

  @Test("allComponentIds contains pixart-sigma-xl-dit-fp16 (NOT -int4)")
  func allComponentIdsContainsFP16Backbone() {
    let recipe = PixArtFP16Recipe()
    #expect(recipe.allComponentIds.contains("pixart-sigma-xl-dit-fp16"))
    #expect(!recipe.allComponentIds.contains("pixart-sigma-xl-dit-int4"))
  }

  @Test("allComponentIds is exactly [t5-xxl-encoder-int4, pixart-sigma-xl-dit-fp16, sdxl-vae-decoder-fp16]")
  func allComponentIdsExactOrder() {
    let recipe = PixArtFP16Recipe()
    #expect(
      recipe.allComponentIds == [
        "t5-xxl-encoder-int4",
        "pixart-sigma-xl-dit-fp16",
        "sdxl-vae-decoder-fp16",
      ])
  }

  // MARK: - componentIdFor regression pins (R2.2)

  @Test("componentIdFor maps .backbone to pixart-sigma-xl-dit-fp16")
  func componentIdForBackboneIsFP16() {
    let recipe = PixArtFP16Recipe()
    #expect(recipe.componentIdFor[.backbone] == "pixart-sigma-xl-dit-fp16")
  }

  @Test("componentIdFor maps .encoder and .decoder to int4 / fp16 respectively")
  func componentIdForEncoderDecoder() {
    let recipe = PixArtFP16Recipe()
    #expect(recipe.componentIdFor[.encoder] == "t5-xxl-encoder-int4")
    #expect(recipe.componentIdFor[.decoder] == "sdxl-vae-decoder-fp16")
  }

  @Test("componentIdFor does not map weightless roles")
  func componentIdForWeightlessRoles() {
    let recipe = PixArtFP16Recipe()
    #expect(recipe.componentIdFor[.scheduler] == nil)
    #expect(recipe.componentIdFor[.renderer] == nil)
  }

  // MARK: - Quantization

  @Test("quantizationFor returns .asStored for every role")
  func quantizationAsStoredForAllRoles() {
    let recipe = PixArtFP16Recipe()
    for role in PipelineRole.allCases {
      if case .asStored = recipe.quantizationFor(role) {
        // expected
      } else {
        Issue.record("Expected .asStored for role \(role), got different config")
      }
    }
  }
}
