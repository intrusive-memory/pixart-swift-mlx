import MLX
import Testing

@testable import PixArtBackbone

@Suite("FinalLayer")
struct FinalLayerTests {

  @Test("FinalLayer output shape: [B, gridH * patchSize, gridW * patchSize, outChannels]")
  func outputShape() {
    let hiddenSize = 16
    let patchSize = 2
    let outChannels = 4
    let B = 1
    let gridH = 4
    let gridW = 6
    let T = gridH * gridW  // token count

    let layer = FinalLayer(hiddenSize: hiddenSize, patchSize: patchSize, outChannels: outChannels)
    let x = MLXArray.zeros([B, T, hiddenSize])
    let t = MLXArray.zeros([B, hiddenSize])  // raw timestep embedding [B, hiddenSize]
    let result = layer(x, t: t, gridH: gridH, gridW: gridW)
    eval(result)

    #expect(result.dim(0) == B)
    #expect(result.dim(1) == gridH * patchSize)
    #expect(result.dim(2) == gridW * patchSize)
    #expect(result.dim(3) == outChannels)
  }

  @Test("FinalLayer variance channel handling: outChannels=8 includes variance channels")
  func varianceChannels() {
    // PixArt outputs 8 channels (4 noise + 4 variance); caller discards [4..7]
    let hiddenSize = 16
    let patchSize = 2
    let outChannels = 8  // 4 noise + 4 variance as in the real model
    let B = 1
    let gridH = 2
    let gridW = 2
    let T = gridH * gridW

    let layer = FinalLayer(hiddenSize: hiddenSize, patchSize: patchSize, outChannels: outChannels)
    let x = MLXArray.zeros([B, T, hiddenSize])
    let t = MLXArray.zeros([B, hiddenSize])
    let result = layer(x, t: t, gridH: gridH, gridW: gridW)
    eval(result)

    // Full output has outChannels
    #expect(result.dim(3) == outChannels)

    // Caller discards variance channels (keeps [0..<4])
    let noiseOnly = result[0..., 0..., 0..., 0..<4]
    eval(noiseOnly)
    #expect(noiseOnly.dim(3) == 4)
  }

  @Test("FinalLayer scaleShiftTable has shape [2, hiddenSize]")
  func scaleShiftTableShape() {
    let hiddenSize = 16
    let layer = FinalLayer(hiddenSize: hiddenSize, patchSize: 2, outChannels: 4)
    eval(layer.scaleShiftTable)
    #expect(layer.scaleShiftTable.dim(0) == 2)
    #expect(layer.scaleShiftTable.dim(1) == hiddenSize)
  }

  @Test("FinalLayer unpatchify: spatial dims = gridH*patchSize, gridW*patchSize")
  func unpatchifyDimensions() {
    let hiddenSize = 32
    let patchSize = 2
    let outChannels = 4
    let B = 1
    let gridH = 3
    let gridW = 5
    let T = gridH * gridW

    let layer = FinalLayer(hiddenSize: hiddenSize, patchSize: patchSize, outChannels: outChannels)
    let x = MLXArray.zeros([B, T, hiddenSize])
    let t = MLXArray.zeros([B, hiddenSize])
    let result = layer(x, t: t, gridH: gridH, gridW: gridW)
    eval(result)

    #expect(result.dim(1) == gridH * patchSize)
    #expect(result.dim(2) == gridW * patchSize)
  }

  @Test("FinalLayer with batch size > 1")
  func batchSize() {
    let hiddenSize = 16
    let patchSize = 2
    let outChannels = 4
    let B = 3
    let gridH = 2
    let gridW = 2
    let T = gridH * gridW

    let layer = FinalLayer(hiddenSize: hiddenSize, patchSize: patchSize, outChannels: outChannels)
    let x = MLXArray.zeros([B, T, hiddenSize])
    let t = MLXArray.zeros([B, hiddenSize])
    let result = layer(x, t: t, gridH: gridH, gridW: gridW)
    eval(result)

    #expect(result.dim(0) == B)
  }
}
