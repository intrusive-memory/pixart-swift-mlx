import Testing

@testable import PixArtBackbone

@Suite("PixArt-Sigma 1024 resolution binning (diffusers ASPECT_RATIO_1024_BIN)")
struct ResolutionBinningTests {

  @Test("The table is diffusers' 33 bins, every edge a multiple of 64, all about 1 MP")
  func table() {
    let bins = PixArtDiTConfiguration.resolutionBins1024
    #expect(bins.count == 33)
    for bin in bins {
      #expect(bin.width % 64 == 0 && bin.height % 64 == 0, "\(bin)")
      #expect((786_432...1_179_648).contains(bin.width * bin.height), "\(bin)")
    }
    #expect(bins.map(\.heightOverWidth) == bins.map(\.heightOverWidth).sorted())
  }

  @Test(
    "Requests map to the bin diffusers picks",
    arguments: [
      (512, 512, 1024, 1024),
      (1024, 1024, 1024, 1024),
      (2160, 2160, 1024, 1024),
      (1344, 768, 1344, 768),
      (1920, 1080, 1344, 768),
      (1080, 1920, 768, 1344),
      (768, 1024, 896, 1152),
      (4000, 500, 2048, 512),
      (500, 4000, 512, 2048),
    ])
  func binned(_ w: Int, _ h: Int, _ bw: Int, _ bh: Int) {
    let bin = PixArtDiTConfiguration.binnedResolution(width: w, height: h)
    #expect(bin.width == bw && bin.height == bh, "\(w)×\(h) → \(bin)")
  }
}
