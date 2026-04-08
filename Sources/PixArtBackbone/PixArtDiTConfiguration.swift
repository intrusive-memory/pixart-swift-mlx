/// Configuration for the PixArt-Sigma DiT backbone.
///
/// All values match the PixArt-Sigma XL architecture (arXiv:2403.04692).
public struct PixArtDiTConfiguration: Sendable {
  /// Hidden dimension of the transformer. Default: 1152.
  public let hiddenSize: Int
  /// Number of attention heads. Default: 16.
  public let numHeads: Int
  /// Dimension per attention head. Default: 72 (hiddenSize / numHeads).
  public let headDim: Int
  /// Number of DiT transformer blocks. Default: 28.
  public let depth: Int
  /// Patch size for the patch embedding convolution. Default: 2.
  public let patchSize: Int
  /// Number of input latent channels (from VAE). Default: 4.
  public let inChannels: Int
  /// Number of output channels (4 noise + 4 variance). Default: 8.
  public let outChannels: Int
  /// MLP expansion ratio. FFN hidden dim = hiddenSize * mlpRatio. Default: 4.0.
  public let mlpRatio: Float
  /// Dimension of text encoder embeddings (T5-XXL). Default: 4096.
  public let captionChannels: Int
  /// Maximum text sequence length. Default: 120.
  public let maxTextLength: Int
  /// Position embedding interpolation factor for PixArt-Sigma XL. Default: 2.
  public let peInterpolation: Float
  /// Base resolution used for position embedding normalization. Default: 512.
  public let baseSize: Int

  public init(
    hiddenSize: Int = 1152,
    numHeads: Int = 16,
    headDim: Int = 72,
    depth: Int = 28,
    patchSize: Int = 2,
    inChannels: Int = 4,
    outChannels: Int = 8,
    mlpRatio: Float = 4.0,
    captionChannels: Int = 4096,
    maxTextLength: Int = 120,
    peInterpolation: Float = 2.0,
    baseSize: Int = 512
  ) {
    self.hiddenSize = hiddenSize
    self.numHeads = numHeads
    self.headDim = headDim
    self.depth = depth
    self.patchSize = patchSize
    self.inChannels = inChannels
    self.outChannels = outChannels
    self.mlpRatio = mlpRatio
    self.captionChannels = captionChannels
    self.maxTextLength = maxTextLength
    self.peInterpolation = peInterpolation
    self.baseSize = baseSize
  }

  // MARK: - 64-Bucket Aspect Ratio Table

  /// 64-bucket aspect ratio scheme from the PixArt-Sigma paper at 1024px base resolution.
  ///
  /// Each bucket is a (width, height) pair in pixels. The backbone rounds user-requested
  /// resolution to the nearest bucket for micro-conditioning, but generates at the
  /// requested resolution (position embeddings are recomputed dynamically).
  ///
  /// Buckets are sorted by ascending aspect ratio (width/height).
  /// Total pixel counts are approximately 1M (1024x1024 = 1,048,576) per bucket.
  ///
  /// Common buckets:
  /// - 1:1 = 1024x1024
  /// - 4:3 = 1152x896
  /// - 3:4 = 896x1152
  /// - 16:9 = 1344x768
  /// - 9:16 = 768x1344
  public static let aspectRatioBuckets: [(width: Int, height: Int)] = [
    (256, 2048),  // 0.125
    (288, 2048),  // 0.141
    (320, 1856),  // 0.172
    (320, 1984),  // 0.161
    (352, 1664),  // 0.211
    (352, 1792),  // 0.196
    (384, 1536),  // 0.250
    (384, 1664),  // 0.231
    (416, 1408),  // 0.295
    (416, 1536),  // 0.271
    (448, 1280),  // 0.350
    (448, 1408),  // 0.318
    (480, 1216),  // 0.395
    (480, 1280),  // 0.375
    (512, 1088),  // 0.470
    (512, 1152),  // 0.444
    (512, 1216),  // 0.421
    (544, 1024),  // 0.531
    (544, 1088),  // 0.500
    (576, 960),  // 0.600
    (576, 1024),  // 0.563
    (608, 896),  // 0.679
    (608, 960),  // 0.633
    (640, 832),  // 0.769
    (640, 896),  // 0.714
    (672, 832),  // 0.808
    (704, 768),  // 0.917
    (704, 832),  // 0.846
    (736, 768),  // 0.958
    (768, 704),  // 1.091
    (768, 736),  // 1.043
    (768, 768),  // 1.000
    (832, 640),  // 1.300
    (832, 672),  // 1.238
    (832, 704),  // 1.182
    (896, 576),  // 1.556
    (896, 608),  // 1.474
    (896, 640),  // 1.400
    (960, 544),  // 1.765
    (960, 576),  // 1.667
    (960, 608),  // 1.579
    (1024, 512),  // 2.000
    (1024, 544),  // 1.882
    (1024, 576),  // 1.778
    (1024, 1024),  // 1.000
    (1088, 480),  // 2.267
    (1088, 512),  // 2.125
    (1088, 544),  // 2.000
    (1152, 448),  // 2.571
    (1152, 480),  // 2.400
    (1152, 512),  // 2.250
    (1216, 448),  // 2.714
    (1216, 480),  // 2.533
    (1280, 416),  // 3.077
    (1280, 448),  // 2.857
    (1344, 384),  // 3.500
    (1344, 416),  // 3.231
    (1408, 384),  // 3.667
    (1472, 352),  // 4.182
    (1536, 352),  // 4.364
    (1600, 320),  // 5.000
    (1664, 320),  // 5.200
    (1792, 288),  // 6.222
    (2048, 256),  // 8.000
  ]

  /// Returns the nearest aspect ratio bucket for the given resolution.
  ///
  /// Finds the bucket whose aspect ratio is closest to the requested aspect ratio.
  /// The backbone uses this for micro-conditioning but generates at the exact
  /// requested resolution.
  ///
  /// - Parameters:
  ///   - width: Requested image width in pixels.
  ///   - height: Requested image height in pixels.
  /// - Returns: The closest matching (width, height) bucket.
  public static func nearestBucket(width: Int, height: Int) -> (width: Int, height: Int) {
    let targetAR = Float(width) / Float(height)
    var bestBucket = aspectRatioBuckets[0]
    var bestDiff = Float.greatestFiniteMagnitude
    for bucket in aspectRatioBuckets {
      let bucketAR = Float(bucket.width) / Float(bucket.height)
      let diff = abs(bucketAR - targetAR)
      if diff < bestDiff {
        bestDiff = diff
        bestBucket = bucket
      }
    }
    return bestBucket
  }
}
