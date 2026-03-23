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
}
