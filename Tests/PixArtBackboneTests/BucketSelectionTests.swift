import Testing

@testable import PixArtBackbone

@Suite("PixArtDiTConfiguration aspect-ratio buckets")
struct BucketSelectionTests {

  // MARK: - aspectRatioBuckets table invariants

  @Test("aspectRatioBuckets has exactly 64 entries")
  func bucketTableSize() {
    #expect(PixArtDiTConfiguration.aspectRatioBuckets.count == 64)
  }

  @Test("aspectRatioBuckets total pixels are within training-distribution range")
  func bucketTotalPixels() {
    // The PixArt-Sigma 64-bucket scheme does NOT keep total pixels constant —
    // extreme-aspect buckets like (256, 2048) sit around 525K pixels while the
    // canonical 1024x1024 bucket sits at ~1.05M. Pinning a wide envelope here.
    for (i, bucket) in PixArtDiTConfiguration.aspectRatioBuckets.enumerated() {
      let total = bucket.width * bucket.height
      #expect(total >= 500_000 && total <= 1_200_000,
        "Bucket \(i) (\(bucket)) total \(total) is outside [500K, 1.2M] envelope")
    }
  }

  @Test("aspectRatioBuckets contains the canonical 1024x1024 entry")
  func bucketContains1024Square() {
    let buckets = PixArtDiTConfiguration.aspectRatioBuckets
    #expect(buckets.contains(where: { $0.width == 1024 && $0.height == 1024 }))
  }

  @Test("aspectRatioBuckets contains a 768x768 entry")
  func bucketContains768Square() {
    let buckets = PixArtDiTConfiguration.aspectRatioBuckets
    #expect(buckets.contains(where: { $0.width == 768 && $0.height == 768 }))
  }

  // MARK: - nearestBucket — known cases

  @Test("nearestBucket(1024, 1024) returns a 1:1 bucket")
  func nearestBucketSquare() {
    let bucket = PixArtDiTConfiguration.nearestBucket(width: 1024, height: 1024)
    let ar = Float(bucket.width) / Float(bucket.height)
    #expect(abs(ar - 1.0) < 1e-6)
  }

  @Test("nearestBucket prefers landscape for landscape requests")
  func nearestBucketLandscape() {
    let bucket = PixArtDiTConfiguration.nearestBucket(width: 1920, height: 1080)
    #expect(bucket.width >= bucket.height,
      "Expected landscape (w>=h) for 1920x1080 request, got \(bucket)")
    let ar = Float(bucket.width) / Float(bucket.height)
    let targetAR: Float = 1920.0 / 1080.0
    // Closest table entry should be within ±0.05 AR of target
    #expect(abs(ar - targetAR) < 0.05,
      "Returned bucket AR \(ar) differs from target \(targetAR) by more than 0.05")
  }

  @Test("nearestBucket prefers portrait for portrait requests")
  func nearestBucketPortrait() {
    let bucket = PixArtDiTConfiguration.nearestBucket(width: 1080, height: 1920)
    #expect(bucket.height >= bucket.width,
      "Expected portrait (h>=w) for 1080x1920 request, got \(bucket)")
    let ar = Float(bucket.width) / Float(bucket.height)
    let targetAR: Float = 1080.0 / 1920.0
    #expect(abs(ar - targetAR) < 0.05)
  }

  @Test("nearestBucket(2048, 256) returns the most extreme wide bucket")
  func nearestBucketExtremeWide() {
    // (2048, 256) is itself in the table at AR=8.0 — the most extreme entry.
    let bucket = PixArtDiTConfiguration.nearestBucket(width: 2048, height: 256)
    #expect(bucket.width == 2048)
    #expect(bucket.height == 256)
  }

  @Test("nearestBucket symmetry: portrait of a landscape request is the swap")
  func nearestBucketPortraitLandscapeSymmetry() {
    // For an aspect ratio that has both orientations in the table.
    let landscape = PixArtDiTConfiguration.nearestBucket(width: 1500, height: 1000)
    let portrait = PixArtDiTConfiguration.nearestBucket(width: 1000, height: 1500)
    #expect(landscape.width == portrait.height)
    #expect(landscape.height == portrait.width)
  }

  @Test("nearestBucket returns a bucket from the canonical table")
  func nearestBucketReturnsTableEntry() {
    let bucket = PixArtDiTConfiguration.nearestBucket(width: 1500, height: 1000)
    let buckets = PixArtDiTConfiguration.aspectRatioBuckets
    #expect(
      buckets.contains(where: { $0.width == bucket.width && $0.height == bucket.height }),
      "Returned bucket \(bucket) is not in the canonical table")
  }

  @Test("nearestBucket minimizes aspect-ratio distance")
  func nearestBucketMinimizesDistance() {
    let request: (width: Int, height: Int) = (1500, 1000)
    let chosen = PixArtDiTConfiguration.nearestBucket(width: request.width, height: request.height)
    let targetAR = Float(request.width) / Float(request.height)
    let chosenDiff = abs(Float(chosen.width) / Float(chosen.height) - targetAR)
    for bucket in PixArtDiTConfiguration.aspectRatioBuckets {
      let bucketDiff = abs(Float(bucket.width) / Float(bucket.height) - targetAR)
      #expect(bucketDiff >= chosenDiff - 1e-6)
    }
  }
}
