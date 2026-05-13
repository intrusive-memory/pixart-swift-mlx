public protocol PixArtTelemetryReporter: Sendable {
  func capture(_ event: PixArtTelemetryEvent) async
}

public struct NoopPixArtTelemetryReporter: PixArtTelemetryReporter {
  public init() {}
  public func capture(_ event: PixArtTelemetryEvent) async {}
}
