import ArgumentParser

@main
struct PixArtCLI: ParsableCommand {
  static let configuration = CommandConfiguration(
    commandName: "pixart",
    abstract: "PixArt-Sigma image generation CLI"
  )

  func run() throws {
    // Placeholder — implementation in subsequent sorties
    print("PixArtCLI — placeholder")
  }
}
