import ArgumentParser

@main
struct PixArtCLI: AsyncParsableCommand {
  static let configuration = CommandConfiguration(
    commandName: "pixart",
    abstract: "PixArt-Sigma image generation CLI",
    subcommands: [GenerateCommand.self, DownloadCommand.self, InfoCommand.self]
  )
}
