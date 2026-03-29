import Foundation

// MARK: - CLI Error

enum CLIError: Error, CustomStringConvertible {
  case unexpectedOutput(String)
  case saveFailed(String)
  case downloadFailed(String)

  var description: String {
    switch self {
    case .unexpectedOutput(let msg): return "Unexpected output: \(msg)"
    case .saveFailed(let msg): return "Save failed: \(msg)"
    case .downloadFailed(let msg): return "Download failed: \(msg)"
    }
  }
}

// MARK: - Async Bridge

/// Run an async throwing closure synchronously from a synchronous context.
/// Used to bridge ArgumentParser's synchronous `run()` into async pipeline calls.
func runAsync<T: Sendable>(_ block: @Sendable @escaping () async throws -> T) throws -> T {
  var result: Result<T, Error>?
  let semaphore = DispatchSemaphore(value: 0)
  Task {
    do {
      result = .success(try await block())
    } catch {
      result = .failure(error)
    }
    semaphore.signal()
  }
  semaphore.wait()
  return try result!.get()
}
