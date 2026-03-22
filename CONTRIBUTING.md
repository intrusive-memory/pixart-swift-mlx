# Contributing to pixart-swift-mlx

## Development Setup

| Requirement | Minimum Version |
|------------|----------------|
| macOS      | 26.0+          |
| Xcode      | 26+            |
| Swift      | 6.2+           |

### Clone and Build

```bash
git clone https://github.com/intrusive-memory/pixart-swift-mlx.git
cd pixart-swift-mlx
xcodebuild build -scheme pixart-swift-mlx -destination 'platform=macOS'
```

## Testing

```bash
xcodebuild test -scheme pixart-swift-mlx -destination 'platform=macOS'
```

## Commit Conventions

- **Add** — A wholly new feature or file.
- **Update** — An enhancement to existing functionality.
- **Fix** — A bug fix.
- **Remove** — Removal of code, files, or features.
- **Refactor** — Code restructuring with no behavior change.
- **Test** — Adding or updating tests only.
- **Docs** — Documentation-only changes.

## Pull Request Process

1. Branch from `development`.
2. Keep changes focused.
3. Ensure tests pass locally.
4. Open the PR against `development`.
5. CI must pass.

## Platform Requirements

pixart-swift-mlx targets iOS 26.0+ and macOS 26.0+ exclusively.
