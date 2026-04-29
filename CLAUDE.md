# Claude-Specific Agent Instructions

**⚠️ Read [AGENTS.md](AGENTS.md) first** for universal project documentation.

This file contains instructions specific to Claude Code agents.

## Build Preferences

- NEVER use `swift build` or `swift test` — use `make` targets or `xcodebuild` (XcodeBuildMCP locally; raw `xcodebuild` in CI)
- Use `make build`, `make test`, `make lint` for standard operations
- See [AGENTS.md](AGENTS.md) for the full Makefile target list

## Claude-Specific Critical Rules

1. NEVER use `swift build` or `swift test` — Makefile targets or `xcodebuild` only
2. Prefer XcodeBuildMCP tools (`build_macos`, `test_macos`, `swift_package_test`, etc.) for local builds when available
3. See [AGENTS.md](AGENTS.md) for universal rules (platform requirements, lint-before-commit, branch policy)
