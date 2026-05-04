# Gemini-Specific Agent Instructions

**⚠️ Read [AGENTS.md](AGENTS.md) first** for universal project documentation.

This file contains instructions specific to Google Gemini agents.

## Build Commands

Use standard Makefile targets for all operations:

```bash
make build     # Debug build
make test      # Run unit tests
make lint      # Format Swift sources
make clean     # Clean build artifacts
```

## Gemini-Specific Critical Rules

1. Use Makefile targets (no MCP access)
2. ONLY supports iOS 26.0+ and macOS 26.0+
3. See [AGENTS.md](AGENTS.md) for universal rules (platform requirements, lint-before-commit, branch policy, App Group configuration)
