# TODO

## CI / Testing

- [ ] Make integration tests run nightly in CI
  - Add a nightly GitHub Actions job (`schedule: cron`) separate from the PR test job
  - Set `SWIFT_ACTIVE_COMPILATION_CONDITIONS: INTEGRATION_TESTS` in the xcodebuild step
  - Pull model weights from Cloudflare R2 CDN (already hosted) or cache via GitHub Actions cache
  - Gate this job on `schedule:` + `workflow_dispatch:` only — not `pull_request:`
  - Runner: `macos-26` (Apple Silicon — required for MLX)
