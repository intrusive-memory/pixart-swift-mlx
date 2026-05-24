# SwiftAcervo 0.16.0 Upgrade

- Target version: **0.16.0**
- Current pinned version: **0.14.0** (`Package.resolved`; `Package.swift` already uses `.upToNextMajor(from: "0.14.0")`)
- Authoritative migration guide: [`/Users/stovak/Projects/SwiftAcervo/UPGRADING.md`](../SwiftAcervo/UPGRADING.md)

This repo has a very narrow SwiftAcervo surface (component-descriptor registration only). It does not switch over `ModelAvailability`, does not call `availability(_:)` / `ensureAvailable(...)` / `isModelAvailable(...)`, does not hand-build `CDNManifest` fixtures, and does not poke the filesystem under Acervo-managed directories. The 0.16.0 breaking changes (new `.partial` case, strict `CDNManifest` decode, slug-keyed APIs, `Acervo.swift` source decomposition) therefore have **no impact on existing code**.

## Required changes

### Package.swift / Package.resolved
- [ ] **Package.swift:60-63** — bump the SwiftAcervo `from:` floor to `"0.16.0"` so a fresh resolve cannot land on a pre-0.16 build. Current declaration:
      ```swift
      sibling(
        "SwiftAcervo",
        remote: "https://github.com/intrusive-memory/SwiftAcervo.git",
        from: "0.14.0"),
      ```
      Change `from: "0.14.0"` → `from: "0.16.0"`. Rationale: UPGRADING.md "Upgrading to 0.16.0" lists `ModelAvailability.partial` as a switch-exhaustiveness break and `CDNManifest.primaryRepo` / `CDNManifest.components` as strict-decode wire-format requirements. Even though this repo isn't currently affected, a downstream consumer that adopts this package as a transitive dep must not be silently held back to 0.15.x.
- [ ] **Package.resolved** — regenerate to pick up SwiftAcervo `0.16.x`. Do this via `make build` (or the standard SwiftPM resolve path your Makefile uses); do not hand-edit. Old pinned revision: `15fd376158be1c1d3c50a15fb8c31562034c9fc2` / `0.14.0`.

## Recommended (non-blocking) adoption

These items follow from UPGRADING.md "Step 5 — Replace any remaining filesystem-poking with library calls" and the general 0.16 philosophy of "ask the library." None of them apply to current pixart-swift-mlx code, but they are good guardrails for any future expansion of this package's runtime surface.

- [ ] If this package ever grows code that loads PixArt weights at runtime (today the recipe is just a configuration carrier — actual weight loading happens in `SwiftVinetas`), gate the load with `await Acervo.availability(modelId)` and handle all four cases including `.partial(missing:)` (UPGRADING.md "Step 1 — Handle the new `ModelAvailability.partial` case"). Do NOT introduce a private `isDownloading: Bool` flag.
- [ ] If this package ever publishes its own CDN manifests (e.g. via a future `intrusive-memory/pixart-sigma-xl-dit-*-mlx` ship workflow), use the spec-driven `acervo ship --spec ... [--dry-run --output-dir ...]` flags from UPGRADING.md "Step 6" for multi-component or renamed-slug uploads. Single-component repos under `intrusive-memory/...` can keep the existing `acervo ship <org/repo>` flow.
- [ ] If you start hand-authoring `CDNManifest` fixtures in tests, include `primaryRepo` and `components` explicitly or use the in-memory initializer that defaults them (UPGRADING.md "Step 3 — Audit any code that reads CDNManifest for primaryRepo / components"). Currently there are none in this repo.
- [ ] Stale agent-readable references to `Acervo.swift` line numbers (none found in this repo's `AGENTS.md` / `CLAUDE.md`, but worth a quick re-scan before merging the bump) — UPGRADING.md "Step 4 — Update agent-readable references after the source decomposition".

## Out of scope (verified during audit)

The following 0.16.0 migration steps from UPGRADING.md do **not** apply to this repo:

- **No `ModelAvailability` switches** anywhere in `Sources/` or `Tests/` — verified via grep for `case .notAvailable`, `case .downloading`, `case .available`, `case .partial`, and `switch.*Acervo`.
- **No `availability(_:)` / `ensureAvailable(...)` / `ensureComponentReady(...)` call sites** — verified via grep.
- **No `isModelAvailable` / `isModelConfigPresent` call sites** — verified via grep.
- **No `CDNManifest` construction or `FileManager.default.contentsOfDirectory(...)`** under Acervo-managed paths — verified via grep.
- **`Acervo.register([...])`** (`PixArtComponents.swift:52`) — signature unchanged in 0.16.0 (`Acervo+ComponentRegistration.swift:44`).
- **`Acervo.component(_:)`** (`ComponentRegistrationTests.swift:50,58,69,85,86`) — signature unchanged in 0.16.0 (`Acervo+ComponentCatalog.swift:35`).
- **`SwiftAcervo.ComponentDescriptor(id:type:displayName:repoId:minimumMemoryBytes:metadata:)`** (`PixArtComponents.swift:6,19`) — un-hydrated initializer unchanged in 0.16.0 (`ComponentDescriptor.swift:155-171`).

## Verification

Per `CLAUDE.md`, never use `swift build` or `swift test`. After applying the Package.swift bump and refreshing `Package.resolved`:

- [ ] `make build` — Debug build with the new SwiftAcervo pin.
- [ ] `make test` — runs `PixArtBackboneTests`, including `ComponentRegistrationTests` (which exercises `Acervo.register` + `Acervo.component`).
- [ ] `make lint` — swift-format pass (required before committing per `AGENTS.md:65`).
- [ ] Confirm `Package.resolved` now records SwiftAcervo `0.16.x` and the new revision.
- [ ] (Optional sanity check) `grep -rn "0.14.0" Package.swift Package.resolved` — should return no SwiftAcervo hits after the bump.
