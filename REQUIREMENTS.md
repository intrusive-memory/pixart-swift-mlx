---
title: "pixart-swift-mlx — SwiftAcervo Integration Requirements"
date: 2026-04-18
version: "1.0"
status: "READY FOR EXECUTION"
priority: "🟡 MEDIUM"
---

# pixart-swift-mlx — SwiftAcervo Integration Requirements

**Mission Context**: Part of SwiftAcervo Consumer Adoption Wave 2+  
**Master Index**: `/Users/stovak/Projects/REQUIREMENTS.md`  
**Audit Source**: `/Users/stovak/Projects/ACERVO_CONSUMER_AUDIT.md` (lines 403–406, 428, addendum §CDN Upload)  
**Architecture Spec**: `./REQUIREMENTS.md` (pre-existing DRAFT, this file complements it)

---

## Audit Findings

### Current State (Pre-Sortie)

| Finding | Status | Impact |
|---------|--------|--------|
| Has only `tests.yml` CI | ❌ Missing CDN | Components NOT on private CDN |
| PixArt DiT NOT on Acervo CDN | ❌ Custom HF download | First-run slow, no centralized caching |
| No CDN upload workflow | ❌ Workflow missing | Cannot ship model to R2 |
| No `acervo ship` integration | ❌ Manual process | Duplicates SwiftBruja/SwiftProyecto CDN logic |

**Critical Issue**: No centralized model caching via SwiftAcervo. PixArt DiT (and T5-XXL, SDXL VAE catalog components) must be uploaded to private R2 CDN using `acervo ship` command.

**Reference**: ACERVO_CONSUMER_AUDIT.md § Addendum (lines 249–535) — Standardize on `acervo` CLI for all CDN workflows.

---

## Sorties

### Sortie 1: Baseline Acervo Integration Assessment

**Objective**: Audit current component descriptor implementation and identify gaps vs. audit requirements.

**Entry Criteria**:
- pixart-swift-mlx repo cloned at `/Users/stovak/Projects/pixart-swift-mlx/`
- Audit findings read (this file, lines 1–30)

**Tasks**:
1. Verify `ComponentDescriptor` definitions exist for:
   - `pixart-sigma-xl-dit-int4` (backbone — owned by this package)
   - `t5-xxl-encoder-int4` (catalog, re-registered for safety)
   - `sdxl-vae-decoder-fp16` (catalog, re-registered for safety)
2. Check Acervo registration code:
   - `enum PixArtComponents` exists
   - Static `registered: Bool` initializer runs at module load
   - All three descriptors have HuggingFace repos, SHA-256 checksums, manifest URLs
3. Validate registration call sites:
   - Pipeline assembly calls `_ = PixArtComponents.registered` before building
   - No other code paths bypass registration
4. Audit component access patterns:
   - All model loads use `AcervoManager.shared.withComponentAccess(id)`
   - Zero direct file path access to model files
   - Zero hardcoded `/Library/SharedModels/` paths

**Exit Criteria**:
- Assessment document created: `./INTEGRATION_BASELINE.md`
  - List of found vs. missing descriptors
  - Code locations (file + line) for each finding
  - Gaps vs. audit requirement: "All components must register + access via withComponentAccess()"
- Exit status: PASS (all registered + all accessed via closure) OR FAIL (gaps documented)

**Owner**: TBD  
**Time Estimate**: 1–2 hours

---

### Sortie 2: Create PixArt ComponentDescriptor (if missing)

**Objective**: If Sortie 1 finds missing descriptor for `pixart-sigma-xl-dit-int4`, create complete definition with HuggingFace metadata.

**Entry Criteria**:
- Sortie 1 PASS or FAIL
- If FAIL: Gap identified is missing `pixart-sigma-xl-dit-int4` descriptor

**Tasks**:
1. Determine canonical HuggingFace repo for PixArt DiT:
   - Must be under `intrusive-memory/` org (per REQUIREMENTS.md P5)
   - Naming: `intrusive-memory/pixart-sigma-xl-dit-int4-mlx`
   - Must contain safetensors weights + `config.json`
2. Create `ComponentDescriptor`:
   - `id: "pixart-sigma-xl-dit-int4"`
   - `type: .backbone`
   - `huggingFaceRepo: "intrusive-memory/pixart-sigma-xl-dit-int4-mlx"`
   - `manifestUrl: "https://intrusive-memory.r2.dev/pixart-sigma-xl-dit-int4-mlx/manifest.json"` (or current CDN URL)
   - `expectedSha256: "<32-byte-hex>"` (fetch from live manifest.json)
   - `size: "~300 MB"` (documented in REQUIREMENTS.md P5)
3. Register in `PixArtComponents.registered` initializer
4. Validate conformance:
   - Matches audit requirement: HuggingFace source + Acervo CDN metadata
   - Matches REQUIREMENTS.md P5 metadata exactly

**Exit Criteria**:
- `ComponentDescriptor` defined in source code (file path documented)
- Registered in `PixArtComponents.registered`
- Passes `make test` without new failures

**Owner**: TBD  
**Time Estimate**: 1–2 hours  
**Depends On**: Sortie 1

---

### Sortie 3: Create CDN Upload Workflow

**Objective**: Create `.github/workflows/ensure-model-cdn.yml` using `acervo ship` to upload PixArt DiT, T5-XXL, SDXL VAE to R2 CDN.

**Entry Criteria**:
- Sorties 1–2 complete (ComponentDescriptor in place)
- Access to `acervo` CLI (installed at `~/.local/bin/acervo` or via `cargo` if needed)
- GitHub repo secrets: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` (R2 credentials)
- CDN R2 bucket details (account, endpoint, bucket name)

**Design Principles**:
- **Single source of truth**: Use `acervo ship` command (never manual curl/shasum/jq)
- **Audit alignment**: Follow pattern in ACERVO_CONSUMER_AUDIT.md § "What Acervo Provides" (line 434–446)
- **Reference implementations**: SwiftBruja & SwiftProyecto `.github/workflows/ensure-model-cdn.yml` (custom logic) are ANTI-PATTERNS; this workflow must be SIMPLER using `acervo` CLI
- **Trigger**: `workflow_dispatch` (manual) + push to `main` (automatic)

**Tasks**:
1. Create `.github/workflows/ensure-model-cdn.yml`:
   - Trigger: `on: [push: branches: [main], workflow_dispatch]`
   - Job: `upload-models`
   - Runner: `macos-26` (per CLAUDE.md SwiftBuild requirements)
   - Steps:
     a. Checkout code
     b. Install `acervo` CLI (if not already on runner)
     c. Authenticate to R2 (set AWS credentials)
     d. Run `acervo ship --model-id "intrusive-memory/pixart-sigma-xl-dit-int4-mlx"`
        - Automatically downloads from HF, generates manifest.json with SHA-256, uploads to R2
     e. Run `acervo ship --model-id "intrusive-memory/t5-xxl-int4-mlx"` (catalog component)
     f. Run `acervo ship --model-id "intrusive-memory/sdxl-vae-fp16-mlx"` (catalog component)
     g. Verify uploads: `acervo verify` on each uploaded manifest
2. Document workflow in README or AGENTS.md:
   - When to trigger manually (e.g., "after weight conversion")
   - Expected outputs (manifest URLs, CDN links)
3. Update ComponentDescriptor manifest URLs (if CDN paths differ):
   - Extract from workflow logs or Acervo output
   - Update `manifestUrl` field in Sortie 2

**Exit Criteria**:
- Workflow file created at `.github/workflows/ensure-model-cdn.yml`
- Passes GitHub Actions syntax check (no parse errors)
- References only `acervo ship` (no manual curl/shasum/jq/aws-cli for manifest generation)
- Documented in project README or AGENTS.md

**Owner**: TBD  
**Time Estimate**: 2–3 hours  
**Depends On**: Sorties 1–2

---

### Sortie 4: Standardize Manifest Format

**Objective**: Ensure PixArt ComponentDescriptor manifest.json format matches Acervo schema (not custom SwiftBruja/SwiftProyecto format).

**Entry Criteria**:
- Sortie 3 complete (workflow generates manifest.json on R2 CDN)
- Access to live manifest.json at CDN URL

**Tasks**:
1. Fetch manifest.json from CDN:
   - `curl "https://intrusive-memory.r2.dev/pixart-sigma-xl-dit-int4-mlx/manifest.json"`
2. Validate schema against Acervo spec:
   - Reference: SwiftAcervo repository `docs/manifest-schema.md` (if exists) OR
   - Reference: ACERVO_CONSUMER_AUDIT.md § "Custom manifest schema" (lines 307–310) for what NOT to do
3. Check required fields:
   - `version` (should be `"2.0"` for standardized Acervo format)
   - `components` array with `id`, `sha256`, `url`, `size`
   - `manifestChecksum` (SHA-256 of entire manifest)
4. Document findings:
   - If format matches Acervo spec: ✅ PASS
   - If format differs: Create task to align with SwiftAcervo schema

**Exit Criteria**:
- Manifest format audit document created
- Format matches Acervo spec (version, required fields)
- No references to SwiftBruja/SwiftProyecto custom schemas

**Owner**: TBD  
**Time Estimate**: 1 hour  
**Depends On**: Sortie 3

---

### Sortie 5: Remove Direct File Path Access

**Objective**: Audit codebase for hardcoded `/Library/SharedModels/`, HuggingFace paths, or `FileManager` access to models. Replace with `withComponentAccess()`.

**Entry Criteria**:
- Sorties 1–2 complete (descriptors registered)
- Sortie 3 complete (CDN workflow in place)

**Tasks**:
1. Search codebase for model file access patterns:
   - Grep for `/Library/SharedModels/`
   - Grep for hardcoded HuggingFace `huggingface.co/` or `hf_token` references
   - Grep for `FileManager` calls to model directories
   - Grep for direct `Bundle.main.path()` or `URL(fileURLWithPath:)` for models
2. For each finding, determine:
   - Is this legacy code (pre-Acervo)?
   - Should this use `AcervoManager.shared.withComponentAccess(componentId)` instead?
3. Create migration list:
   - Old pattern → New pattern mapping
   - Call sites needing refactoring
4. If refactoring is in scope: Execute replacements
   - Ensure all model access is within `withComponentAccess { ... }` closure
5. If refactoring is out of scope: Document as follow-up task

**Exit Criteria**:
- Audit document created: `./MODEL_ACCESS_AUDIT.md`
- List of direct file access patterns found (or "none found" if clean)
- Call sites documented with refactoring recommendations
- If refactored: `make test` passes without new failures

**Owner**: TBD  
**Time Estimate**: 1–2 hours  
**Depends On**: Sorties 1–2

---

## Key Audit References

### ACERVO_CONSUMER_AUDIT.md Sections

| Section | Lines | Topic |
|---------|-------|-------|
| pixart-swift-mlx findings | 403–406 | NO CDN workflow, components NOT on CDN |
| Impact assessment | 410–430 | 3 projects missing CDN (mlx-audio, SwiftTuberia, pixart-swift-mlx) |
| What Acervo Provides | 434–446 | `acervo ship` command single source of truth |
| Required Changes Phase 1 | 450+ | Standardize on `acervo` CLI |

### Master Index

- **Parent**: `/Users/stovak/Projects/REQUIREMENTS.md` — Complete mission index for all 6 consumer projects
- **Wave 2 Status**: `/Users/stovak/Projects/MEMORY.md` — Wave 2 completion summary (SwiftBruja, SwiftProyecto, mlx-audio-swift, SwiftVoxAlta all complete ✅)

---

## Acceptance Criteria (This File)

- [ ] Sortie 1: Baseline assessment complete, INTEGRATION_BASELINE.md created
- [ ] Sortie 2: ComponentDescriptor for PixArt DiT created/verified
- [ ] Sortie 3: CDN upload workflow created and functional
- [ ] Sortie 4: Manifest format audit complete
- [ ] Sortie 5: Direct file path access audit complete or refactored
- [ ] All sorties documented in memory or MEMORY.md

---

## Notes

### Architecture Spec Compatibility

This file **complements** the pre-existing `/Users/stovak/Projects/pixart-swift-mlx/REQUIREMENTS.md` (P1–P11 sections). That document covers:
- Package structure, dependencies, platforms
- PixArt DiT backbone implementation
- Weight key mapping, pipeline recipe
- **Acervo component descriptors are mentioned in P5** but implementation details (manifest URLs, SHA-256) are NOW addressed by this file (Sorties 1–4)

### Why 🟡 MEDIUM Priority

1. ✅ **Core functionality exists** — PixArt backbone, weight loading, pipeline all implemented
2. ❌ **CDN integration missing** — Models currently download from HuggingFace on first run (slow, not cached)
3. ⏱️ **Not blocking immediate use** — Package can be built and tested locally
4. 🔗 **Prerequisite for Wave 3** — Once complete, enables SwiftVinetas integration (P10) and broad adoption

### Time Estimate

- **Total sorties**: 5
- **Per-sortie**: 1–3 hours
- **Parallel work**: Sorties 1–2 can run in parallel with Sorties 3–5 (minor dependency on Sortie 1 completion)
- **Total elapsed**: ~1 week if 1 agent, ~3–4 days if 2 agents

---

**Mission Status**: Ready for agent assignment. Each sortie is self-contained and measurable.
