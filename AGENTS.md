# AGENTS.md

Guidance for AI agents and human contributors working in this repository.

## Project Overview

- `loftr-rs` is a Rust workspace for a native `tch`-based implementation of LoFTR.
- The current public surface is intentionally small.
- The only publishable crate today is `crates/loftr`.
- Pretrained weights are generated locally and must not be committed to git.

## Build And Verification

Run these from the repository root.

### Preferred Commands

```bash
cargo fmt --all
cargo check --workspace --all-targets --features download-libtorch
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --features download-libtorch
cargo test --doc -p loftr --features doc-only
cargo deny check
cargo publish --dry-run -p loftr --locked --features download-libtorch
```

### `just` Shortcuts

If `just` is installed, these map to the common local commands:

```bash
just fmt
just check
just clippy
just test
just publish-dry-run
```

## Repository Layout

```text
Cargo.toml                  workspace root
crates/loftr/              publishable library crate
crates/loftr/examples/     runnable examples
crates/loftr/tests/        integration tests
scripts/                   weight export, validation, changelog helpers
.github/workflows/         CI and release automation
```

## Code Conventions

- Rust edition is `2024`; MSRV is `1.85.0`.
- Keep the public API high-level and narrow. Avoid exposing internal LoFTR building blocks unless there is a clear product need.
- Use `LoftrError` for fallible library behavior. Do not introduce panics in library code.
- Avoid `unwrap()` and `expect()` in non-test code.
- Keep module boundaries clean; prefer internal modules over expanding `pub mod` surface.
- When changing public APIs, update both the root `README.md` and `crates/loftr/README.md`.
- Follow Conventional Commits for all commits: `feat:`, `fix:`, `docs:`, `ci:`, `refactor:`, `test:`, `chore:`.

## Weights And Artifacts

- Never commit model weights, exported checkpoints, or other large generated artifacts.
- Generated weights belong under `artifacts/weights/`, which is intentionally gitignored.
- Use `./scripts/generate_loftr_state_dict.sh` to export the local LoFTR safetensors file.
- Use `./scripts/validate_loftr_against_kornia.sh <left> <right>` for parity checks against the Kornia Python reference.

## Documentation

- Keep public documentation self-contained and project-focused.
- Do not add internal-source-history notes to the public README or crate metadata.
- When release behavior changes, update `RELEASING.md` and any affected workflow files in the same change.
- The release flow uses `git-cliff` and a tracked `CHANGELOG.md`; do not hand-edit release sections if the workflow is the source of truth.

## Release Notes

- The first crates.io publish for a crate must be done manually before Trusted Publishing can be enabled.
- After that, the GitHub `Release` workflow is the intended publish path.
- Do not reintroduce a long-lived `CRATES_IO_TOKEN` secret into the workflow without a clear reason.
