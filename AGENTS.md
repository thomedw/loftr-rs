# AGENTS.md

Guidance for AI agents and human contributors working in this repository.

## Project Overview

**loftr-rs** is a Rust workspace for a native `tch`-based implementation of LoFTR.

- The workspace currently contains one publishable crate: `crates/loftr`
- The public API is intentionally small and high-level
- Pretrained weights are generated locally and must not be committed to git

---

## Build & Development Commands

Run commands from the repository root.

### Rust

```bash
cargo fmt --all
cargo check --workspace --all-targets --features download-libtorch
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --features download-libtorch
cargo test --doc -p loftr --features doc-only
cargo deny check
cargo publish --dry-run -p loftr --locked --features download-libtorch
```

### With `just`

If `just` is installed, these shortcuts map to the common local workflow:

```bash
just fmt
just check
just clippy
just test
just publish-dry-run
just release-notes 0.1.0
just changelog 0.1.0
```

### Scripts

```bash
./scripts/generate_loftr_state_dict.sh
./scripts/validate_loftr_against_kornia.sh <left> <right>
python3 scripts/update_changelog.py CHANGELOG.md target/release-notes.md
```

---

## Architecture

### Workspace Structure

```text
Cargo.toml              ← workspace root
crates/loftr/           ← publishable library crate
crates/loftr/examples/  ← runnable example binaries
crates/loftr/tests/     ← integration tests
scripts/                ← export, validation, and changelog helpers
.github/workflows/      ← CI and release automation
```

### Public Surface

The public API should stay high-level unless there is a clear reason to expand it.

Current top-level exports include:

- `LoftrConfig`
- `LoftrModel`
- `LoftrMatches`
- `LoftrDebugStages`
- `normalize_loftr_image`
- `LoftrError`

Internal LoFTR building blocks such as the backbone, transformer, attention, and matching modules should remain private unless there is an explicit API decision to expose them.

### Weights And Artifacts

- Generated weights belong under `artifacts/weights/`
- `artifacts/weights/` is intentionally gitignored
- Never commit `.safetensors`, `.pt`, `.pth`, `.onnx`, or other generated model artifacts

---

## Code Conventions

### General

- Rust edition **2024**, MSRV **1.85.0**
- Run `rustfmt`, `clippy`, and the relevant tests before every commit
- Warnings are denied in CI
- Prefer **borrowing over cloning**, especially for tensors and image buffers
- Keep the public API narrow and stable
- No `unwrap()` or `expect()` in library code — propagate errors with `?`
- Follow [Conventional Commits](https://www.conventionalcommits.org): `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `ci:`, `chore:`

### Documentation

Every public item should have a doc comment.

For public functions, prefer this structure when applicable:

```rust
/// One-line summary.
///
/// Longer explanation if needed.
///
/// # Arguments
///
/// * `image0` - Left grayscale image tensor.
/// * `image1` - Right grayscale image tensor.
///
/// # Returns
///
/// Matching outputs or an error.
///
/// # Errors
///
/// Returns [`LoftrError`] if the input shapes are invalid or weights cannot be loaded.
///
/// # Example
///
/// ```rust
/// # use loftr::{LoftrConfig, LoftrModel};
/// # use tch::Device;
/// let model = LoftrModel::new(Device::Cpu, LoftrConfig::outdoor())?;
/// # Ok::<(), loftr::LoftrError>(())
/// ```
```

Rules:

- Document all `pub` functions, structs, enums, and traits
- Add `# Errors` when the function can fail
- Include examples for non-trivial public APIs
- Keep examples compilable when practical

### Safety

- Avoid `unsafe` unless it is strictly necessary
- Every `unsafe` block must be preceded by a `// SAFETY:` comment explaining why it is sound
- Prefer safe abstractions over raw pointer manipulation

### Error Handling

- Use `thiserror` for library-facing error types
- Prefer descriptive error variants with context
- Do not use `panic!()`, `.unwrap()`, or `.expect()` in library code
- In tests, `.expect()` is acceptable when it makes failures clearer

### Performance

- Avoid unnecessary allocations in hot paths
- Do not clone tensors or intermediate results unless required
- Keep internal modules focused and avoid growing the public API surface as a side effect of refactors

---

## Contributing

### Pull Requests

1. Keep PRs focused — one concern per PR when possible.
2. Test locally before opening a PR. At minimum, run:

   ```bash
   cargo fmt --all
   cargo clippy --workspace --all-targets --all-features -- -D warnings
   cargo test --workspace --features download-libtorch
   cargo deny check
   ```

3. Update documentation when public APIs, release flow, or validation flow changes.
4. New functionality should include tests. Bug fixes should include regression coverage when practical.

### Contributing To Documentation

- Keep public docs self-contained and project-focused
- Update both `README.md` and `crates/loftr/README.md` when public usage changes
- Update `RELEASING.md` when release behavior changes
- Do not add internal-source-history notes to public README or crate metadata

### Release Workflow

- The first crates.io release for a crate must be published manually before Trusted Publishing can be enabled
- After that, the GitHub `Release` workflow is the intended publish path
- The release workflow uses `git-cliff`, updates `CHANGELOG.md`, publishes the crate, pushes a tag, and creates a GitHub Release
- Do not reintroduce a long-lived `CRATES_IO_TOKEN` secret without a clear reason

---

## Quick Reference Checklist

- [ ] `cargo fmt --all` passes
- [ ] `cargo clippy --workspace --all-targets --all-features -- -D warnings` passes
- [ ] `cargo test --workspace --features download-libtorch` passes
- [ ] `cargo deny check` passes
- [ ] Public API docs were updated if needed
- [ ] No model weights or generated artifacts are staged
- [ ] Commit message follows Conventional Commits
