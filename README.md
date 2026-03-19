# `loftr-rs`

Rust workspace for a native [`tch`](https://crates.io/crates/tch)-based implementation of LoFTR.

`loftr-rs` is currently alpha-quality software. Expect APIs, docs, and runtime behavior to change
before `1.0`, and validate results carefully before relying on it in production workflows.

## Demo

<table>
  <tr>
    <td align="center"><strong>Kornia reference</strong></td>
    <td align="center"><strong><code>loftr-rs</code> visualizer</strong></td>
  </tr>
  <tr>
    <td><img src="docs/images/kornia-matching-loftr.jpg" alt="Original Kornia LoFTR image matching demo" width="100%" /></td>
    <td><img src="docs/images/loftr-rs-demo-rust.png" alt="Rust-generated loftr-rs image matching demo" width="100%" /></td>
  </tr>
</table>

The left image is the original Kornia LoFTR application demo from the
[Image Matching docs](https://kornia.readthedocs.io/en/latest/applications/image_matching.html).
The right image is generated locally with the Rust visualizer example in this repo, using the same
`kn_church-2.jpg` and `kn_church-8.jpg` pair that Kornia uses in its LoFTR tutorial. Kornia resizes
that pair to `600x375` (`H x W`); the current native port rounds the width up to `376` to keep the
fine-matching scale integral. The visualizer uses a confidence-colored top-k rendering similar to
the original LoFTR demo:

```bash
curl -L -o /tmp/kn_church-2.jpg https://github.com/kornia/data/raw/main/matching/kn_church-2.jpg
curl -L -o /tmp/kn_church-8.jpg https://github.com/kornia/data/raw/main/matching/kn_church-8.jpg
cargo run -p loftr --example render_demo --features download-libtorch -- \
  /path/to/loftr_outdoor_state_dict.safetensors \
  /tmp/kn_church-2.jpg \
  /tmp/kn_church-8.jpg \
  docs/images/loftr-rs-demo-rust.png
```

## Current Scope

The first public release is intentionally small:

- one publishable crate: `loftr`
- native Rust/tch LoFTR model construction and inference
- local weight export and validation scripts

Not included in v0.1:

- panorama stitching helpers
- calibration pipelines
- application-specific runtime glue

## Public API

The main crate exports a small, high-level surface:

- `LoftrConfig` for preset and custom model configuration
- `LoftrModel` for model construction, weight loading, and inference
- `LoftrMatches` for match outputs
- `LoftrDebugStages` for validation-oriented stage summaries
- `normalize_loftr_image` for converting supported image layouts into LoFTR input tensors
- `LoftrError` for public error handling

## Upstream References

- [LoFTR upstream](https://github.com/zju3dv/LoFTR)
- [Kornia upstream](https://github.com/kornia/kornia)
- [kornia-rs upstream](https://github.com/kornia/kornia-rs)

## Toolchain

- Rust edition: `2024`
- MSRV: `1.85.0`
- CI target: MSRV plus latest stable Rust

## Weights

Pretrained weights are not committed to git and are not bundled into the published crate.

Generate local weights with:

```bash
./scripts/generate_loftr_state_dict.sh
```

By default, this writes:

```text
artifacts/weights/loftr_outdoor_state_dict.safetensors
```

## Test Fixtures

The weighted end-to-end test now uses prepared fixtures under `crates/loftr/tests/data/`.

Bootstrap them locally with:

```bash
./scripts/prepare_test_fixtures.sh
```

or:

```bash
just prepare-test-fixtures
```

CI prepares these fixtures automatically. If you run `cargo test` without them, the end-to-end
test fails with a setup message instead of being ignored.
`just test` bootstraps them for you before running the workspace test suite.

## Libtorch Setup

This crate uses `tch`, so you need a working libtorch setup.

Common options:

```bash
LIBTORCH_USE_PYTORCH=1 cargo test
```

or:

```bash
LIBTORCH=/path/to/libtorch cargo test
```

If you want automatic downloads, enable the feature that forwards to `tch`:

```bash
cargo test -p loftr --features download-libtorch
```

The published docs use the `doc-only` feature on docs.rs so they can render
without linking libtorch.

## Workspace Commands

```bash
just fmt
just check
just prepare-test-fixtures
just test
just coverage
just coverage-html
just publish-dry-run
```

Install `cargo-llvm-cov` once if you want to run the local coverage commands:

```bash
cargo install cargo-llvm-cov
```

## Validation

For local parity checks against Kornia's Python LoFTR reference, use:

```bash
./scripts/validate_loftr_against_kornia.sh /path/to/left.jpg /path/to/right.jpg
```
