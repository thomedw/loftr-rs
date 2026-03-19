set positional-arguments

fmt:
    cargo fmt --all

check:
    cargo check --workspace --all-targets --features download-libtorch

clippy:
    cargo clippy --workspace --all-targets --all-features -- -D warnings

test:
    cargo test --workspace --features download-libtorch

publish-dry-run:
    cargo publish --dry-run -p loftr --features download-libtorch
