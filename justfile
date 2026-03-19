set positional-arguments

fmt:
    cargo fmt --all

prepare-test-fixtures:
    ./scripts/prepare_test_fixtures.sh

check:
    cargo check --workspace --all-targets --features download-libtorch

clippy:
    cargo clippy --workspace --all-targets --all-features -- -D warnings

test:
    ./scripts/prepare_test_fixtures.sh
    cargo test --workspace --features download-libtorch

publish-dry-run:
    cargo publish --dry-run -p loftr --locked --features download-libtorch

release-notes version:
    mkdir -p target/release
    git-cliff --config cliff.toml --unreleased --tag v{{version}} --strip header --output target/release/release-notes-v{{version}}.md

changelog version:
    git-cliff --config cliff.toml --tag v{{version}} --output CHANGELOG.md
