# Releasing `loftr`

## Checklist

1. For the very first crates.io release only:
   - publish `loftr` manually from a maintainer machine
   - open the crate settings on crates.io and configure Trusted Publishing for `thomedw/loftr-rs`
2. Update crate version in `Cargo.toml` files if needed.
3. Run:
   - `just fmt`
   - `just check`
   - `just test`
   - `just publish-dry-run`
4. Confirm weights are not staged.
5. Push the version bump commit to `main`.
6. Run the GitHub `Release` workflow from `main` with:
   - `version = <crate version without the v prefix>`
   - `dry_run = true` for a rehearsal, then `false` for the real release
7. The workflow will:
   - generate release notes with `git-cliff`
   - update and commit `CHANGELOG.md`
   - publish `loftr` to crates.io via Trusted Publishing
   - create and push the annotated git tag `v<version>`
   - create the GitHub Release from the generated notes

## Trusted Publishing Notes

- The GitHub workflow does not use a long-lived `CRATES_IO_TOKEN`.
- Real publishes depend on crates.io Trusted Publishing being enabled for this crate and repository.
- The workflow requests an OIDC token with `id-token: write` and exchanges it with crates.io during the publish job.

## Local Preview

If you want to preview the release notes locally, install `git-cliff` and run:

```bash
git-cliff --config cliff.toml --unreleased --tag v0.1.0 --strip header --output target/release-notes.md
python3 scripts/update_changelog.py CHANGELOG.md target/release-notes.md
```
