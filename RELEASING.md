# Releasing `loftr`

## Checklist

1. Update crate version in `Cargo.toml` files if needed.
2. Run:
   - `just fmt`
   - `just check`
   - `just test`
   - `just publish-dry-run`
3. Confirm weights are not staged.
4. Push the version bump commit to `main`.
5. Run the GitHub `Release` workflow from `main` with:
   - `version = <crate version without the v prefix>`
   - `dry_run = true` for a rehearsal, then `false` for the real release
6. The workflow will:
   - generate release notes with `git-cliff`
   - update and commit `CHANGELOG.md`
   - publish `loftr` to crates.io
   - create and push the annotated git tag `v<version>`
   - create the GitHub Release from the generated notes

## Local Preview

If you want to preview the release notes locally, install `git-cliff` and run:

```bash
git-cliff --config cliff.toml --unreleased --tag v0.1.0 --strip header --output target/release-notes.md
python3 scripts/update_changelog.py CHANGELOG.md target/release-notes.md
```
