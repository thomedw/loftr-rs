# Releasing `loftr`

## Checklist

1. Confirm crates.io Trusted Publishing is configured for:
   - owner `thomedw`
   - repository `loftr-rs`
   - workflow file `publish.yml`
   - environment `release`
2. Run:
   - `just fmt`
   - `just check`
   - `just test`
   - `just publish-dry-run`
3. Confirm weights are not staged.
4. Run the GitHub `Prepare Release PR` workflow from `main`.
5. Review the generated PR:
   - version bump in `Cargo.toml`
   - regenerated `CHANGELOG.md`
   - release notes in the PR body
6. If additional commits land on `main`, rerun `Prepare Release PR` before merge so the release PR stays current.
7. Merge the release PR into `main`.
8. The `Publish Release` workflow will then:
   - authenticate to crates.io via Trusted Publishing
   - publish `loftr`
   - create and push the annotated git tag `v<version>`
   - create the GitHub Release from the generated notes

## Trusted Publishing Notes

- The GitHub workflow does not use a long-lived `CRATES_IO_TOKEN`.
- Real publishes depend on crates.io Trusted Publishing being enabled for this crate and repository.
- The workflow requests an OIDC token with `id-token: write` and exchanges it with crates.io during the publish job.
- The publish job uses the GitHub environment named `release`; the crates.io trusted publisher must be configured with the same environment.

## Protected Branch Notes

- The repository should require pull requests for `main`.
- The publish workflow must not push commits to `main`; only the release PR updates tracked files.
- Prefer enabling “require branches to be up to date before merging” for the release PR so the generated changelog matches the commit set being released.

## Local Preview

If you want to preview the release notes locally, install `git-cliff` and run:

```bash
git-cliff --config cliff.toml --unreleased --tag v0.1.0 --strip header --output target/release-notes.md
git-cliff --config cliff.toml --tag v0.1.0 --output CHANGELOG.md
```
