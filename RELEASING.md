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
4. The repo does not track `Cargo.lock`; local commands may generate it, but git ignores it.
5. Run the GitHub `Prepare Release PR` workflow from `main`.
6. If the current workspace version is not fully released yet, `Prepare Release PR` will recover the missing release artifacts first by dispatching `Publish Release` on the original release-prep commit.
7. Review the generated PR:
   - version bump in `Cargo.toml`
   - regenerated `CHANGELOG.md`
   - release notes in the PR body
8. If additional commits land on `main`, rerun `Prepare Release PR` before merge so the release PR stays current.
9. Merge the release PR into `main`.
10. The merge push to `main` will trigger the `Publish Release` workflow, which will:
   - authenticate to crates.io via Trusted Publishing
   - publish `loftr`
   - create and push the annotated git tag `v<version>`
   - create the GitHub Release from the generated notes

## Trusted Publishing Notes

- The GitHub workflow does not use a long-lived `CRATES_IO_TOKEN`.
- Real publishes depend on crates.io Trusted Publishing being enabled for this crate and repository.
- The workflow requests an OIDC token with `id-token: write` and exchanges it with crates.io during the publish job.
- The publish job uses the GitHub environment named `release`; the crates.io trusted publisher must be configured with the same environment.
- crates.io Trusted Publishing does not support the `pull_request_target` event, so the publish workflow intentionally runs from the `push` to `main` created by merging the automated release PR.
- Recovery dispatches also use `publish.yml`, so there is a single publish path for normal releases and missed-release repair.

## Protected Branch Notes

- The repository should require pull requests for `main`.
- The publish workflow must not push commits to `main`; only the release PR updates tracked files.
- The publish workflow validates that the `main` push came from the workflow-generated `release/v*` pull request before it attempts a publish.
- `Prepare Release PR` may recover multiple unfinished prepared releases in order before it opens the next release PR.
- Prefer enabling “require branches to be up to date before merging” for the release PR so the generated changelog matches the commit set being released.

## Local Preview

If you want to preview the release notes locally, install `git-cliff` and run:

```bash
git-cliff --config cliff.toml --unreleased --tag v0.1.0 --strip header --output target/release-notes.md
git-cliff --config cliff.toml --tag v0.1.0 --output CHANGELOG.md
```
