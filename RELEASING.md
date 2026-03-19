# Releasing `loftr`

## Checklist

1. Confirm GitHub Actions is allowed to create and approve pull requests in the repository settings.
2. Confirm crates.io Trusted Publishing is configured for:
   - owner `thomedw`
   - repository `loftr-rs`
   - workflow file `publish.yml`
   - environment `release`
3. Run:
   - `just fmt`
   - `just check`
   - `just test`
   - `just publish-dry-run`
4. Confirm weights are not staged.
5. Merge the changes you want released into `main`.
6. The `Publish` workflow on `main` will:
   - run `release-plz release-pr`
   - open or update the automated release PR when there is a releasable commit
7. Review the generated release PR:
   - version bump in `Cargo.toml`
   - regenerated `CHANGELOG.md`
8. Merge the release PR into `main`.
9. The next `Publish` run on the merge push will:
   - authenticate to crates.io via Trusted Publishing
   - publish `loftr`
   - create and push the git tag `v<version>`
   - create the GitHub Release from the generated changelog

## Trusted Publishing Notes

- The GitHub workflow does not use `rust-lang/crates-io-auth-action`.
- The GitHub workflow does not use a long-lived `CRATES_IO_TOKEN`.
- Real publishes depend on crates.io Trusted Publishing being enabled for this crate and repository.
- The release job requests an OIDC token with `id-token: write`, and `release-plz release` exchanges it with crates.io when it needs a publish token.
- The publish job uses the GitHub environment named `release`; the crates.io trusted publisher must be configured with the same environment.
- New crates still need one manual publish before Trusted Publishing can be used. `loftr` already satisfies that bootstrap requirement.

## Protected Branch Notes

- The repository should require pull requests for `main`.
- `release-plz release-pr` updates tracked files only in its release PR; the publish step never pushes a commit to `main`.
- If the workflow uses the default `GITHUB_TOKEN`, pull request checks on the release PR will not start automatically.
- If you want release PR checks and tag-driven workflows to trigger automatically, switch the workflow to a GitHub App token or machine-user PAT later.
- Prefer enabling “require branches to be up to date before merging” for the release PR so the generated changelog matches the commit set being released.

## Local Preview

If you want a local simulation, use a disposable clone or worktree and run:

```bash
cargo install --locked release-plz
git clone . /tmp/loftr-rs-release-preview
release-plz update --manifest-path /tmp/loftr-rs-release-preview/Cargo.toml --allow-dirty
```
