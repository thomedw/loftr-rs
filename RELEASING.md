# Releasing `loftr`

## Checklist

1. Update crate version in `Cargo.toml` files if needed.
2. Run:
   - `just fmt`
   - `just check`
   - `just test`
   - `just publish-dry-run`
3. Confirm weights are not staged.
4. Push the release commit.
5. Run the GitHub `publish` workflow with `dry_run = false`.

