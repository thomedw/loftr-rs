#!/usr/bin/env python3
"""Release automation helpers shared by GitHub Actions workflows."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import urlopen


PREPARED_RELEASE_RE = re.compile(r"^chore\(release\): prepare (v\d+\.\d+\.\d+)(?: \(#\d+\))?$")
RELEASE_BRANCH_RE = re.compile(r"release/v\d+\.\d+\.\d+")


@dataclass(frozen=True)
class PreparedRelease:
    version: str
    tag: str
    commit: str


@dataclass(frozen=True)
class ReleaseState:
    crate_published: bool
    tag_exists: bool
    tag_target: str
    github_release_exists: bool
    complete: bool


def run_command(
    args: list[str],
    *,
    check: bool = True,
    capture: bool = True,
    env: dict[str, str] | None = None,
) -> str:
    merged_env = os.environ.copy()
    if env is not None:
        merged_env.update(env)

    result = subprocess.run(
        args,
        check=check,
        capture_output=capture,
        env=merged_env,
        text=True,
    )
    if capture:
        return result.stdout.strip()
    return ""


def git(*args: str, capture: bool = True, check: bool = True) -> str:
    return run_command(["git", *args], capture=capture, check=check)


def gh(*args: str, capture: bool = True, check: bool = True) -> str:
    return run_command(["gh", *args], capture=capture, check=check)


def gh_api(
    endpoint: str,
    *,
    method: str = "GET",
    fields: dict[str, str] | None = None,
    headers: list[str] | None = None,
    expect_json: bool = True,
) -> Any:
    args = ["gh", "api", endpoint]
    if method != "GET":
        args.extend(["--method", method])
    for header in headers or []:
        args.extend(["-H", header])
    for key, value in (fields or {}).items():
        args.extend(["-f", f"{key}={value}"])

    output = run_command(args, capture=True)
    if not expect_json:
        return output
    if not output:
        return None
    return json.loads(output)


def append_outputs(values: dict[str, str | bool]) -> None:
    output_path = os.environ.get("GITHUB_OUTPUT")
    lines = []
    for key, value in values.items():
        if isinstance(value, bool):
            rendered = "true" if value else "false"
        else:
            rendered = value
        lines.append(f"{key}={rendered}")

    if output_path:
        with Path(output_path).open("a", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")
    else:
        print("\n".join(lines))


def append_summary(text: str) -> None:
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with Path(summary_path).open("a", encoding="utf-8") as handle:
            handle.write(text)


def fail(message: str) -> None:
    raise SystemExit(message)


def cargo_version() -> str:
    metadata = run_command(["cargo", "metadata", "--no-deps", "--format-version", "1"])
    data = json.loads(metadata)
    package = next(pkg for pkg in data["packages"] if pkg["name"] == "loftr")
    return str(package["version"])


def latest_tag() -> str:
    return git("tag", "--list", "v[0-9]*", "--sort=-version:refname")


def first_tag_line(tags_output: str) -> str:
    return tags_output.splitlines()[0] if tags_output else ""


def crates_version_exists(version: str) -> bool:
    url = f"https://crates.io/api/v1/crates/loftr/{version}"
    try:
        with urlopen(url) as response:
            return response.status == 200
    except HTTPError as error:
        if error.code == 404:
            return False
        raise
    except URLError as error:
        raise SystemExit(f"failed to query crates.io for version {version}: {error}") from error


def github_release_exists(repo: str, tag: str) -> bool:
    result = subprocess.run(
        ["gh", "release", "view", tag, "--repo", repo],
        capture_output=True,
        text=True,
        env=os.environ.copy(),
        check=False,
    )
    return result.returncode == 0


def tag_state(tag: str) -> tuple[bool, str]:
    result = subprocess.run(
        ["git", "rev-parse", "-q", "--verify", f"refs/tags/{tag}"],
        capture_output=True,
        text=True,
        env=os.environ.copy(),
        check=False,
    )
    if result.returncode != 0:
        return False, ""
    return True, git("rev-list", "-n1", tag)


def release_state(repo: str, prepared: PreparedRelease) -> ReleaseState:
    crate_published = crates_version_exists(prepared.version)
    tag_exists, tag_target = tag_state(prepared.tag)
    release_exists = github_release_exists(repo, prepared.tag)
    complete = (
        crate_published
        and tag_exists
        and release_exists
        and tag_target == prepared.commit
    )
    return ReleaseState(
        crate_published=crate_published,
        tag_exists=tag_exists,
        tag_target=tag_target,
        github_release_exists=release_exists,
        complete=complete,
    )


def list_prepared_releases(revision: str = "origin/main") -> list[PreparedRelease]:
    output = git("log", "--reverse", "--format=%H%x1f%s", revision)
    releases: list[PreparedRelease] = []
    for line in output.splitlines():
        commit, subject = line.split("\x1f", 1)
        match = PREPARED_RELEASE_RE.match(subject)
        if match is None:
            continue
        tag = match.group(1)
        releases.append(
            PreparedRelease(
                version=tag.removeprefix("v"),
                tag=tag,
                commit=commit,
            )
        )
    return releases


def compute_next_version(current_version: str, subjects_and_bodies: str) -> str:
    major, minor, patch = map(int, current_version.split("."))
    records = [record for record in subjects_and_bodies.split("\x1e") if record.strip()]

    breaking = False
    has_feat = False
    for record in records:
        subject, body = (record.split("\x1f", 1) + [""])[:2]
        subject = subject.strip()
        body = body.strip()
        if re.match(r"^[a-z]+(?:\([^)]+\))?!:", subject) or "BREAKING CHANGE:" in body or "BREAKING-CHANGE:" in body:
            breaking = True
        if re.match(r"^feat(?:\([^)]+\))?:", subject):
            has_feat = True

    if breaking:
        return f"{major + 1}.0.0"
    if has_feat:
        return f"{major}.{minor + 1}.0"
    return f"{major}.{minor}.{patch + 1}"


def parse_release_notes(tag: str, changelog_path: Path) -> str:
    changelog = changelog_path.read_text(encoding="utf-8")
    pattern = re.compile(
        rf"^## \[{re.escape(tag)}\][^\n]*\n.*?(?=^## \[|\Z)",
        re.MULTILINE | re.DOTALL,
    )
    match = pattern.search(changelog)
    if match is None:
        fail(f"failed to find release notes for {tag} in {changelog_path}")
    return match.group(0).strip() + "\n"


def find_release_pr_for_commit(repo: str, commit_sha: str) -> dict[str, Any] | None:
    prs = gh_api(
        f"repos/{repo}/commits/{commit_sha}/pulls",
        headers=["Accept: application/vnd.github+json"],
    )
    matches: list[dict[str, Any]] = []
    for pr in prs:
        if pr["state"] != "closed" or pr["merged_at"] is None:
            continue
        if pr["base"]["ref"] != "main":
            continue

        head_ref = pr["head"]["ref"]
        if RELEASE_BRANCH_RE.fullmatch(head_ref) is None:
            continue

        labels = {label["name"] for label in pr.get("labels", [])}
        if "release:automated" not in labels:
            continue

        expected_title = f"chore(release): prepare {head_ref.removeprefix('release/')}"
        if pr["title"] != expected_title:
            continue

        if pr["user"]["login"] not in {"github-actions[bot]", "app/github-actions"}:
            continue

        matches.append(pr)

    if len(matches) > 1:
        fail(f"expected at most one automated release PR for {commit_sha}, found {len(matches)}")
    if not matches:
        return None
    return matches[0]


def wait_for_dispatched_publish_run(repo: str, tag: str, started_at: datetime) -> str:
    expected_title = f"Publish Release Recovery {tag}"
    run_id = ""
    for _ in range(30):
        payload = gh_api(
            f"repos/{repo}/actions/workflows/publish.yml/runs?event=workflow_dispatch&per_page=20"
        )
        candidates = []
        for run in payload["workflow_runs"]:
            created_at = datetime.fromisoformat(run["created_at"].replace("Z", "+00:00"))
            if created_at >= started_at and run.get("display_title") == expected_title:
                candidates.append(run)
        candidates.sort(key=lambda run: run["created_at"])
        if candidates:
            run_id = str(candidates[-1]["id"])
            break
        time.sleep(10)

    if not run_id:
        fail(f"failed to locate recovery publish run for {tag}")

    gh("run", "watch", run_id, "--repo", repo, "--interval", "10", "--exit-status", capture=False)
    return gh("run", "view", run_id, "--repo", repo, "--json", "url", "--jq", ".url")


def command_prepare(args: argparse.Namespace) -> int:
    git("fetch", "--force", "origin", "main", "--tags", capture=False)

    start_main_sha = git("rev-parse", "origin/main")
    current_version = cargo_version()
    latest = first_tag_line(latest_tag())
    if not latest:
        fail("no existing release tag found; bootstrap the first release manually before using this workflow")

    prepared = list_prepared_releases("origin/main")
    prepared_versions = {release.version for release in prepared}
    if current_version != latest.removeprefix("v") and current_version not in prepared_versions:
        fail(
            f"current version {current_version} is ahead of latest tag {latest} but has no automated release-prep commit on main"
        )

    latest_complete = latest
    recovery_queue: list[PreparedRelease] = []
    for release in prepared:
        state = release_state(args.repo, release)
        if state.complete:
            latest_complete = release.tag
        else:
            recovery_queue.append(release)

    append_outputs(
        {
            "current_version": current_version,
            "latest_complete_tag": latest_complete,
            "prepared_current": current_version in prepared_versions,
            "queue_count": str(len(recovery_queue)),
            "recovery_needed": bool(recovery_queue),
            "start_main_sha": start_main_sha,
        }
    )
    append_summary(
        "## Release state\n\n"
        f"- Current workspace version: `{current_version}`\n"
        f"- Latest completed tag at start: `{latest_complete}`\n"
        f"- Missing prepared releases detected: `{len(recovery_queue)}`\n"
    )

    for release in recovery_queue:
        started_at = datetime.now(timezone.utc)
        gh_api(
            f"repos/{args.repo}/actions/workflows/publish.yml/dispatches",
            method="POST",
            fields={
                "ref": "main",
                "inputs[expected_version]": release.version,
                "inputs[source_ref]": release.commit,
                "inputs[caller_run_id]": args.run_id,
            },
            expect_json=False,
        )
        run_url = wait_for_dispatched_publish_run(args.repo, release.tag, started_at)
        append_summary(
            f"## Recovered {release.tag}\n\n"
            f"- Source commit: `{release.commit}`\n"
            f"- Publish run: {run_url}\n"
        )

    if recovery_queue:
        git("fetch", "--force", "origin", "main", "--tags", capture=False)
        current_main_sha = git("rev-parse", "origin/main")
        append_outputs({"current_main_sha": current_main_sha})
        if current_main_sha != start_main_sha:
            append_outputs({"create_pr": False, "continue": False})
            append_summary(
                "## Recovery completed\n\n"
                "- Prior release recovery succeeded.\n"
                f"- `main` moved from `{start_main_sha}` to `{current_main_sha}` while recovery was running.\n"
                "- Rerun `Prepare Release PR` on the newer `main`.\n"
            )
            return 0
        append_outputs({"continue": True})

    latest = first_tag_line(latest_tag())
    if not latest:
        fail("no release tag exists after recovery")

    current_version = cargo_version()
    if current_version != latest.removeprefix("v"):
        fail(f"workspace version {current_version} does not match latest completed tag {latest} after recovery")

    commit_count = int(git("rev-list", "--count", "--no-merges", f"{latest}..origin/main"))
    if commit_count == 0:
        append_outputs({"create_pr": False})
        append_summary(
            "## No new release needed\n\n"
            f"- Latest completed release: `{latest}`\n"
            "- There are no commits on `main` after that release.\n"
        )
        return 0

    subjects_and_bodies = git("log", "--no-merges", "--format=%s%x1f%b%x1e", f"{latest}..origin/main")
    next_version = compute_next_version(current_version, subjects_and_bodies)
    tag = f"v{next_version}"
    append_outputs(
        {
            "create_pr": True,
            "latest_tag": latest,
            "version": next_version,
            "tag": tag,
            "branch": f"release/{tag}",
            "title": f"chore(release): prepare {tag}",
        }
    )
    return 0


def command_publish_context(args: argparse.Namespace) -> int:
    git("fetch", "--force", "origin", "main", "--tags", capture=False)

    if args.event_name == "push":
        release_pr = find_release_pr_for_commit(args.repo, args.commit_sha)
        if release_pr is None:
            append_outputs({"release_pr": False})
            append_summary(
                "## Publish skipped\n\n"
                f"No merged automated release PR is associated with `{args.commit_sha}`.\n"
            )
            return 0
        head_ref = str(release_pr["head"]["ref"])
        source_ref = args.commit_sha
        append_summary(
            "## Release publish candidate\n\n"
            f"- Release branch: `{head_ref}`\n"
            f"- Source commit: `{source_ref}`\n"
        )
    else:
        if not args.expected_version or not args.source_ref:
            fail("workflow_dispatch publish context requires --expected-version and --source-ref")
        resolved_head = git("rev-parse", "HEAD")
        if resolved_head != args.source_ref:
            fail(f"checked out {resolved_head}, expected {args.source_ref}")
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", args.source_ref, "origin/main"],
            check=True,
            capture_output=True,
            text=True,
            env=os.environ.copy(),
        )
        head_ref = f"release/v{args.expected_version}"
        source_ref = args.source_ref
        append_summary(
            "## Release recovery candidate\n\n"
            f"- Version: `v{args.expected_version}`\n"
            f"- Source commit: `{source_ref}`\n"
        )

    version = cargo_version()
    if args.expected_version and version != args.expected_version:
        fail(f"checked out version {version}, expected {args.expected_version}")

    tag = f"v{version}"
    expected_branch = f"release/{tag}"
    if head_ref != expected_branch:
        fail(f"release branch {head_ref} does not match crate version {version}; expected {expected_branch}")

    tag_exists, tag_target = tag_state(tag)
    if tag_exists and tag_target != source_ref:
        fail(f"tag {tag} already points at {tag_target}, expected {source_ref}")

    crate_published = crates_version_exists(version)
    release_exists = github_release_exists(args.repo, tag)
    release_complete = crate_published and tag_exists and release_exists

    append_outputs(
        {
            "release_pr": True,
            "head_ref": head_ref,
            "source_ref": source_ref,
            "tag": tag,
            "tag_exists": tag_exists,
            "tag_target": tag_target,
            "title": f"loftr {tag}",
            "version": version,
            "crate_published": crate_published,
            "github_release_exists": release_exists,
            "release_complete": release_complete,
        }
    )
    return 0


def command_extract_release_notes(args: argparse.Namespace) -> int:
    notes = parse_release_notes(args.tag, Path(args.input))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(notes, encoding="utf-8")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare", help="Recover unfinished releases and derive the next release PR metadata.")
    prepare_parser.add_argument("--repo", required=True)
    prepare_parser.add_argument("--run-id", required=True)
    prepare_parser.set_defaults(func=command_prepare)

    publish_parser = subparsers.add_parser("publish-context", help="Derive publish metadata for push and recovery workflows.")
    publish_parser.add_argument("--event-name", required=True, choices=["push", "workflow_dispatch"])
    publish_parser.add_argument("--repo", required=True)
    publish_parser.add_argument("--commit-sha", default="")
    publish_parser.add_argument("--expected-version", default="")
    publish_parser.add_argument("--source-ref", default="")
    publish_parser.set_defaults(func=command_publish_context)

    notes_parser = subparsers.add_parser("extract-release-notes", help="Extract a tagged section from CHANGELOG.md.")
    notes_parser.add_argument("--tag", required=True)
    notes_parser.add_argument("--input", required=True)
    notes_parser.add_argument("--output", required=True)
    notes_parser.set_defaults(func=command_extract_release_notes)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
