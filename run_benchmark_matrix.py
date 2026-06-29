#!/usr/bin/env python3
"""Run a matrix of benchmarks across git branches and commits.

For each entry in the config, the script checks out the given branch/commit,
runs each benchmark command in order, finds the newest subdirectory in the
configured output directory, and moves it into the archive directory under a
chosen name. Failures are logged and skipped; a summary is printed at the end.

YAML config format:

    output_dir: benchmarks/charts_generator
    archive_dir: benchmarks/backup_results
    runs:
      - branch: llama_app_2
        benchmarks:
          - cmd: ./benchmarks/run_llama.sh --bs 8
            rename_to: llama_bs8_branch
          - cmd: ./benchmarks/run_llama.sh --bs 16
            rename_to: llama_bs16_branch
      - commit: 6d53606f3
        benchmarks:
          - cmd: ./benchmarks/run_llama.sh --bs 8
            rename_to: llama_bs8_pre_disable

Usage:

    python benchmarks/run_benchmark_matrix.py path/to/matrix.yaml
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import yaml


def git(*args: str, capture: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], capture_output=capture, text=True)


def working_tree_has_tracked_changes() -> bool:
    # Ignore untracked files: `git checkout` doesn't touch them, so they're
    # safe across branch switches. Only tracked modifications (M/A/D/R/...)
    # could be clobbered by a checkout.
    result = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        check=True,
        capture_output=True,
        text=True,
    )
    return bool(result.stdout.strip())


def current_branch() -> str:
    # Returns "HEAD" if detached; the user shouldn't be starting from a
    # detached HEAD anyway, and we surface that in the summary.
    result = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def newest_subdir(parent: Path) -> Path | None:
    subdirs = [p for p in parent.iterdir() if p.is_dir()]
    if not subdirs:
        return None
    return max(subdirs, key=lambda p: p.stat().st_mtime)


def run_benchmark(cmd: str) -> int:
    print(f"  $ {cmd}", flush=True)
    return subprocess.run(cmd, shell=True).returncode


TORCH_COMPILE_CACHE = Path("~/.cache/vllm/torch_compile_cache").expanduser()


def clear_torch_compile_cache() -> None:
    if TORCH_COMPILE_CACHE.exists():
        shutil.rmtree(TORCH_COMPILE_CACHE)
        print(f"  cleared {TORCH_COMPILE_CACHE}", flush=True)


def parse_ref(entry: dict) -> tuple[str, str]:
    if "branch" in entry and "commit" in entry:
        raise ValueError(f"entry has both 'branch' and 'commit': {entry}")
    if "branch" in entry:
        return "branch", entry["branch"]
    if "commit" in entry:
        return "commit", entry["commit"]
    raise ValueError(f"entry missing 'branch' or 'commit': {entry}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config", type=Path, help="YAML config path")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    output_dir = Path(cfg["output_dir"])
    archive_dir = Path(cfg["archive_dir"])
    runs = cfg["runs"]

    if not output_dir.is_dir():
        sys.exit(f"output_dir does not exist: {output_dir}")
    archive_dir.mkdir(parents=True, exist_ok=True)

    if working_tree_has_tracked_changes():
        sys.exit(
            "Working tree has tracked modifications that a checkout would clobber.\n"
            "Commit or stash them first (untracked files are OK and ignored)."
        )

    starting_branch = current_branch()
    if starting_branch == "HEAD":
        sys.exit("Refusing to start from a detached HEAD; check out a branch first.")
    print(f"Starting branch: {starting_branch}")

    successes: list[tuple[str, str, str]] = []
    failures: list[tuple[str, str, str]] = []

    try:
        for entry in runs:
            kind, ref = parse_ref(entry)
            print(f"\n=== {kind}: {ref} ===", flush=True)

            checkout = git("checkout", ref)
            if checkout.returncode != 0:
                failures.append((ref, "<checkout>", "git checkout failed"))
                continue

            for bench in entry.get("benchmarks", []):
                cmd = bench["cmd"]
                rename_to = bench["rename_to"]

                target = archive_dir / rename_to
                if target.exists():
                    failures.append((ref, cmd, f"target already exists: {target}"))
                    continue

                rc = run_benchmark(cmd)
                clear_torch_compile_cache()
                if rc != 0:
                    failures.append((ref, cmd, f"exit {rc}"))
                    continue

                newest = newest_subdir(output_dir)
                if newest is None:
                    failures.append((ref, cmd, f"no subdirectory found in {output_dir}"))
                    continue

                shutil.move(str(newest), str(target))
                successes.append((ref, cmd, str(target)))
                print(f"  archived: {newest.name} -> {target}", flush=True)
    finally:
        print(f"\nRestoring branch: {starting_branch}", flush=True)
        git("checkout", starting_branch)

    print("\n=== Summary ===")
    print(f"Succeeded: {len(successes)}")
    for ref, cmd, target in successes:
        print(f"  [{ref}] {cmd} -> {target}")
    print(f"Failed: {len(failures)}")
    for ref, cmd, why in failures:
        print(f"  [{ref}] {cmd}: {why}")

    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
