#!/usr/bin/env python3
"""Compare vllm bench outputs across git refs.

For each ref in the config, the script:
  1. `git checkout`s the ref
  2. starts `vllm serve <model> <serve_args>` in the background
  3. polls GET /health until ready (or times out)
  4. runs `vllm bench serve` `warmup_runs` times against that server,
     discarding the results (cold kernels / torch-compile cache can make
     the first measured run diverge from later ones)
  5. runs `vllm bench serve` `num_runs_per_branch` times against that
     server, saving each run's results JSON and a text-only extract
     (the jq -r '.generated_texts[]' equivalent, done in Python so we
     don't depend on jq being installed)
  6. tears the server down

After all refs have run, it compares the text-only files:
  - within each ref: every pair of runs must be byte-identical
  - across refs at the same run index: run #i of ref A vs run #i of ref B

A failed expected-match is reported with the count of differing prompts and
a snippet of the first few divergences.

YAML config format: see output_comparison_matrix.yaml beside this file.

Usage:
    python run_output_comparison.py output_comparison_matrix.yaml
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from itertools import combinations
from pathlib import Path

import yaml


# ----------------------------------------------------------------------------
# Git helpers (mirrors run_benchmark_matrix.py style)
# ----------------------------------------------------------------------------

def git(*args: str, capture: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], capture_output=capture, text=True)


def working_tree_has_tracked_changes() -> bool:
    # Ignore untracked files: `git checkout` doesn't touch them.
    result = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        check=True, capture_output=True, text=True,
    )
    return bool(result.stdout.strip())


def current_branch() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        check=True, capture_output=True, text=True,
    )
    return result.stdout.strip()


def parse_ref(entry: dict) -> tuple[str, str]:
    if "branch" in entry and "commit" in entry:
        raise ValueError(f"entry has both 'branch' and 'commit': {entry}")
    if "branch" in entry:
        return "branch", entry["branch"]
    if "commit" in entry:
        return "commit", entry["commit"]
    raise ValueError(f"entry missing 'branch' or 'commit': {entry}")


# ----------------------------------------------------------------------------
# Torch compile cache (kept between branches, same as the existing script)
# ----------------------------------------------------------------------------

TORCH_COMPILE_CACHE = Path("~/.cache/vllm/torch_compile_cache").expanduser()


def clear_torch_compile_cache() -> None:
    if TORCH_COMPILE_CACHE.exists():
        shutil.rmtree(TORCH_COMPILE_CACHE)
        print(f"  cleared {TORCH_COMPILE_CACHE}", flush=True)


# ----------------------------------------------------------------------------
# Server lifecycle
# ----------------------------------------------------------------------------

def port_is_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) != 0


def wait_for_port_free(port: int, timeout_sec: int) -> bool:
    """Poll until nothing is listening on `port`.

    The parent vllm process can exit before its tensor-parallel worker
    children release the listening socket, so `proc.wait()` returning is
    not enough to know the port is reusable.
    """
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        if port_is_free(port):
            return True
        time.sleep(1)
    return False


def wait_for_health(port: int, timeout_sec: int) -> bool:
    url = f"http://127.0.0.1:{port}/health"
    deadline = time.monotonic() + timeout_sec
    last_err = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as e:
            last_err = e
        time.sleep(2)
    print(f"  /health did not return 200 within {timeout_sec}s (last error: {last_err})",
          flush=True)
    return False


class VllmServer:
    """Context manager that owns one `vllm serve` process."""

    def __init__(self, model: str, serve_args: str, port: int, env: dict,
                 health_timeout_sec: int, shutdown_timeout_sec: int,
                 port_release_timeout_sec: int,
                 log_path: Path):
        self.model = model
        self.serve_args = serve_args
        self.port = port
        self.env = env
        self.health_timeout_sec = health_timeout_sec
        self.shutdown_timeout_sec = shutdown_timeout_sec
        self.port_release_timeout_sec = port_release_timeout_sec
        self.log_path = log_path
        self.proc: subprocess.Popen | None = None
        self.log_fh = None

    def __enter__(self) -> "VllmServer":
        if not port_is_free(self.port):
            raise RuntimeError(
                f"port {self.port} is already in use; refusing to start vllm serve"
            )
        cmd = f"vllm serve {self.model} {self.serve_args} --port {self.port}"
        print(f"  $ {cmd}", flush=True)
        print(f"  server log -> {self.log_path}", flush=True)
        self.log_fh = open(self.log_path, "w")
        # start_new_session=True so we can signal the whole process group on
        # shutdown (vllm spawns children for tensor-parallel workers).
        self.proc = subprocess.Popen(
            cmd, shell=True, env=self.env,
            stdout=self.log_fh, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        if not wait_for_health(self.port, self.health_timeout_sec):
            self._terminate()
            raise RuntimeError("vllm server failed to become healthy")
        print(f"  server healthy on port {self.port}", flush=True)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._terminate()

    def _terminate(self) -> None:
        if self.proc is None:
            return
        if self.proc.poll() is None:
            print("  stopping vllm server...", flush=True)
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                self.proc.wait(timeout=self.shutdown_timeout_sec)
            except subprocess.TimeoutExpired:
                print("  SIGTERM grace expired, sending SIGKILL", flush=True)
                try:
                    os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
                self.proc.wait()
        if self.log_fh is not None:
            self.log_fh.close()
            self.log_fh = None
        self.proc = None

        # The parent exiting doesn't mean the listening socket is free:
        # tensor-parallel worker processes can outlive the parent briefly.
        # Wait until the port is actually reusable before returning, so the
        # next branch's server doesn't fail with "address already in use".
        print(f"  waiting for port {self.port} to be released "
              f"(up to {self.port_release_timeout_sec}s)...", flush=True)
        if wait_for_port_free(self.port, self.port_release_timeout_sec):
            print(f"  port {self.port} released", flush=True)
        else:
            # Last resort: try to kill anything still holding the port.
            print(f"  port {self.port} still in use after "
                  f"{self.port_release_timeout_sec}s; attempting fuser -k",
                  flush=True)
            subprocess.run(
                ["fuser", "-k", f"{self.port}/tcp"],
                capture_output=True, check=False,
            )
            # Give the OS a moment to reclaim the socket.
            if wait_for_port_free(self.port, 10):
                print(f"  port {self.port} released after fuser -k", flush=True)
            else:
                print(f"  WARNING: port {self.port} STILL in use; the next "
                      f"server start will likely fail", flush=True)


# ----------------------------------------------------------------------------
# Bench invocation + text extraction
# ----------------------------------------------------------------------------

def build_bench_cmd(model: str, port: int, bench_cfg: dict, results_json: Path) -> str:
    parts = [
        "vllm bench serve",
        f"--backend vllm",
        f"--model {model}",
        f"--base-url http://127.0.0.1:{port}",
        f"--endpoint /v1/completions",
        f"--dataset-name {bench_cfg['dataset_name']}",
        f"--dataset-path {bench_cfg['dataset_path']}",
        f"--sharegpt-output-len {bench_cfg['sharegpt_output_len']}",
        f"--num-prompts {bench_cfg['num_prompts']}",
        f"--seed {bench_cfg['seed']}",
        f"--temperature {bench_cfg['temperature']}",
        "--save-detailed",
        "--save-result",
        f"--result-filename {results_json}",
    ]
    return " ".join(parts)


def extract_text_only(results_json: Path, text_only: Path) -> bool:
    """Write a human-readable text-only file: one generation per line.

    Equivalent of: jq -r '.generated_texts[]' results.json > text_only.txt

    This file is for human inspection / piping into `diff` manually. The
    script's own comparison reads from results.json instead, because any
    generation containing '\\r', '\\u2028', or similar would be split by
    splitlines() on read-back and inflate the difference count.
    """
    texts = load_generations(results_json)
    if texts is None:
        return False
    text_only.write_text("\n".join(texts) + "\n")
    return True


# ----------------------------------------------------------------------------
# Comparison
# ----------------------------------------------------------------------------

def load_generations(results_json: Path) -> list[str] | None:
    """Read .generated_texts from a results JSON. Returns None on failure."""
    try:
        data = json.loads(results_json.read_text())
    except (OSError, json.JSONDecodeError) as e:
        print(f"  failed to read {results_json}: {e}", flush=True)
        return None
    texts = data.get("generated_texts")
    if not isinstance(texts, list):
        print(f"  {results_json} has no .generated_texts array", flush=True)
        return None
    return texts


def compare_generations(a_json: Path, b_json: Path,
                        max_snippets: int) -> tuple[bool, str]:
    """Compare the .generated_texts arrays of two results JSONs.

    Returns (equal, report). The report is empty when equal.

    Comparing JSON arrays directly (not the text_only.txt extracts) is
    important: a generated text can contain '\\r', '\\u2028', or other
    line-terminator characters that splitlines() would treat as record
    boundaries, inflating the apparent difference count.
    """
    a_texts = load_generations(a_json)
    b_texts = load_generations(b_json)
    if a_texts is None or b_texts is None:
        return False, "  (could not load one or both results files)"

    if a_texts == b_texts:
        return True, ""

    a_label = f"{a_json.parent.name}/{a_json.name}"
    b_label = f"{b_json.parent.name}/{b_json.name}"

    lines: list[str] = []
    if len(a_texts) != len(b_texts):
        lines.append(
            f"  different number of generations: "
            f"{a_label}={len(a_texts)} vs {b_label}={len(b_texts)}"
        )

    n = min(len(a_texts), len(b_texts))
    diff_indices = [i for i in range(n) if a_texts[i] != b_texts[i]]
    total_diff = len(diff_indices) + abs(len(a_texts) - len(b_texts))
    lines.append(
        f"  {total_diff} differing generation(s) "
        f"(out of {max(len(a_texts), len(b_texts))})"
    )

    for idx in diff_indices[:max_snippets]:
        lines.append(f"  --- prompt #{idx}: {a_label} vs {b_label} ---")
        # repr() makes embedded \r, \n, unicode line separators visible,
        # which is the whole point of comparing JSON directly. Split into
        # one entry per line so unified_diff is meaningful even for
        # multi-line generations.
        a_lines = a_texts[idx].splitlines(keepends=True) or [""]
        b_lines = b_texts[idx].splitlines(keepends=True) or [""]
        diff = difflib.unified_diff(
            a_lines, b_lines,
            fromfile=a_label, tofile=b_label, lineterm="", n=1,
        )
        for d in diff:
            lines.append(f"    {d.rstrip(chr(10))}")

    if len(diff_indices) > max_snippets:
        lines.append(f"  ... and {len(diff_indices) - max_snippets} more")

    return False, "\n".join(lines)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config", type=Path, help="YAML config path")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    archive_dir = Path(cfg["archive_dir"])
    num_runs = int(cfg.get("num_runs_per_branch", 3))
    warmup_runs = int(cfg.get("warmup_runs", 0))
    server_cfg = cfg.get("server", {})
    port = int(server_cfg.get("port", 8000))
    health_timeout = int(server_cfg.get("health_timeout_sec", 600))
    shutdown_timeout = int(server_cfg.get("shutdown_timeout_sec", 60))
    port_release_timeout = int(server_cfg.get("port_release_timeout_sec", 120))
    env_overrides = {k: str(v) for k, v in cfg.get("env", {}).items()}
    bench_defaults = cfg["bench_defaults"]
    max_snippets = int(cfg.get("divergence_snippets", 3))
    runs = cfg["runs"]

    if len(runs) < 2:
        sys.exit("need at least 2 entries under `runs:` to compare anything")

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

    full_env = os.environ.copy()
    full_env.update(env_overrides)

    # label -> [Path, Path, ...] of results.json files in run order.
    # We track the JSON files (not text_only.txt) because comparison reads
    # .generated_texts directly to avoid splitlines() ambiguity on any
    # exotic line-terminator chars inside a generation.
    results_files: dict[str, list[Path]] = {}
    # (label, run_idx, what) -> reason
    failures: list[tuple[str, int, str, str]] = []

    try:
        for entry in runs:
            label = entry["label"]
            kind, ref = parse_ref(entry)
            model = entry["model"]
            serve_args = entry.get("serve_args", "")
            bench_overrides = entry.get("bench_overrides", {})
            bench_cfg = {**bench_defaults, **bench_overrides}

            print(f"\n=== label={label}  {kind}={ref} ===", flush=True)

            ref_dir = archive_dir / label
            if ref_dir.exists():
                failures.append((label, -1, "<setup>",
                                 f"target already exists: {ref_dir}"))
                continue
            ref_dir.mkdir(parents=True)

            checkout = git("checkout", ref)
            if checkout.returncode != 0:
                failures.append((label, -1, "<checkout>",
                                 f"git checkout {ref} failed"))
                continue

            results_files[label] = []
            server_log = ref_dir / "vllm_serve.log"
            try:
                with VllmServer(
                    model=model, serve_args=serve_args, port=port, env=full_env,
                    health_timeout_sec=health_timeout,
                    shutdown_timeout_sec=shutdown_timeout,
                    port_release_timeout_sec=port_release_timeout,
                    log_path=server_log,
                ):
                    # Warmup runs: discarded, not compared. The first
                    # bench call after a fresh `vllm serve` may take a
                    # different code path (cold kernels, empty torch
                    # compile cache, autotune decisions) than subsequent
                    # calls, which can cause run #1 to diverge from runs
                    # #2..N even with VLLM_BATCH_INVARIANT=1.
                    for warm_idx in range(1, warmup_runs + 1):
                        warm_dir = ref_dir / "warmup" / f"run_{warm_idx}"
                        warm_dir.mkdir(parents=True)
                        warm_json = warm_dir / "results.json"
                        cmd = build_bench_cmd(model, port, bench_cfg, warm_json)
                        print(f"  [warmup {warm_idx}/{warmup_runs}] $ {cmd}",
                              flush=True)
                        rc = subprocess.run(
                            cmd, shell=True, env=full_env
                        ).returncode
                        if rc != 0:
                            # Don't abort — warmup is best-effort. Just log it.
                            print(f"  [warmup {warm_idx}/{warmup_runs}] "
                                  f"exit {rc} (continuing)", flush=True)

                    for run_idx in range(1, num_runs + 1):
                        run_dir = ref_dir / f"run_{run_idx}"
                        run_dir.mkdir()
                        results_json = run_dir / "results.json"
                        text_only = run_dir / "text_only.txt"

                        cmd = build_bench_cmd(model, port, bench_cfg, results_json)
                        print(f"  [run {run_idx}/{num_runs}] $ {cmd}", flush=True)
                        rc = subprocess.run(cmd, shell=True, env=full_env).returncode
                        if rc != 0:
                            failures.append((label, run_idx, "<bench>",
                                             f"exit {rc}"))
                            continue

                        if not results_json.exists():
                            failures.append((label, run_idx, "<bench>",
                                             f"missing {results_json}"))
                            continue

                        if not extract_text_only(results_json, text_only):
                            failures.append((label, run_idx, "<extract>",
                                             "text extraction failed"))
                            continue

                        results_files[label].append(results_json)
                        print(f"  [run {run_idx}/{num_runs}] -> {results_json}",
                              flush=True)
            except RuntimeError as e:
                failures.append((label, -1, "<server>", str(e)))

            clear_torch_compile_cache()
    finally:
        print(f"\nRestoring branch: {starting_branch}", flush=True)
        git("checkout", starting_branch)

    # ------------------------------------------------------------------
    # Comparison phase
    # ------------------------------------------------------------------

    print("\n=== Comparison ===")
    comparison_report_path = archive_dir / "comparison_report.md"
    report_lines: list[str] = ["# vllm output comparison\n"]

    mismatches = 0
    matches = 0

    # Within each branch: all pairs of runs.
    for label, files in results_files.items():
        report_lines.append(f"## Within-branch: {label}\n")
        if len(files) < 2:
            report_lines.append(f"- only {len(files)} successful run(s); nothing to compare\n")
            continue
        for i, j in combinations(range(len(files)), 2):
            equal, detail = compare_generations(files[i], files[j], max_snippets)
            tag = "OK" if equal else "MISMATCH"
            line = f"- {tag}: run {i + 1} vs run {j + 1}"
            print(f"  [{label}] {line[2:]}", flush=True)
            report_lines.append(line)
            if equal:
                matches += 1
            else:
                mismatches += 1
                report_lines.append("```")
                report_lines.append(detail)
                report_lines.append("```")
        report_lines.append("")

    # Across branches: only compare runs whose `group` matches. Group
    # defaults to the model string, so by default we never compare
    # different models against each other. An explicit `group:` field on
    # a run entry overrides this (useful for e.g. comparing fp16 vs fp8
    # of the same model within the same group).
    label_to_group: dict[str, str] = {}
    for r in runs:
        if r["label"] not in results_files:
            continue
        label_to_group[r["label"]] = r.get("group", r["model"])

    # group -> ordered list of labels in that group (preserves config order).
    groups: dict[str, list[str]] = {}
    for label, group in label_to_group.items():
        groups.setdefault(group, []).append(label)

    for group, group_labels in groups.items():
        if len(group_labels) < 2:
            # Only one branch ran for this group; nothing to cross-compare.
            continue
        for la, lb in combinations(group_labels, 2):
            report_lines.append(
                f"## Across-branch ({group}): {la} vs {lb} (same run index)\n"
            )
            fa, fb = results_files[la], results_files[lb]
            n = min(len(fa), len(fb))
            if n == 0:
                report_lines.append("- no overlapping runs to compare\n")
                continue
            if len(fa) != len(fb):
                report_lines.append(
                    f"- note: {la} has {len(fa)} run(s), {lb} has {len(fb)}; "
                    f"comparing the first {n}\n"
                )
            for i in range(n):
                equal, detail = compare_generations(fa[i], fb[i], max_snippets)
                tag = "OK" if equal else "MISMATCH"
                line = f"- {tag}: run {i + 1}"
                print(f"  [{la} vs {lb}] {line[2:]}", flush=True)
                report_lines.append(line)
                if equal:
                    matches += 1
                else:
                    mismatches += 1
                    report_lines.append("```")
                    report_lines.append(detail)
                    report_lines.append("```")
            report_lines.append("")

    comparison_report_path.write_text("\n".join(report_lines))

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------

    print("\n=== Summary ===")
    print(f"Refs processed: {len(results_files)} / {len(runs)}")
    for label, files in results_files.items():
        print(f"  {label}: {len(files)} / {num_runs} runs succeeded")
    print(f"Comparisons: {matches} OK, {mismatches} MISMATCH")
    print(f"Report: {comparison_report_path}")
    if failures:
        print(f"Failures: {len(failures)}")
        for label, run_idx, what, why in failures:
            run_tag = f"run {run_idx}" if run_idx > 0 else "setup"
            print(f"  [{label} / {run_tag}] {what}: {why}")

    # Nonzero exit if anything failed to run OR any expected match diverged.
    return 0 if not failures and mismatches == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
