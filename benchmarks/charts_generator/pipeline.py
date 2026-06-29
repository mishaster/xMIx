#!/usr/bin/env python3
"""
Benchmark Pipeline Orchestrator

Watches a source directory for the newest subdirectory containing a summary.csv,
then runs the full pipeline: rename → copy → average → chart → move.

Usage:
    python pipeline.py --config pipeline_config.json

    # Or with CLI overrides:
    python pipeline.py \
        --source-dir /data/benchmarks/incoming \
        --work-dir /data/benchmarks/work \
        --archive-dir /data/benchmarks/archive \
        --output-xlsx all_results_llama.xlsx \
        --extra-xlsx all_results_qwen.xlsx \
        --chart-config groups.json \
        --max_num_seqs 32 --random_input_len 512 --random_output_len 128

Config file format (pipeline_config.json):
    {
        "source_dir": "/data/benchmarks/incoming",
        "work_dir": "/data/benchmarks/work",
        "archive_dir": "/data/benchmarks/archive",
        "output_xlsx": "all_results_llama.xlsx",
        "extra_xlsx": ["all_results_qwen.xlsx"],
        "chart_config": "groups.json",
        "chart_output": "benchmark_dash.html",
        "max_num_seqs": 32,
        "random_input_len": 512,
        "random_output_len": 128,
        "scripts_dir": "."
    }
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


def find_newest_subdir(source_dir):
    """Find the most recently created/modified subdirectory in source_dir."""
    source = Path(source_dir)
    if not source.is_dir():
        print(f"ERROR: Source directory does not exist: {source_dir}", file=sys.stderr)
        sys.exit(1)

    subdirs = [d for d in source.iterdir() if d.is_dir()]
    if not subdirs:
        print(f"ERROR: No subdirectories found in {source_dir}", file=sys.stderr)
        sys.exit(1)

    newest = max(subdirs, key=lambda d: d.stat().st_mtime)
    return newest


def strip_date_prefix(name):
    """Remove a leading date/timestamp pattern from a directory name.

    Handles patterns like:
        20260428_074443_2gpus_qwen_eager  →  2gpus_qwen_eager
        2026-04-28_2gpus_qwen_eager       →  2gpus_qwen_eager
        20260428-074443-2gpus_qwen_eager  →  2gpus_qwen_eager
    """
    # Pattern: YYYYMMDD[_-]HHMMSS[_-] or YYYY-MM-DD[_-]
    stripped = re.sub(r'^\d{4}-?\d{2}-?\d{2}[-_]?(\d{6}[-_]?)?', '', name)
    # Remove leading separators left over
    stripped = stripped.lstrip('_- ')
    return stripped if stripped else name


def run_command(cmd, description):
    """Run a command and exit on failure."""
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"  $ {' '.join(cmd)}")
    print(f"{'='*60}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"ERROR: {description} failed with exit code {result.returncode}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark pipeline: find → rename → average → chart → archive",
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("--config", type=str, default=None,
                        help="JSON config file (CLI args override config values)")
    parser.add_argument("--source-dir", type=str, default=None,
                        help="Directory to scan for the newest subdirectory")
    parser.add_argument("--work-dir", type=str, default=None,
                        help="Directory to copy the renamed CSV into and run scripts from")
    parser.add_argument("--archive-dir", type=str, default=None,
                        help="Directory to move the processed subdirectory into")
    parser.add_argument("--output-xlsx", type=str, default=None,
                        help="Output Excel filename for average_csv.py (e.g. all_results_llama.xlsx)")
    parser.add_argument("--extra-xlsx", type=str, action="append", default=None,
                        help="Additional Excel file(s) to pass to excel_to_charts.py (repeatable)")
    parser.add_argument("--chart-config", type=str, default=None,
                        help="JSON config for excel_to_charts.py (groups.json)")
    parser.add_argument("--chart-output", type=str, default=None,
                        help="Output HTML filename for the dashboard")
    parser.add_argument("--max_num_seqs", type=int, default=None)
    parser.add_argument("--random_input_len", type=int, default=None)
    parser.add_argument("--random_output_len", type=int, default=None)
    parser.add_argument("--scripts-dir", type=str, default=None,
                        help="Directory where average_csv.py and excel_to_charts.py live")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be done without executing")
    args = parser.parse_args()

    # Load config file as defaults
    cfg = {}
    if args.config:
        with open(args.config) as f:
            cfg = json.load(f)

    # Merge: CLI args override config file
    source_dir = args.source_dir or cfg.get("source_dir")
    work_dir = args.work_dir or cfg.get("work_dir")
    archive_dir = args.archive_dir or cfg.get("archive_dir")
    output_xlsx = args.output_xlsx or cfg.get("output_xlsx")
    extra_xlsx = args.extra_xlsx or cfg.get("extra_xlsx", [])
    chart_config = args.chart_config or cfg.get("chart_config")
    chart_output = args.chart_output or cfg.get("chart_output")
    max_num_seqs = args.max_num_seqs or cfg.get("max_num_seqs", 32)
    random_input_len = args.random_input_len or cfg.get("random_input_len", 512)
    random_output_len = args.random_output_len or cfg.get("random_output_len", 128)
    scripts_dir = args.scripts_dir or cfg.get("scripts_dir", ".")

    # Validate required fields
    missing = []
    if not source_dir:   missing.append("source_dir")
    if not work_dir:     missing.append("work_dir")
    if not archive_dir:  missing.append("archive_dir")
    if not output_xlsx:  missing.append("output_xlsx")
    if missing:
        print(f"ERROR: Missing required config: {', '.join(missing)}", file=sys.stderr)
        print("Provide via --config file or CLI arguments.", file=sys.stderr)
        sys.exit(1)

    # Resolve paths
    scripts_dir = Path(scripts_dir).resolve()
    work_dir = Path(work_dir).resolve()
    archive_dir = Path(archive_dir).resolve()

    # Step 1: Find newest subdirectory
    print(f"Scanning {source_dir} for newest subdirectory...")
    newest_dir = find_newest_subdir(source_dir)
    dir_name = newest_dir.name
    print(f"Found: {newest_dir}")

    # Step 2: Check for summary.csv
    summary_csv = newest_dir / "summary.csv"
    if not summary_csv.exists():
        print(f"ERROR: No summary.csv found in {newest_dir}", file=sys.stderr)
        sys.exit(1)

    # Step 3: Rename summary.csv → summary_<dirname>.csv
    renamed_csv = newest_dir / f"summary_{dir_name}.csv"
    print(f"Renaming: summary.csv → summary_{dir_name}.csv")
    if not args.dry_run:
        summary_csv.rename(renamed_csv)

    # Step 4: Copy renamed CSV to work directory
    work_dir.mkdir(parents=True, exist_ok=True)
    dest_csv = work_dir / renamed_csv.name
    print(f"Copying to: {dest_csv}")
    if not args.dry_run:
        shutil.copy2(renamed_csv, dest_csv)

    # Step 5: Run average_csv.py
    label = strip_date_prefix(dir_name).replace('_', ' ')
    output_xlsx_path = work_dir / output_xlsx

    avg_cmd = [
        "python3", str(scripts_dir / "average_csv.py"),
        "--input", f"{dest_csv}:{label}",
        "-o", str(output_xlsx_path),
    ]

    if args.dry_run:
        print(f"\n[DRY RUN] Would run: {' '.join(avg_cmd)}")
    else:
        run_command(avg_cmd, "Running average_csv.py")

    # Step 6: Run excel_to_charts.py
    # Build the list of Excel files: output_xlsx + any extra xlsx files
    xlsx_files = [str(output_xlsx_path)]
    for extra in (extra_xlsx or []):
        extra_path = Path(extra)
        if not extra_path.is_absolute():
            extra_path = work_dir / extra_path
        xlsx_files.append(str(extra_path))

    # Determine dashboard output name
    if chart_output:
        dash_output = str(work_dir / chart_output)
    else:
        dash_output = str(output_xlsx_path.with_suffix('').with_name(
            output_xlsx_path.stem + "_dash.html"))

    chart_cmd = [
        "python3", str(scripts_dir / "excel_to_charts.py"),
        *xlsx_files,
        "--max_num_seqs", str(max_num_seqs),
        "--random_input_len", str(random_input_len),
        "--random_output_len", str(random_output_len),
    ]
    if chart_config:
        chart_config_path = Path(chart_config)
        if not chart_config_path.is_absolute():
            chart_config_path = work_dir / chart_config_path
        chart_cmd.extend(["--config", str(chart_config_path)])
    chart_cmd.extend(["-o", dash_output])

    if args.dry_run:
        print(f"\n[DRY RUN] Would run: {' '.join(chart_cmd)}")
    else:
        run_command(chart_cmd, "Running excel_to_charts.py")

    # Step 7: Move the processed subdirectory to archive
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive_dest = archive_dir / dir_name
    print(f"\nArchiving: {newest_dir} → {archive_dest}")
    if not args.dry_run:
        shutil.move(str(newest_dir), str(archive_dest))

    print(f"\nDone! Dashboard: {dash_output}")


if __name__ == "__main__":
    main()
