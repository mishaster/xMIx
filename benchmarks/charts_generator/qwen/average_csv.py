#!/usr/bin/env python3
"""
CSV Benchmark Averaging Script

Replicates the Office Script (.osts) logic in Python:
  - Reads a raw benchmark CSV with multiple runs per parameter combo
  - Groups rows by (max_num_seqs, random_input_len, random_output_len)
  - Averages all numeric columns within each group
  - Carries forward the config params from the group
  - Outputs a summary Excel file with one row per group

Usage:
    python average_csv.py summary_2gpus_qwen_eager.csv -o averaged_qwen_eager.xlsx
    python average_csv.py file1.csv file2.csv file3.csv -o all_results.xlsx

    # With a custom label (mode name) for the first column:
    python average_csv.py summary_2gpus_qwen_eager.csv --label "2 gpus qwen eager" -o out.xlsx

    # Merge multiple CSVs into one Excel, each with its own label:
    python average_csv.py \
        --input "summary_qwen_eager.csv:2 gpus qwen eager" \
        --input "summary_qwen_vanilla.csv:2 gpus qwen vanilla" \
        -o all_results_qwen.xlsx
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd


# Columns that define each parameter group
GROUP_COLS = ["max_num_seqs", "random_input_len", "random_output_len"]

# Columns to carry forward (not averaged) — taken from the first row of each group
CARRY_COLS = ["max_num_seqs", "max_num_batched_tokens", "random_input_len", "random_output_len"]

# Columns to drop from the output (not useful in summary)
DROP_COLS = ["Unnamed: 0", "date", "endpoint_type", "backend", "label", "model_id",
             "tokenizer_id", "num_prompts", "request_rate", "burstiness",
             "max_concurrency", "run_number", "completed", "failed"]


def derive_label(filepath):
    """Derive a human-readable label from the filename."""
    name = Path(filepath).stem
    # Remove 'summary_' prefix and timestamp pattern
    name = re.sub(r'^summary_', '', name)
    name = re.sub(r'\d{8}_\d{6}_', '', name)
    # Replace underscores with spaces
    return name.replace('_', ' ')


def process_csv(filepath, label=None):
    """Read a CSV, group by params, average numeric columns, return summary DataFrame."""
    df = pd.read_csv(filepath)

    # Derive label if not provided
    if not label:
        label = derive_label(filepath)

    # Identify numeric columns to average (exclude group/carry/drop cols)
    skip = set(GROUP_COLS + CARRY_COLS + DROP_COLS)
    numeric_cols = [c for c in df.columns
                    if c not in skip and pd.api.types.is_numeric_dtype(df[c])]

    rows = []
    for group_vals, group_df in df.groupby(GROUP_COLS, sort=False):
        row = {"label": label}

        # Average the numeric columns
        for col in numeric_cols:
            row[col] = group_df[col].mean()

        # Carry forward config params
        first = group_df.iloc[0]
        for col in CARRY_COLS:
            if col in group_df.columns:
                row[col] = first[col]

        rows.append(row)

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Average benchmark CSV runs and produce a summary Excel file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single file, auto-detect label from filename:
  python average_csv.py summary_2gpus_qwen_eager.csv -o averaged.xlsx

  # Multiple files merged into one Excel:
  python average_csv.py file1.csv file2.csv file3.csv -o all_results.xlsx

  # Explicit labels per file:
  python average_csv.py \\
      --input "summary_eager.csv:2 gpus qwen eager" \\
      --input "summary_vanilla.csv:2 gpus qwen vanilla" \\
      -o all_results_qwen.xlsx
        """)

    parser.add_argument("files", nargs="*", help="CSV files (label auto-derived from filename)")
    parser.add_argument("--input", action="append", dest="labeled_inputs", metavar="FILE:LABEL",
                        help="CSV file with explicit label, format: 'path/to/file.csv:my label'")
    parser.add_argument("-o", "--output", default="averaged_results.xlsx",
                        help="Output Excel file (default: averaged_results.xlsx)")
    args = parser.parse_args()

    # Collect all file/label pairs
    file_label_pairs = []

    for f in (args.files or []):
        file_label_pairs.append((f, None))

    for entry in (args.labeled_inputs or []):
        if ":" in entry:
            path, label = entry.split(":", 1)
            file_label_pairs.append((path.strip(), label.strip()))
        else:
            file_label_pairs.append((entry.strip(), None))

    if not file_label_pairs:
        parser.error("No input files provided. Pass CSV files as arguments or use --input.")

    # Process all files
    all_dfs = []
    for filepath, label in file_label_pairs:
        resolved_label = label or derive_label(filepath)
        print(f"Processing: {filepath} → label='{resolved_label}'")
        summary = process_csv(filepath, label)
        all_dfs.append(summary)

    # Combine and write output
    combined = pd.concat(all_dfs, ignore_index=True)

    # Reorder columns: label first, then carry cols at end
    metric_cols = [c for c in combined.columns if c not in ["label"] + CARRY_COLS]
    col_order = ["label"] + metric_cols + [c for c in CARRY_COLS if c in combined.columns]
    combined = combined[col_order]

    output_path = Path(args.output)
    combined.to_excel(output_path, index=False)
    print(f"\nSummary written to {output_path}")
    print(f"  {len(combined)} rows ({len(file_label_pairs)} file(s), "
          f"{len(combined) // len(file_label_pairs)} param combos each)")


if __name__ == "__main__":
    main()
