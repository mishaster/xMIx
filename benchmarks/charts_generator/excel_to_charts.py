#!/usr/bin/env python3
"""
Excel Benchmark Results → Interactive HTML Bar Chart Dashboard

Converts Excel files (one per model) into an HTML page with bar charts
comparing modes across total_token_throughput, mean_itl_ms, and p99_itl_ms.

Supports configurable CHART GROUPS via a JSON config file, so you can
control which modes appear in which charts. Modes from DIFFERENT Excel
files can be mixed into the same chart group.

Usage:
    # Default: all modes from all files in one group
    python excel_to_charts.py All_results_llama.xlsx All_results_qwen.xlsx \
        --max_num_seqs 64 --random_input_len 512 --random_output_len 128

    # With custom groups
    python excel_to_charts.py All_results_llama.xlsx All_results_qwen.xlsx \
        --max_num_seqs 64 --random_input_len 512 --random_output_len 128 \
        --config groups.json -o dashboard.html

Config file format (groups.json):
    {
        "baseline": "vanilla",
        "groups": [
            {
                "name": "Baseline vs Eager",
                "match": ["vanilla", "eager"]
            },
            {
                "name": "All Modes",
                "match": ["*"]
            },
            {
                "name": "Llama Only",
                "match": ["*"],
                "models": ["llama"]
            },
            {
                "name": "Cross-Model Vanilla",
                "match": ["vanilla"],
                "models": ["*"]
            }
        ]
    }

"baseline" (optional) — substring to identify the baseline/reference mode.
  Each bar shows the % difference vs the baseline. The baseline bar itself
  shows "baseline". Omit to disable percentage badges.

Each group's "match" list contains substrings — a mode is included if ANY
substring matches (case-insensitive). Use ["*"] to include all modes.

Each group's optional "models" list filters by model name (from filename).
  Use ["*"] or omit to include all models. Substring matching, case-insensitive.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


DEFAULT_CONFIG = {
    "baseline": None,
    "groups": [
        {"name": "All Modes", "match": ["*"]}
    ]
}


def parse_excel(filepath, max_num_seqs, random_input_len, random_output_len):
    """Parse an Excel file and extract benchmark data for the given params."""
    df = pd.read_excel(filepath)

    mode_col = df.columns[0]
    df[mode_col] = df[mode_col].ffill()

    mask = (
        (df["max_num_seqs"] == max_num_seqs)
        & (df["random_input_len"] == random_input_len)
        & (df["random_output_len"] == random_output_len)
    )
    filtered = df[mask].copy()

    if filtered.empty:
        print(f"WARNING: No data in {filepath} for max_num_seqs={max_num_seqs}, "
              f"input_len={random_input_len}, output_len={random_output_len}")
        return None

    model_name = Path(filepath).stem.replace("All_results_", "")

    records = []
    for _, row in filtered.iterrows():
        mode = str(row[mode_col]).strip()
        records.append({
            "mode": mode,
            "total_token_throughput": round(row["total_token_throughput"], 2),
            "mean_itl_ms": round(row["mean_itl_ms"], 3),
            "p99_itl_ms": round(row["p99_itl_ms"], 3),
        })

    return {"model": model_name, "records": records}


def filter_records_by_group(records, match_patterns, model_patterns=None):
    """Filter records by mode match patterns and optional model patterns.

    Each record must have 'mode' and 'model' keys.
    match_patterns: substrings matched against mode (case-insensitive). ["*"] = all.
    model_patterns: substrings matched against model (case-insensitive). ["*"] or None = all.
    """
    # Filter by model first
    if model_patterns and "*" not in model_patterns:
        filtered = []
        for rec in records:
            model_lower = rec["model"].lower()
            if any(p.lower() in model_lower for p in model_patterns):
                filtered.append(rec)
    else:
        filtered = records

    # Then filter by mode
    if "*" in match_patterns:
        return filtered
    result = []
    seen = set()
    for rec in filtered:
        mode_lower = rec["mode"].lower()
        key = (rec["model"], rec["mode"])
        for pattern in match_patterns:
            if pattern.lower() in mode_lower and key not in seen:
                result.append(rec)
                seen.add(key)
                break
    return result


def find_baseline(records, baseline_pattern):
    """Find the baseline record by substring match. Returns None if not found."""
    if not baseline_pattern:
        return None
    for rec in records:
        if baseline_pattern.lower() in rec["mode"].lower():
            return rec
    return None


def generate_html(all_data, params, groups, baseline_pattern):
    """Generate an interactive HTML dashboard from the parsed data."""

    # Flatten all records into a single pool, each tagged with its model
    all_records = []
    for model_data in all_data:
        for rec in model_data["records"]:
            all_records.append({**rec, "model": model_data["model"]})

    html = f"""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Benchmark Dashboard</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=JetBrains+Mono:wght@400;500&display=swap');

  :root {{
    --bg: #0f1117;
    --surface: #1a1d27;
    --surface-hover: #222632;
    --border: #2a2e3a;
    --text: #e4e6ed;
    --text-dim: #8b90a0;
    --accent-1: #6c8cff;
    --accent-2: #ff6c8c;
    --accent-3: #6cffc8;
    --accent-4: #ffc86c;
    --accent-5: #c86cff;
    --accent-6: #6cfff0;
  }}

  * {{ margin: 0; padding: 0; box-sizing: border-box; }}

  body {{
    font-family: 'DM Sans', sans-serif;
    background: var(--bg);
    color: var(--text);
    min-height: 100vh;
    padding: 2rem;
  }}

  .header {{
    max-width: 1200px;
    margin: 0 auto 2.5rem;
  }}

  .header h1 {{
    font-size: 1.75rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    margin-bottom: 0.5rem;
  }}

  .header .params {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    color: var(--text-dim);
    background: var(--surface);
    display: inline-block;
    padding: 0.4rem 0.8rem;
    border-radius: 6px;
    border: 1px solid var(--border);
  }}

  .group-section {{
    max-width: 1200px;
    margin: 0 auto 2.5rem;
  }}

  .group-title {{
    font-size: 1.15rem;
    font-weight: 600;
    color: var(--accent-4);
    margin-bottom: 1rem;
    padding-left: 0.5rem;
    border-left: 3px solid var(--accent-4);
  }}

  .charts-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(340px, 1fr));
    gap: 1.5rem;
  }}

  .chart-card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem;
    transition: border-color 0.2s;
  }}

  .chart-card:hover {{
    border-color: var(--accent-1);
  }}

  .chart-card h3 {{
    font-size: 0.95rem;
    font-weight: 500;
    color: var(--text-dim);
    margin-bottom: 1.2rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }}

  .bar-group {{
    margin-bottom: 0.85rem;
  }}

  .bar-label {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: var(--text-dim);
    margin-bottom: 0.3rem;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }}

  .bar-track {{
    width: 100%;
    height: 32px;
    background: var(--bg);
    border-radius: 6px;
    position: relative;
    overflow: hidden;
  }}

  .bar-fill {{
    height: 100%;
    border-radius: 6px;
    display: flex;
    align-items: center;
    justify-content: flex-end;
    padding-right: 10px;
    min-width: 60px;
    transition: width 0.8s cubic-bezier(0.22, 1, 0.36, 1);
    position: relative;
  }}

  .bar-fill::after {{
    content: '';
    position: absolute;
    inset: 0;
    border-radius: 6px;
    background: linear-gradient(90deg, transparent 60%, rgba(255,255,255,0.08));
  }}

  .bar-value {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    font-weight: 500;
    color: #fff;
    position: relative;
    z-index: 1;
    text-shadow: 0 1px 3px rgba(0,0,0,0.4);
  }}

  .legend {{
    display: flex;
    flex-wrap: wrap;
    gap: 1rem;
    margin-top: 0.5rem;
    padding-top: 1rem;
    border-top: 1px solid var(--border);
  }}

  .legend-item {{
    display: flex;
    align-items: center;
    gap: 0.4rem;
    font-size: 0.78rem;
    color: var(--text-dim);
  }}

  .legend-swatch {{
    width: 12px;
    height: 12px;
    border-radius: 3px;
  }}

  .higher-better {{ color: var(--accent-3); font-size: 0.7rem; font-weight: 500; }}
  .lower-better {{ color: var(--accent-2); font-size: 0.7rem; font-weight: 500; }}

  .pct-badge {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    font-weight: 600;
    padding: 0.15rem 0.45rem;
    border-radius: 4px;
    margin-left: 8px;
    display: inline-block;
    position: relative;
    z-index: 1;
    text-shadow: 0 1px 2px rgba(0,0,0,0.3);
  }}
  .pct-good {{
    background: rgba(0, 180, 120, 0.85);
    color: #fff;
  }}
  .pct-bad {{
    background: rgba(220, 50, 80, 0.85);
    color: #fff;
  }}
  .pct-baseline {{
    background: rgba(255, 255, 255, 0.2);
    color: #fff;
  }}
</style>
</head>
<body>

<div class="header">
  <h1>Benchmark Results Dashboard</h1>
  <div class="params">max_num_seqs={params['max_num_seqs']}  &middot;  input_len={params['random_input_len']}  &middot;  output_len={params['random_output_len']}</div>
</div>
"""

    colors = ["var(--accent-1)", "var(--accent-2)", "var(--accent-3)",
              "var(--accent-4)", "var(--accent-5)", "var(--accent-6)"]

    metrics = [
        ("total_token_throughput", "Total Token Throughput (tokens/s)", True),
        ("mean_itl_ms", "Mean ITL (ms)", False),
        ("p99_itl_ms", "P99 ITL (ms)", False),
    ]

    for group in groups:
        group_name = group["name"]
        model_patterns = group.get("models", None)
        group_records = filter_records_by_group(all_records, group["match"], model_patterns)

        if not group_records:
            continue

        html += f'<div class="group-section">\n'
        html += f'  <div class="group-title">{group_name}</div>\n'
        html += f'  <div class="charts-grid">\n'

        for metric_key, metric_label, higher_is_better in metrics:
            values = [r[metric_key] for r in group_records]
            max_val = max(values) if values else 1
            direction = "higher-better" if higher_is_better else "lower-better"
            direction_label = "&#9650; higher is better" if higher_is_better else "&#9660; lower is better"

            # Find baseline value for this metric in this group
            baseline_rec = find_baseline(group_records, baseline_pattern)
            baseline_val = baseline_rec[metric_key] if baseline_rec else None

            html += f'    <div class="chart-card">\n'
            html += f'      <h3>{metric_label} <span class="{direction}">{direction_label}</span></h3>\n'

            for idx, rec in enumerate(group_records):
                val = rec[metric_key]
                pct = (val / max_val * 100) if max_val > 0 else 0
                pct = max(pct, 5)
                color = colors[idx % len(colors)]
                label = rec["mode"]

                if metric_key == "total_token_throughput":
                    formatted = f"{{:.1f}}".format(val)
                else:
                    formatted = f"{{:.2f}}".format(val)

                # Compute percentage diff badge
                pct_badge = ""
                if baseline_val is not None:
                    is_baseline = (baseline_rec and rec["mode"] == baseline_rec["mode"]
                                   and rec["model"] == baseline_rec["model"])
                    if is_baseline:
                        pct_badge = '<span class="pct-badge pct-baseline">baseline</span>'
                    elif baseline_val != 0:
                        diff_pct = ((val - baseline_val) / abs(baseline_val)) * 100
                        sign = "+" if diff_pct > 0 else ""
                        if higher_is_better:
                            css_class = "pct-good" if diff_pct >= 0 else "pct-bad"
                        else:
                            css_class = "pct-good" if diff_pct <= 0 else "pct-bad"
                        pct_badge = f'<span class="pct-badge {css_class}">{sign}{diff_pct:.1f}%</span>'

                html += f'      <div class="bar-group">\n'
                html += f'        <div class="bar-label" title="{label}">{label}</div>\n'
                html += f'        <div class="bar-track">\n'
                html += f'          <div class="bar-fill" style="width:{pct:.1f}%;background:{color};">\n'
                html += f'            <span class="bar-value">{formatted}{pct_badge}</span>\n'
                html += f'          </div>\n'
                html += f'        </div>\n'
                html += f'      </div>\n'

            html += f'    </div>\n'

        html += f'  </div>\n'

        # Legend for this group
        html += f'  <div class="legend">\n'
        for idx, rec in enumerate(group_records):
            color = colors[idx % len(colors)]
            html += f'    <div class="legend-item"><div class="legend-swatch" style="background:{color}"></div>{rec["mode"]}</div>\n'
        html += f'  </div>\n'
        html += f'</div>\n'

    html += """\
</body>
</html>
"""
    return html


def main():
    parser = argparse.ArgumentParser(
        description="Convert benchmark Excel files to HTML bar charts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Config file example (groups.json):
  {
    "baseline": "vanilla",
    "groups": [
      { "name": "Baseline vs Eager",           "match": ["vanilla", "eager"] },
      { "name": "All Modes",                   "match": ["*"] },
      { "name": "Llama Only",                  "match": ["*"], "models": ["llama"] },
      { "name": "Cross-Model Vanilla",         "match": ["vanilla"], "models": ["*"] }
    ]
  }

"baseline": substring to identify the reference mode (e.g. "vanilla").
  Each bar shows % diff vs the baseline. Omit to disable.
"match": substrings matched against mode names (case-insensitive). ["*"] = all.
"models" (optional): substrings matched against model names (from filename).
  ["*"] or omit = all models. Modes from different files mix in the same chart.
        """)
    parser.add_argument("files", nargs="+", help="Excel files (one per model)")
    parser.add_argument("--max_num_seqs", type=float, default=64)
    parser.add_argument("--random_input_len", type=float, default=512)
    parser.add_argument("--random_output_len", type=float, default=128)
    parser.add_argument("--config", type=str, default=None,
                        help="JSON config file defining chart groups (optional)")
    parser.add_argument("-o", "--output", default="benchmark_dashboard.html")
    args = parser.parse_args()

    params = {
        "max_num_seqs": int(args.max_num_seqs),
        "random_input_len": int(args.random_input_len),
        "random_output_len": int(args.random_output_len),
    }

    # Load groups config
    if args.config:
        with open(args.config) as f:
            config = json.load(f)
        groups = config["groups"]
        baseline_pattern = config.get("baseline", None)
        print(f"Loaded {len(groups)} chart group(s) from {args.config}"
              + (f", baseline='{baseline_pattern}'" if baseline_pattern else ""))
    else:
        groups = DEFAULT_CONFIG["groups"]
        baseline_pattern = DEFAULT_CONFIG["baseline"]

    all_data = []
    for f in args.files:
        result = parse_excel(f, args.max_num_seqs, args.random_input_len, args.random_output_len)
        if result:
            all_data.append(result)

    if not all_data:
        print("ERROR: No data found matching the given parameters.", file=sys.stderr)
        sys.exit(1)

    for d in all_data:
        modes = [r["mode"] for r in d["records"]]
        print(f"Model '{d['model']}' — modes found: {modes}")

    html = generate_html(all_data, params, groups, baseline_pattern)
    output_path = Path(args.output)
    output_path.write_text(html)
    print(f"Dashboard written to {output_path} ({len(all_data)} model(s), {len(groups)} group(s))")


if __name__ == "__main__":
    main()
