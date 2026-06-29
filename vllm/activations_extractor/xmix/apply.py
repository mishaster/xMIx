# SPDX-License-Identifier: Apache-2.0
"""Apply / revert synthesized xmix blocks into the live vLLM source.

This is the *external overlay* step: it physically splices the blocks from a
synthesized ``appN_synth.py`` into the matching vLLM source files, wrapping each
in begin/end sentinels so it can be removed again cleanly. The synthesized code
is NOT imported or exec'd — it is inserted as ordinary source the normal vLLM
import then loads.

How targeting works (no line numbers, nothing in the .xmix):
  * The compiler emits, above each block, a machine tag
        # >>> XMIX-APPLY file=<file> anchor=<anchor> <<<
    where <file> is the per-model file (resolved from the .xmix `#model:` pragma)
    and <anchor> is a landmark name.
  * You add the landmark once, as a harmless comment, in each target file:
        # xmix:anchor <anchor>
  * This tool finds that comment, captures its indentation, and inserts the
    block right after it, re-indented, wrapped as:
        # xmix:begin <app> <anchor>
        ... block (re-indented) ...
        # xmix:end <app> <anchor>

Reverting strips exactly those sentinel regions — nothing else is touched, so
the only lasting change to vLLM is the (harmless) anchor comments.

Multiple synth files can be applied at once; their blocks stack at each shared
anchor in the order listed (additively — already-applied blocks are kept), and
the batch is all-or-nothing (a missing anchor aborts before anything is written).

Usage:
    python -m vllm.activations_extractor.xmix.apply app2_synth.py [--dry-run]
    python -m vllm.activations_extractor.xmix.apply app2_synth.py app3_synth.py app6_synth.py
    python -m vllm.activations_extractor.xmix.apply --revert app2_synth.py app3_synth.py
    python -m vllm.activations_extractor.xmix.apply --revert --app app2
    python -m vllm.activations_extractor.xmix.apply --status
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from vllm.activations_extractor.xmix.spec import SPEC

_APPLY_RE = re.compile(r"^#\s*>>> XMIX-APPLY file=(\S+) anchor=(\S+) <<<\s*$")
_FOOTPRINT_RE = re.compile(
    r"^#\s*>>> XMIX-FOOTPRINT flag=(\S+) submodule=(\S+) layers=(\S+) <<<")
# Persisted footprint comment inside an applied app's sentinels (app-tagged).
_APPLIED_FP_RE = re.compile(r"^#\s*xmix:footprint (\S+) (\S+) (\S+) (\S+)")
# Meta lines emitted by codegen that must NOT be injected (display only).
_META_PREFIXES = ("# >>> ", "#     (executes at:", "# === ", "# --- user preamble")

_ALL = "all"  # layer-set sentinel: every decoder layer


class ApplyError(Exception):
    """A located-free error in the apply/revert step."""


# ──────────────────────────────────────────────────────────────────────────
# parsing the synthesized file into (file, anchor, code) blocks
# ──────────────────────────────────────────────────────────────────────────

def parse_blocks(text: str) -> list[tuple[str, str, list[str]]]:
    """Split synthesized source into ``(file, anchor, code_lines)`` per region.

    A ``# >>> XMIX-APPLY file=… anchor=… <<<`` line opens a region; codegen meta
    lines are dropped; everything else (code + provenance comments) is the block.
    """
    blocks: list[tuple[str, str, list[str]]] = []
    cur: tuple[str, str, list[str]] | None = None
    for line in text.splitlines():
        m = _APPLY_RE.match(line.strip())
        if m:
            if cur is not None:
                blocks.append(cur)
            cur = (m.group(1), m.group(2), [])
            continue
        if cur is None:
            continue  # file header, before the first tag
        if _is_meta(line):
            continue
        cur[2].append(line)
    if cur is not None:
        blocks.append(cur)
    return [(f, a, ls) for f, a, c in blocks if (ls := _trim_blank(c))]


def _is_meta(line: str) -> bool:
    s = line.strip()
    return any(s.startswith(p) for p in _META_PREFIXES)


def _trim_blank(lines: list[str]) -> list[str]:
    start, end = 0, len(lines)
    while start < end and not lines[start].strip():
        start += 1
    while end > start and not lines[end - 1].strip():
        end -= 1
    return lines[start:end]


# ──────────────────────────────────────────────────────────────────────────
# footprints: the per-layer hook slots an intervention occupies (conflict check)
# ──────────────────────────────────────────────────────────────────────────

def _parse_layers(spec: str):
    return _ALL if spec == _ALL else frozenset(
        int(x) for x in spec.split(",") if x)


def _layers_str(layers) -> str:
    return _ALL if layers == _ALL else ",".join(str(x) for x in sorted(layers))


def parse_footprints(text: str) -> list[tuple[str, str, object]]:
    """``XMIX-FOOTPRINT`` tags from a synth file → ``(flag, submodule, layers)``
    where layers is a frozenset of ints or the ``_ALL`` sentinel."""
    out = []
    for line in text.splitlines():
        m = _FOOTPRINT_RE.match(line.strip())
        if m:
            out.append((m.group(1), m.group(2), _parse_layers(m.group(3))))
    return out


def _scan_applied_footprints() -> list[tuple[str, str, str, object]]:
    """Persisted ``# xmix:footprint <app> <flag> <submodule> <layers>`` lines
    across all target files → ``(app, flag, submodule, layers)`` of applied apps."""
    out = []
    for file in _all_anchor_files():
        path = Path(file)
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            m = _APPLIED_FP_RE.match(line.strip())
            if m:
                out.append((m.group(1), m.group(2), m.group(3),
                            _parse_layers(m.group(4))))
    return out


def _overlap(a, b) -> bool:
    return a == _ALL or b == _ALL or bool(a & b)


def _overlap_str(a, b) -> str:
    if a == _ALL or b == _ALL:
        return _ALL
    return "{" + ",".join(str(x) for x in sorted(a & b)) + "}"


# ──────────────────────────────────────────────────────────────────────────
# source-file editing (anchors + sentinels)
# ──────────────────────────────────────────────────────────────────────────

def _anchor_comment(anchor: str) -> str:
    return f"# xmix:anchor {anchor}"


def _indent_of(line: str) -> str:
    return line[:len(line) - len(line.lstrip())]


def _find_anchor(lines: list[str], anchor: str) -> int | None:
    target = _anchor_comment(anchor)
    for i, line in enumerate(lines):
        if line.strip() == target:
            return i
    return None


def _strip_app_regions(lines: list[str],
                       app_ids: set[str] | None) -> list[str]:
    """Remove every ``# xmix:begin <app> …`` … ``# xmix:end <app> …`` region whose
    app is in ``app_ids``. ``app_ids=None`` strips all apps' regions. Idempotent.
    """
    out: list[str] = []
    skipping = False
    for line in lines:
        s = line.strip()
        if not skipping and s.startswith("# xmix:begin "):
            app = s.split()[2] if len(s.split()) > 2 else None
            if app_ids is None or app in app_ids:
                skipping = True
                continue
        if skipping and s.startswith("# xmix:end "):
            skipping = False
            continue
        if not skipping:
            out.append(line)
    return out


def _insertion_point(lines: list[str], anchor_idx: int) -> int:
    """Return the index to insert at: just after the anchor line, advanced past
    any existing ``# xmix:begin … # xmix:end …`` regions (and blank lines)
    directly following it — so a new block stacks *below* already-applied ones."""
    i = anchor_idx + 1
    n = len(lines)
    while i < n:
        s = lines[i].strip()
        if not s:
            i += 1
            continue
        if s.startswith("# xmix:begin "):
            i += 1
            while i < n and not lines[i].strip().startswith("# xmix:end "):
                i += 1
            i += 1  # past the end marker
            continue
        break
    return i


def _insert_stacked(lines: list[str], anchor: str, app_id: str,
                    code: list[str]) -> list[str]:
    """Insert ``code`` (sentinel-wrapped, re-indented to the anchor) below the
    anchor and any blocks already stacked there. Anchor presence is pre-validated."""
    idx = _find_anchor(lines, anchor)
    indent = _indent_of(lines[idx])
    block = [f"{indent}# xmix:begin {app_id} {anchor}"]
    block += [(indent + cl) if cl.strip() else "" for cl in code]
    block.append(f"{indent}# xmix:end {app_id} {anchor}")
    at = _insertion_point(lines, idx)
    return lines[:at] + block + lines[at:]


def _all_anchor_files() -> list[str]:
    files: set[str] = set()
    for m in SPEC["models"].values():
        files.update(m.get("anchors", {}).values())
    return sorted(files)


# ──────────────────────────────────────────────────────────────────────────
# commands
# ──────────────────────────────────────────────────────────────────────────

def cmd_apply(paths: list[str], app_override: str | None,
              dry_run: bool) -> None:
    """Apply one or more synth files. Blocks stack at each anchor in CLI order,
    additively (existing apps kept); all-or-nothing if any anchor is missing."""
    # Build the plan in CLI order; each file's app id from its name (or override).
    plan: list[tuple[str, str, str, list[str]]] = []  # (file, anchor, app, code)
    batch_apps: list[str] = []
    app_fps: dict[str, list[tuple[str, str, object]]] = {}
    for p in paths:
        app_id = _app_id_from(p, app_override)
        if app_id in batch_apps:
            raise ApplyError(f"duplicate app id '{app_id}' in this invocation")
        batch_apps.append(app_id)
        text = Path(p).read_text()
        blocks = parse_blocks(text)
        if not blocks:
            raise ApplyError(f"no XMIX-APPLY blocks found in {p}")
        for file, anchor, code in blocks:
            plan.append((file, anchor, app_id, code))
        app_fps[app_id] = parse_footprints(text)

    # Refuse overlapping interventions (same flag+submodule+layers) within the
    # batch and against already-applied apps. All-or-nothing — before any write.
    _check_conflicts(batch_apps, app_fps)

    # Group by target file, preserving order.
    by_file: dict[str, list[tuple[str, str, list[str]]]] = {}
    for file, anchor, app, code in plan:
        by_file.setdefault(file, []).append((anchor, app, code))

    # Pass 1: read every target file once and validate all anchors up front.
    file_lines: dict[str, list[str]] = {}
    missing: list[str] = []
    for file, items in by_file.items():
        path = Path(file)
        if not path.exists():
            missing.append(f"{file} (file not found)")
            continue
        lines = path.read_text().splitlines()
        file_lines[file] = lines
        for anchor in dict.fromkeys(a for a, _, _ in items):
            if _find_anchor(lines, anchor) is None:
                missing.append(f"'{_anchor_comment(anchor)}' in {file}")
    if missing:
        raise ApplyError("aborting (nothing written) — missing: "
                         + "; ".join(missing))

    # Pass 2: refresh this batch's apps, stack the blocks, write/preview once.
    # An app's footprints are persisted into its runner_postload block so the
    # conflict check can see them on a later apply (removed on revert).
    batch_set = set(batch_apps)
    for file, items in by_file.items():
        lines = _strip_app_regions(file_lines[file], batch_set)
        for anchor, app, code in items:
            if anchor == "runner_postload" and app_fps.get(app):
                code = [f"# xmix:footprint {app} {flag} {sub} {_layers_str(lay)}"
                        for flag, sub, lay in app_fps[app]] + code
            lines = _insert_stacked(lines, anchor, app, code)
        text = "\n".join(lines) + "\n"
        if dry_run:
            sys.stdout.write(f"\n===== {file} (dry-run) =====\n{text}")
        else:
            Path(file).write_text(text)
            print(f"applied {sorted(batch_set)} -> {file}", file=sys.stderr)


def _check_conflicts(batch_apps: list[str],
                     app_fps: dict[str, list[tuple[str, str, object]]]) -> None:
    """Raise if any per-layer slot (flag+submodule+overlapping layers) is claimed
    by two apps — within this batch or against an already-applied app."""
    batch = [(app, flag, sub, lay)
             for app in batch_apps for flag, sub, lay in app_fps.get(app, [])]
    existing = [e for e in _scan_applied_footprints()
                if e[0] not in set(batch_apps)]  # batch apps get refreshed
    for i, (app, flag, sub, lay) in enumerate(batch):
        for app2, flag2, sub2, lay2 in batch[i + 1:] + existing:
            if flag == flag2 and sub == sub2 and _overlap(lay, lay2):
                raise ApplyError(
                    f"'{app}' conflicts with '{app2}' at ({flag}, {sub}) "
                    f"layers {_overlap_str(lay, lay2)}; nothing applied "
                    f"(only one hook would survive at that slot)")


def cmd_revert(paths: list[str], app_override: str | None) -> None:
    """Revert blocks. With synth paths, strip each file's derived app id; with no
    paths, strip `--app` everywhere, or all xmix blocks if no `--app` either."""
    if paths:
        app_ids: set[str] | None = {_app_id_from(p, app_override) for p in paths}
        files = sorted({f for p in paths
                        for f, _, _ in parse_blocks(Path(p).read_text())})
    else:
        app_ids = {app_override} if app_override else None
        files = _all_anchor_files()
    label = ", ".join(sorted(app_ids)) if app_ids else "all"
    for file in files:
        path = Path(file)
        if not path.exists():
            continue
        lines = path.read_text().splitlines()
        stripped = _strip_app_regions(lines, app_ids)
        if stripped != lines:
            path.write_text("\n".join(stripped) + "\n")
            print(f"reverted '{label}' <- {file}", file=sys.stderr)


def cmd_status() -> None:
    found = False
    for file in _all_anchor_files():
        path = Path(file)
        if not path.exists():
            continue
        for i, line in enumerate(path.read_text().splitlines(), start=1):
            s = line.strip()
            if s.startswith("# xmix:begin "):
                parts = s.split()
                app = parts[2] if len(parts) > 2 else "?"
                anchor = parts[3] if len(parts) > 3 else "?"
                print(f"{file}:{i}  app={app} anchor={anchor}")
                found = True
    if not found:
        print("no xmix blocks currently applied")


def _app_id_from(path: str, override: str | None) -> str:
    if override:
        return override
    stem = Path(path).stem
    return stem[:-len("_synth")] if stem.endswith("_synth") else stem


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m vllm.activations_extractor.xmix.apply",
        description="Apply / revert synthesized xmix blocks into vLLM source.",
    )
    parser.add_argument("paths", nargs="*",
                        help="synthesized .py file(s), applied in this order "
                             "(e.g. app2_synth.py app3_synth.py)")
    parser.add_argument("--revert", action="store_true",
                        help="remove injected blocks instead of adding them")
    parser.add_argument("--status", action="store_true",
                        help="list currently-injected blocks and exit")
    parser.add_argument("--app", default=None,
                        help="override the app id (single file only; default: "
                             "synth filename stem without '_synth')")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the patched files instead of writing them")
    args = parser.parse_args(argv)

    if args.app and len(args.paths) > 1:
        parser.error("--app applies to a single file; omit it for multiple "
                     "(each file's app id comes from its name)")

    try:
        if args.status:
            cmd_status()
            return 0
        if args.revert:
            cmd_revert(args.paths, args.app)
            return 0
        if not args.paths:
            parser.error("at least one synth .py path is required to apply")
        cmd_apply(args.paths, args.app, args.dry_run)
        return 0
    except (ApplyError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
