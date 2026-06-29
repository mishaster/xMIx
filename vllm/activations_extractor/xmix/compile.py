# SPDX-License-Identifier: Apache-2.0
"""CLI for the xmix compiler.

Usage:
    python -m vllm.activations_extractor.xmix.compile steer.xmix [-o out.py]

Prints the synthesized Python source to stdout, and optionally writes it to a
file. On a validation error, prints the located message to stderr and exits 1.
"""

from __future__ import annotations

import argparse
import sys

from vllm.activations_extractor.xmix.codegen import compile_xmix
from vllm.activations_extractor.xmix.errors import XmixError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m vllm.activations_extractor.xmix.compile",
        description="Translate an .xmix DSL file into synthesized hook code.",
    )
    parser.add_argument("path", help="path to the .xmix file")
    parser.add_argument("-o", "--output", default=None,
                        help="also write the synthesized code to this file")
    args = parser.parse_args(argv)

    try:
        code = compile_xmix(args.path)
    except XmixError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except OSError as e:
        print(f"error: cannot read {args.path}: {e}", file=sys.stderr)
        return 1

    sys.stdout.write(code)
    if args.output:
        with open(args.output, "w") as f:
            f.write(code)
        print(f"wrote {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
