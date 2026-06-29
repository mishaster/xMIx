# SPDX-License-Identifier: Apache-2.0
"""Error type for the xmix DSL, with located (file:line:col) messages."""

from __future__ import annotations

from collections.abc import Iterable

from vllm.activations_extractor.xmix.ir import SourceLoc


class XmixError(Exception):
    """Raised on any xmix validation/lowering failure.

    Formats as ``<file>:<line>:<col>: <msg>  (valid: ...)`` so the user gets an
    immediately actionable, located error. Column is reported 1-based for human
    consumption (ast stores it 0-based).
    """

    def __init__(
        self,
        loc: SourceLoc | None,
        msg: str,
        alternatives: Iterable[str] | None = None,
    ):
        self.loc = loc
        self.msg = msg
        self.alternatives = sorted(alternatives) if alternatives is not None else None

        if loc is not None:
            located = f"{loc.file}:{loc.line}:{loc.col + 1}: {msg}"
        else:
            located = msg
        if self.alternatives:
            located += f"  (valid: {', '.join(self.alternatives)})"
        super().__init__(located)
