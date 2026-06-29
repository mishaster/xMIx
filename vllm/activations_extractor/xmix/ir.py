# SPDX-License-Identifier: Apache-2.0
"""Intermediate representation produced by the xmix parser (Phase A).

These dataclasses are GPU-free and hold only validated, symbolic information.
Arguments are recorded as *symbolic name strings* (registry keys) — they are
never literals and are not resolved to real objects until Phase B (lowering).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class OpKind(Enum):
    WRITE = "w"   # m.write(...)
    READ = "r"    # m.read(...)  — reserved; SPEC-ready, not used by the v1 example


@dataclass(frozen=True)
class SourceLoc:
    file: str
    line: int        # 1-based (ast lineno)
    col: int         # 0-based (ast col_offset)


@dataclass(frozen=True)
class CondClause:
    """The ``.cond(...)`` clause — two forms share the install fields below.

    Probe-class form (steer.xmix):
        ``LinearProbe.biggerThan(a, b).layer([..]).submodule("..")``
        ⇒ probe_op / compare_op / args set; the compiler synthesizes the probe
          + the user-defined bridge.

    Prebuilt-instance form (app3.xmix):
        ``self.cosine_similarity_instance.gates("vec_indices").layer([..]).submodule("..")``
        ⇒ instance_expr / cond_cls / gate_buffer set; the condition object is
          hand-built in the preamble and the DSL only declares the gating
          relationship. The plug fn/kind default from SPEC["cond_instances"]
          (overridable via ``.gates(buf, via=.., kind=..)``).
    """

    layers: tuple[int, ...]       # from .layer([18]) — common to both forms
    submodule: str                # e.g. "mlp.post" — common to both forms
    loc: SourceLoc
    # ── probe-class form ──
    probe_op: str | None = None       # SPEC cond-op key, e.g. "LinearProbe"
    compare_op: str | None = None     # compare method key, e.g. "biggerThan"
    args: tuple[str, ...] = ()        # symbolic names, e.g. ("cond_arg1", "cond_arg2")
    # ── prebuilt-instance ("gates") form ──
    instance_expr: str | None = None  # e.g. "self.cosine_similarity_instance"
    cond_cls: str | None = None       # resolved class, e.g. "CondProjCosSim"
    method: str = "run"               # condition read-hook method
    gate_buffer: str | None = None    # steerer buffer name, e.g. "vec_indices"
    plug_fn: str | None = None        # override of SPEC plug_fn, else None
    plug_kind: str | None = None      # override of SPEC plug_kind, else None
    # .layer("model"): the condition runs at model scope (LlamaModel.forward),
    # not as a per-layer hook. submodule is "" and layers is () in this form.
    model_level: bool = False


@dataclass(frozen=True)
class Statement:
    """One top-level ``m.write(...)...`` statement."""

    op: OpKind
    op_name: str                  # SPEC op key, e.g. "SteeringVector"
    args: tuple[str, ...]         # symbolic names, e.g. ("arg1",)
    method: str                   # trailing attr, e.g. "run"
    layers: tuple[int, ...]       # from .layers([23, 24, 25]); () when all_layers
    submodule: str                # e.g. "attention.post"
    cond: CondClause | None       # optional conditional clause
    loc: SourceLoc
    # Hand-written instance reference (e.g. "self.steer_refusal_vec"): when set,
    # Phase B skips construction synthesis and installs this object's method
    # directly. None ⇒ synthesized path (compiler builds the object from args).
    instance_expr: str | None = None
    # .layer("all"): install on every decoder layer, no idx filter. layers is ().
    all_layers: bool = False


@dataclass(frozen=True)
class PreambleSegment:
    """One verbatim non-DSL top-level node (raw Python the user hand-writes).

    ``targets`` are the dotted assignment targets in that node (e.g.
    ``("self.token_detector",)``), empty for imports/bare expressions. Phase B
    uses them to label which preamble line builds a model-scope instance and
    therefore belongs in a different destination (see codegen destination markers).

    ``block_cls`` is set when this segment sits inside a ``#token_detector: init``
    … ``#token_detector: end`` block: the (model-scope) class that block
    constructs. Every segment in the block — supporting imports/token logic *and*
    the construction — carries it, so the whole block routes to that class's
    destination together.
    """

    text: str
    targets: tuple[str, ...] = ()
    block_cls: str | None = None


@dataclass(frozen=True)
class ParseResult:
    """Output of Phase A: verbatim preamble segments plus validated statements."""

    preamble: tuple[PreambleSegment, ...]  # raw non-DSL top-level source, in file order
    statements: tuple[Statement, ...]
    # Model family declared via `#model: <name>` (else SPEC["default_model"]).
    # Selects which per-model file each apply-anchor resolves to.
    model: str = "llama"
