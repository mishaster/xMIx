# SPDX-License-Identifier: Apache-2.0
"""Phase B (codegen): lower validated xmix IR into synthesized Python source.

This stage does NOT touch a live model. It translates the validated statements
into a readable block of Python — object construction (fully expanded) plus the
per-layer install loop — that belongs at the matching points in the runtime
code. The user inspects this text before we wire it into ``gpu_model_runner``.

Generated code references the DSL's symbolic names (``arg1``, ``cond_arg1`` ...)
and a few registry-convention names (``_max_tokens``, ``_cond_bias``,
``_steered_tokens_num``) as *bare names assumed in scope* at the insertion
point. A header comment lists every assumed-in-scope name.
"""

from __future__ import annotations

from vllm.activations_extractor.xmix.errors import XmixError
from vllm.activations_extractor.xmix.ir import OpKind, Statement
from vllm.activations_extractor.xmix.parser import parse_and_validate
from vllm.activations_extractor.xmix.spec import SPEC

# Registry-convention names the generated code assumes are in scope.
_MAX_TOKENS = "_max_tokens"
_COND_BIAS = "_cond_bias"
_STEERED_TOKENS_NUM = "_steered_tokens_num"

# The generated INSTALL code lives in a gpu_model_runner method, where ``self``
# is the runner and ``self.model.model`` is the inner (decoder) model that owns
# the layers and the model-scope detector.
_INNER_MODEL = "self.model.model"


def compile_xmix(path: str) -> str:
    """Parse + validate an .xmix file and return synthesized Python source."""
    result = parse_and_validate(path)
    return generate(result.statements, source_name=path,
                    preamble=result.preamble, model=result.model)


class _Sink:
    """Accumulates emitted lines, prefixing a machine-readable ``XMIX-APPLY`` tag
    plus a human ``DESTINATION`` banner whenever the destination changes (so a run
    of same-destination lines shares one tag). The apply tool keys on the
    ``XMIX-APPLY`` line; the banner/note are for the reader. Lines added without a
    preceding ``.to(...)`` are destination-neutral (e.g. the DSL echo comment).

    A *destination descriptor* is a dict ``{"label", "file", "anchor", "note"?}``:
    ``label``/``note`` are display text; ``file``/``anchor`` are the apply target.
    """

    def __init__(self) -> None:
        self.lines: list[str] = []
        self._cur: tuple[str, str] | None = None

    def to(self, dest: dict) -> None:
        key = (dest["file"], dest["anchor"])
        if key != self._cur:
            self.lines.append("")
            self.lines.append(
                f"# >>> XMIX-APPLY file={dest['file']} anchor={dest['anchor']} <<<")
            self.lines.append(f"# >>> DESTINATION: {dest['label']} <<<")
            if dest.get("note"):
                self.lines.append(f"#     (executes at: {dest['note']})")
            self._cur = key

    def add(self, *lines: str) -> None:
        self.lines.extend(lines)


def _dest(anchor: str, model: str) -> dict:
    """Runner destination descriptor: label from SPEC["destinations"], concrete
    file resolved per-model from SPEC["models"][model]["anchors"]."""
    return {
        "label": SPEC["destinations"][anchor],
        "anchor": anchor,
        "file": SPEC["models"][model]["anchors"][anchor],
    }


def _model_dest(cspec: dict, model: str) -> dict:
    """Model-scope construction descriptor for a cond instance (e.g. the token
    detector): file resolved per-model from the instance's ``dest_anchor``."""
    anchor = cspec["dest_anchor"]
    return {
        "label": cspec["dest"],
        "anchor": anchor,
        "file": SPEC["models"][model]["anchors"][anchor],
        "note": cspec.get("run_dest"),
    }


def generate(statements: list[Statement], source_name: str = "<xmix>",
             preamble: tuple = (), model: str | None = None) -> str:
    """Translate validated statements (+ verbatim preamble) into Python source.

    ``model`` selects which per-model file each apply-anchor resolves to (from
    SPEC["models"]); defaults to SPEC["default_model"]."""
    model = model or SPEC["default_model"]
    setup = _Sink()
    install = _Sink()
    assumed: set[str] = {"torch"}

    # Preamble assignments that build a model-scope cond instance belong in a
    # different destination (the model file). Map their reference -> class.
    model_bound = {
        s.cond.instance_expr: s.cond.cond_cls
        for s in statements
        if s.cond is not None and s.cond.model_level and s.cond.instance_expr
    }

    if preamble:
        setup.add("", "# --- user preamble (verbatim) ---")
        for seg in preamble:
            # A `#token_detector: init` block routes the whole block (imports +
            # custom token logic + construction) to the model destination;
            # otherwise a lone model-scope construction line is routed by target.
            cls = seg.block_cls or next(
                (model_bound[t] for t in seg.targets if t in model_bound), None)
            if cls is not None:
                setup.to(_model_dest(SPEC["cond_instances"][cls], model))
            else:
                setup.to(_dest("runner_setup", model))
            setup.add(*seg.text.splitlines())

    for i, stmt in enumerate(statements):
        _emit_statement(i, stmt, setup, install, assumed, model)

    header = [
        f"# xmix synthesized from {source_name}  (model: {model})",
        "# assumes in scope: " + ", ".join(sorted(assumed)),
        "#",
        "# Excerpts are grouped by destination. Each '# >>> XMIX-APPLY file=… "
        "anchor=… <<<'",
        "# tag is read by the apply tool to splice the block below it at that "
        "anchor;",
        "# the '# >>> DESTINATION: … <<<' banner is the human-readable label.",
    ]
    # Footprint tags: the per-layer hook slots this file occupies. The apply tool
    # reads them to refuse overlapping interventions (same flag+submodule+layer).
    header += _footprint_tags(statements)
    lines = header + ["", "# === SETUP (construct once — see destinations) ==="]
    lines += setup.lines
    lines += ["", "# === INSTALL (after load_model) ==="]
    lines += install.lines
    return "\n".join(lines) + "\n"


def _footprint_tags(statements: list[Statement]) -> list[str]:
    """One ``# >>> XMIX-FOOTPRINT … <<<`` per per-layer hook slot occupied: the
    steering op, plus a per-layer cond's read hook. Model-scope conds occupy no
    per-layer slot and are skipped."""
    tags: list[str] = []
    for stmt in statements:
        flag = SPEC["ops"][stmt.op_name].flag
        tags.append(_footprint_tag(flag, stmt.submodule, stmt.layers,
                                   stmt.all_layers))
        cond = stmt.cond
        if cond is not None and not cond.model_level:
            tags.append(_footprint_tag(_cond_flag(cond), cond.submodule,
                                       cond.layers, False))
    return tags


def _footprint_tag(flag: str, submodule: str, layers: tuple[int, ...],
                   all_layers: bool) -> str:
    spec = "all" if all_layers else ",".join(str(x) for x in sorted(layers))
    return (f"# >>> XMIX-FOOTPRINT flag={flag} submodule={submodule} "
            f"layers={spec} <<<")


# ──────────────────────────────────────────────────────────────────────────
# per-statement emission
# ──────────────────────────────────────────────────────────────────────────

def _emit_statement(i: int, stmt: Statement, setup: _Sink,
                    install: _Sink, assumed: set[str], model: str) -> None:
    op_spec = SPEC["ops"][stmt.op_name]
    sub_spec = SPEC["submodules"][stmt.submodule]
    cls_name = _cls_name(op_spec.cls_path)

    # ── SETUP: construct the steerer ──────────────────────────────────────
    if stmt.instance_expr is not None:
        # Hand-written instance: built in the preamble, referenced directly.
        steer_var = stmt.instance_expr
    else:
        assumed.add(cls_name)
        for a in stmt.args:
            assumed.add(a)
        steer_var = f"sv_{i}"
        setup.to(_dest("runner_setup", model))
        setup.add(*_emit_op_build(op_spec, cls_name, steer_var, stmt.args, i))

    # ── SETUP: construct cond probe + bridge ──────────────────────────────
    # Model-scope conds (e.g. the token detector) wire entirely in INSTALL; they
    # also need their forward run-call injected into the model file.
    cond_var = None
    if stmt.cond is not None and not stmt.cond.model_level:
        cond_var = _emit_cond_setup(i, stmt, steer_var, setup, assumed, model)
    elif stmt.cond is not None and stmt.cond.model_level:
        _emit_model_run(stmt.cond, setup, model)

    # ── INSTALL: per-layer loop ───────────────────────────────────────────
    method_arity = _method_arity(stmt.op_name, op_spec.method)
    site_arity = sub_spec["write_arity" if op_spec.flag == "w" else "read_arity"]
    hook_expr = _reconcile_expr(
        f"{steer_var}.{op_spec.method}", method_arity, site_arity,
        _extra_arg_exprs(stmt.op_name, steer_var), assumed)

    install.to(_dest("runner_postload", model))
    install.add("")
    # The DSL echo comment documents the whole statement at its install site.
    install.add(*_dsl_comment(stmt))
    where = "ALL layers" if stmt.all_layers else f"layers {list(stmt.layers)}"
    install.add(f"# steer {stmt.op_name} on {where} "
                f"at '{stmt.submodule}' (flag '{op_spec.flag}')")
    install.add(f"for layer in {_INNER_MODEL}.layers:")
    if stmt.all_layers:
        install.add(f"    layer.{sub_spec['setter']}({hook_expr}, "
                    f"\"{op_spec.flag}\")")
    else:
        install.add("    idx = layer.self_attn.layer_idx")
        install.add(f"    if idx in {_tuple_literal(stmt.layers)}:")
        install.add(f"        layer.{sub_spec['setter']}({hook_expr}, "
                    f"\"{op_spec.flag}\")")
    if stmt.cond is not None:
        if stmt.cond.model_level:
            _emit_model_cond_install(stmt.cond, steer_var, install, model)
        else:
            cond_sub = SPEC["submodules"][stmt.cond.submodule]
            install.add(f"    if idx in {_tuple_literal(stmt.cond.layers)}:")
            install.add(f"        layer.{cond_sub['setter']}({cond_var}, "
                        f"\"{_cond_flag(stmt.cond)}\")")


def _emit_model_cond_install(cond, steer_var: str, install: _Sink,
                             model: str) -> None:
    """Model-scope cond (e.g. the token detector): wire its output buffer to the
    steerer's gate buffer, outside the per-layer loop.

    The detector is constructed in the model file (model_construct anchor) and
    invoked there (model_forward anchor), so the model already owns it — we only
    wire its sink here. The plug call uses a positional list when plug_kind is
    None (set_output_buffers([..])), else the kwarg form.
    """
    cspec = SPEC["cond_instances"][cond.cond_cls]
    plug_fn = cond.plug_fn or cspec["plug_fn"]
    model_attr = cspec["model_attr"]
    target = f"{steer_var}.{cond.gate_buffer}"
    model_ref = f"{_INNER_MODEL}.{model_attr}"
    run_dest = cspec.get("run_dest", "model scope")
    install.to(_dest("runner_postload", model))
    install.add("")
    install.add(f"# token-gate: {cond.cond_cls} runs at model scope "
                f"({run_dest}); drives {target}")
    install.add(f"{model_ref}.{_plug_call(plug_fn, cond, cspec, target)}")


def _emit_model_run(cond, setup: _Sink, model: str) -> None:
    """Emit the model-scope detector's forward run-call (e.g. the EOL detector's
    ``if idx == 0 …`` invocation) as its own region, targeting the model file's
    ``run_anchor`` (inside Model.forward). Implicit: any model-scope cond whose
    class declares a ``run_call`` gets this. Template ``{attr}`` -> model_attr."""
    cspec = SPEC["cond_instances"][cond.cond_cls]
    run_call = cspec.get("run_call")
    if not run_call:
        return
    anchor = cspec["run_anchor"]
    attr = cspec["model_attr"]
    setup.to({
        "label": f"model file — {cond.cond_cls}.forward run-site (in the layer loop)",
        "anchor": anchor,
        "file": SPEC["models"][model]["anchors"][anchor],
    })
    setup.add(f"# run {cond.cond_cls} once per forward")
    setup.add(*(line.format(attr=attr) for line in run_call))


def _plug_call(plug_fn: str, cond, cspec: dict, target: str) -> str:
    """Render the buffer-wiring call: positional ``fn([t])`` when plug_kind is
    None, else the kwarg ``fn(kind=[t])`` form."""
    plug_kind = cond.plug_kind or cspec["plug_kind"]
    if plug_kind is None:
        return f"{plug_fn}([{target}])"
    return f"{plug_fn}({plug_kind}=[{target}])"


def _cond_flag(cond) -> str:
    if cond.instance_expr is not None:
        return SPEC["cond_instances"][cond.cond_cls]["flag"]
    return SPEC["cond_ops"][cond.probe_op]["flag"]


def _emit_cond_setup(i: int, stmt: Statement, steer_var: str,
                     setup: _Sink, assumed: set[str], model: str) -> str:
    cond = stmt.cond
    # Ordering: the condition must run before the steerer in the same forward.
    # (Skipped for all-layers steering, where there is no single min layer.)
    if stmt.layers and max(cond.layers) >= min(stmt.layers):
        raise XmixError(
            cond.loc,
            f"cond layer(s) {list(cond.layers)} must precede steered layer(s) "
            f"{list(stmt.layers)} so the mask is populated before use")

    if cond.instance_expr is not None:
        return _emit_cond_instance_setup(cond, steer_var, setup, model)

    cspec = SPEC["cond_ops"][cond.probe_op]
    probe_cls = _cls_name(cspec["cls_path"])
    assumed.add(probe_cls)
    assumed.update({cond.args[0], cond.args[1], _COND_BIAS, _MAX_TOKENS})
    # The bridge function name (compare op) is user-defined; assume in scope.
    assumed.add(cond.compare_op)

    probe_var = f"lp_{i}"
    cond_var = f"cond_{i}"
    weights, threshold = cond.args[0], cond.args[1]
    setup.to(_dest("runner_setup", model))
    setup.add(f"{probe_var} = {probe_cls}({weights}, {_COND_BIAS}, "
              f"{_MAX_TOKENS})")
    setup.add(f"{cond_var} = {cond.compare_op}({probe_var}, {threshold}, "
              f"sv_{i}.input_map)   # user-defined probs->mask bridge")
    return cond_var


def _emit_cond_instance_setup(cond, steer_var: str, setup: _Sink,
                              model: str) -> str:
    """Gates form: emit the plug wiring; return the condition read-hook expr.

    The plug function and sink-kind come from SPEC["cond_instances"] for the
    condition's class, unless overridden per-statement via .gates(.., via=, kind=).
    The condition instance and steerer are bare names built in the preamble.
    """
    cspec = SPEC["cond_instances"][cond.cond_cls]
    plug_fn = cond.plug_fn or cspec["plug_fn"]
    target = f"{steer_var}.{cond.gate_buffer}"
    setup.to(_dest("runner_setup", model))
    setup.add(
        f"{cond.instance_expr}.{_plug_call(plug_fn, cond, cspec, target)}"
        f"   # gate: {cond.cond_cls} output drives {target}")
    return f"{cond.instance_expr}.{cond.method}"


# ──────────────────────────────────────────────────────────────────────────
# op-specific SETUP emitters (keyed by OpSpec.build_kind)
# ──────────────────────────────────────────────────────────────────────────

def _emit_op_build(op_spec, cls_name: str, var: str, args: tuple[str, ...],
                   i: int) -> list[str]:
    if op_spec.build_kind is None:
        return [f"{var} = {cls_name}({', '.join(args)})"]
    return _OP_EMITTERS[op_spec.build_kind](cls_name, var, args, i)


def _emit_steering_vector_adder(cls_name: str, var: str,
                                args: tuple[str, ...], i: int) -> list[str]:
    """Fully-expanded raw-torch construction for a single steering vector."""
    src = args[0]
    r = f"_r{i}"
    mv = f"_max_vecs{i}"
    h = f"_hidden{i}"
    scales = f"_scales{i}"
    vi = f"_vec_indices{i}"
    npt = f"_n_vecs_per_token{i}"
    return [
        f"{r} = {src}.unsqueeze(0) if {src}.ndim == 1 else {src}",
        f"{mv}, {h} = {r}.shape",
        f"{scales} = torch.ones(({_MAX_TOKENS}, {mv}), dtype={r}.dtype, "
        f"device={r}.device)",
        f"{vi} = torch.arange({mv}, dtype=torch.int32, device={r}.device)"
        f".repeat({_MAX_TOKENS}, 1)",
        f"{npt} = torch.full(({_MAX_TOKENS},), {mv}, dtype=torch.int32, "
        f"device={r}.device)",
        f"{var} = {cls_name}({r}, {scales}, {vi}, {npt})",
    ]


_OP_EMITTERS = {
    "steering_vector_adder": _emit_steering_vector_adder,
}

# Per-op source expressions for the trailing args needed when a hook site
# passes *fewer* args than the op's method accepts.
_EXTRA_ARG_EXPRS = {
    "SteeringVector": lambda var: (f"{var}.r", _STEERED_TOKENS_NUM),
}


def _extra_arg_exprs(op_name: str, var: str) -> tuple[str, ...]:
    return _EXTRA_ARG_EXPRS.get(op_name, lambda v: ())(var)


# ──────────────────────────────────────────────────────────────────────────
# arity reconciliation (emits a callable *expression*)
# ──────────────────────────────────────────────────────────────────────────

def _reconcile_expr(method_expr: str, method_arity: int, site_arity: int,
                    extra: tuple[str, ...], assumed: set[str]) -> str:
    if method_arity == site_arity:
        return method_expr
    params = [f"_a{j}" for j in range(site_arity)]
    if method_arity > site_arity:
        need = method_arity - site_arity
        if len(extra) != need:
            raise XmixError(
                None,
                f"cannot adapt {method_expr}: site passes {site_arity} arg(s), "
                f"method needs {method_arity}; {need} extra expr(s) required "
                f"but {len(extra)} provided")
        for e in extra:
            if not e.startswith("_a") and "." not in e:
                assumed.add(e)
        call_args = params + list(extra)
        return f"lambda {', '.join(params)}: {method_expr}({', '.join(call_args)})"
    # method_arity < site_arity: drop the surplus site args.
    call_args = params[:method_arity]
    return f"lambda {', '.join(params)}: {method_expr}({', '.join(call_args)})"


# ──────────────────────────────────────────────────────────────────────────
# small helpers
# ──────────────────────────────────────────────────────────────────────────

def _cls_name(dotted: str) -> str:
    return dotted.rpartition(".")[2]


def _tuple_literal(xs: tuple[int, ...]) -> str:
    # always a tuple, even for a single element
    inner = ", ".join(str(x) for x in xs)
    return f"({inner},)" if len(xs) == 1 else f"({inner})"


def _method_arity(op_name: str, method: str) -> int:
    """Resolve the real positional arity of an op's installed method.

    Falls back to a known table when the class cannot be imported in a
    torch-free context; for v1 we import lazily and introspect.
    """
    import importlib
    import inspect

    spec = SPEC["ops"][op_name]
    module_path, _, name = spec.cls_path.rpartition(".")
    cls = getattr(importlib.import_module(module_path), name)
    fn = getattr(cls, method)
    sig = inspect.signature(fn)
    positional = (inspect.Parameter.POSITIONAL_ONLY,
                  inspect.Parameter.POSITIONAL_OR_KEYWORD)
    n = sum(1 for p in sig.parameters.values() if p.kind in positional)
    # unbound function includes 'self' — subtract it
    return n - 1


def _dsl_comment(stmt: Statement) -> list[str]:
    """Echo the original DSL statement as a comment (best-effort reconstruct)."""
    verb = "write" if stmt.op == OpKind.WRITE else "read"
    if stmt.instance_expr is not None:
        op_part = f"{stmt.instance_expr}.{stmt.method}"
    else:
        op_part = f"{stmt.op_name}({', '.join(stmt.args)}).{stmt.method}"
    layer_part = '.layer("all")' if stmt.all_layers \
        else f".layers({list(stmt.layers)})"
    head = (f"# {stmt.loc.file}:{stmt.loc.line}  m.{verb}({op_part})"
            f"{layer_part}.submodule(\"{stmt.submodule}\")")
    if stmt.cond is None:
        return [head]
    c = stmt.cond
    if c.instance_expr is not None:
        gate_args = f'"{c.gate_buffer}"'
        if c.plug_fn is not None:
            gate_args += f', via="{c.plug_fn}"'
        if c.plug_kind is not None:
            gate_args += f', kind="{c.plug_kind}"'
        cond_expr = f"{c.instance_expr}.gates({gate_args})"
    else:
        cond_expr = f"{c.probe_op}.{c.compare_op}({', '.join(c.args)})"
    if c.model_level:
        tail = '.layer("model")'
    else:
        tail = f".layer({list(c.layers)}).submodule(\"{c.submodule}\")"
    cont = f"#               .cond({cond_expr}{tail})"
    return [head, cont]
