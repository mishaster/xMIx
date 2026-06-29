# SPDX-License-Identifier: Apache-2.0
"""Phase A: static parse + validation of an .xmix file.

The file is parsed with the ``ast`` module and NEVER executed. Every name,
method, argument shape and submodule string is validated against ``SPEC``.
Failures raise :class:`XmixError` with a located (file:line:col) message.

Symbolic argument *existence* (whether ``arg1`` is in the REGISTRY) is NOT
checked here — that is deferred to Phase B. Here we only verify that args are
bare names rather than literals.
"""

from __future__ import annotations

import ast
import re
import warnings

from vllm.activations_extractor.xmix.errors import XmixError
from vllm.activations_extractor.xmix.ir import (
    CondClause,
    OpKind,
    ParseResult,
    PreambleSegment,
    SourceLoc,
    Statement,
)
from vllm.activations_extractor.xmix.spec import SPEC

_STMT_METHODS = ("layer", "layers", "submodule", "cond")
_COND_METHODS = ("layer", "submodule")


def parse_and_validate(path: str) -> ParseResult:
    """Parse and statically validate an .xmix file.

    Returns a :class:`ParseResult` holding the verbatim non-DSL preamble lines
    (raw Python the user hand-writes for object construction) and the validated
    DSL statements. The file is parsed with ``ast`` and never executed.
    """
    with open(path) as f:
        src = f.read()

    try:
        tree = ast.parse(src, filename=path)
    except SyntaxError as e:
        loc = SourceLoc(path, e.lineno or 1, (e.offset or 1) - 1)
        raise XmixError(loc, f"invalid Python syntax in xmix file: {e.msg}") from e

    # First pass: split top-level nodes into DSL statements vs verbatim preamble,
    # and record any ``<name> = <Class>(...)`` so variable-reference ops can
    # resolve their class for static .run/arity validation.
    class_map: dict[str, str] = {}
    stmt_nodes: list[ast.Call] = []
    pre_nodes: list[tuple[ast.AST, str, tuple[str, ...]]] = []
    for node in tree.body:
        if _is_m_statement(node):
            stmt_nodes.append(node.value)
            continue
        # Validate any `self.x = KnownClass(...)` construction against its SPEC
        # arg schema; the returned text has defaults injected (or is None ⇒ keep
        # the verbatim source). Raises a located XmixError on a bad construction.
        augmented = _validate_construction(path, node, src)
        segment = augmented if augmented is not None \
            else ast.get_source_segment(src, node)
        pre_nodes.append((node, segment if segment is not None else "",
                          _segment_targets(node)))
        _collect_assignment(node, class_map)

    # ``#token_detector: init`` … ``#token_detector: end`` blocks group supporting
    # init code with the model-scope construction it builds, so the whole block
    # shares a destination.
    ranges = _scan_init_blocks(src, path)
    range_cls = _resolve_init_blocks(path, ranges, pre_nodes, class_map)

    preamble = [
        PreambleSegment(text=text, targets=targets,
                        block_cls=_owner_cls(node.lineno, range_cls))
        for node, text, targets in pre_nodes
    ]
    statements = [_parse_statement(path, call, class_map) for call in stmt_nodes]
    model = _scan_model(src, path)
    return ParseResult(tuple(preamble), tuple(statements), model)


_MODEL_RE = re.compile(r"^#\s*model\s*:\s*(\S+)")


def _scan_model(src: str, path: str) -> str:
    """Read the optional ``#model: <name>`` pragma (raw-line scan, since ast drops
    comments). Validates against SPEC["models"]; defaults to SPEC["default_model"].
    """
    found: str | None = None
    for lineno, raw in enumerate(src.splitlines(), start=1):
        m = _MODEL_RE.match(raw.strip())
        if not m:
            continue
        name = m.group(1)
        if name not in SPEC["models"]:
            raise XmixError(
                SourceLoc(path, lineno, 0), f"unknown model '{name}'",
                alternatives=SPEC["models"].keys())
        if found is not None and found != name:
            raise XmixError(SourceLoc(path, lineno, 0),
                            f"conflicting #model: pragmas ('{found}' vs '{name}')")
        found = name
    return found or SPEC["default_model"]


_INIT_RE = re.compile(r"^#\s*token_detector\s*:\s*init\b")
_END_RE = re.compile(r"^#\s*token_detector\s*:\s*end\b")


def _scan_init_blocks(src: str, path: str) -> list[tuple[int, int]]:
    """Find ``#token_detector: init`` … ``#token_detector: end`` comment markers
    (raw-line scan, since ast drops comments). Returns inclusive
    ``(begin_line, end_line)`` ranges."""
    ranges: list[tuple[int, int]] = []
    open_line: int | None = None
    for lineno, raw in enumerate(src.splitlines(), start=1):
        s = raw.strip()
        if _INIT_RE.match(s):
            if open_line is not None:
                raise XmixError(
                    SourceLoc(path, lineno, 0),
                    "#token_detector: init nested inside another init block")
            open_line = lineno
        elif _END_RE.match(s):
            if open_line is None:
                raise XmixError(
                    SourceLoc(path, lineno, 0),
                    "#token_detector: end without a matching #token_detector: init")
            ranges.append((open_line, lineno))
            open_line = None
    if open_line is not None:
        raise XmixError(
            SourceLoc(path, open_line, 0),
            "#token_detector: init block not closed with #token_detector: end")
    return ranges


def _resolve_init_blocks(
    path: str, ranges: list[tuple[int, int]],
    pre_nodes: list[tuple[ast.AST, str, tuple[str, ...]]],
    class_map: dict[str, str],
) -> list[tuple[int, int, str]]:
    """Infer the model-scope class each init block constructs (Q2: inferred, not
    named). Each block must build exactly one model-scope instance."""
    resolved: list[tuple[int, int, str]] = []
    for begin, end in ranges:
        owners: list[str] = []
        for node, _text, targets in pre_nodes:
            if begin < node.lineno < end:
                for t in targets:
                    cls = class_map.get(t)
                    if cls and SPEC["cond_instances"].get(cls, {}).get(
                            "scope") == "model":
                        owners.append(cls)
        if not owners:
            raise XmixError(
                SourceLoc(path, begin, 0),
                "#token_detector: init block must construct a model-scope "
                "instance (e.g. EOLTokenDetector)")
        if len(owners) > 1:
            raise XmixError(
                SourceLoc(path, begin, 0),
                f"#token_detector: init block builds multiple model-scope "
                f"instances {owners}; use one block each")
        resolved.append((begin, end, owners[0]))
    return resolved


def _owner_cls(lineno: int,
               range_cls: list[tuple[int, int, str]]) -> str | None:
    for begin, end, cls in range_cls:
        if begin < lineno < end:
            return cls
    return None


# ──────────────────────────────────────────────────────────────────────────
# preamble constructor-arg validation + default injection
# ──────────────────────────────────────────────────────────────────────────

_NOT_SPEC = object()  # sentinel: the class is not known to xmix


def _arg_schema(cls_name: str):
    """Return the class's arg schema, ``None`` (SPEC-known but no schema), or
    ``_NOT_SPEC`` (not an xmix class at all)."""
    op = SPEC["ops"].get(cls_name)
    if op is not None:
        return op.args
    inst = SPEC["cond_instances"].get(cls_name)
    if inst is not None:
        return inst.get("args")
    return _NOT_SPEC


def _validate_construction(path: str, node: ast.AST, src: str) -> str | None:
    """Validate a ``<target> = KnownClass(...)`` preamble construction against the
    class's SPEC arg schema. Required args must be present; unknown args are an
    error; omitted optional args get their default injected. Returns the rewritten
    construction text when defaults were injected, else None (keep verbatim).
    Raises a located :class:`XmixError` for any violation."""
    if not (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)):
        return None
    cls_name = node.value.func.id
    schema = _arg_schema(cls_name)
    if schema is _NOT_SPEC:
        return None  # not an xmix class — leave the line verbatim
    call = node.value
    if schema is None:
        raise XmixError(
            _loc(path, call),
            f"'{cls_name}' is a known xmix class but has no args schema in "
            f"SPEC; add one so its construction can be validated")

    keys = list(schema.keys())
    provided: dict[str, str] = {}
    if len(call.args) > len(keys):
        raise XmixError(_loc(path, call),
                        f"{cls_name} takes at most {len(keys)} positional args")
    for i, arg in enumerate(call.args):
        provided[keys[i]] = ast.get_source_segment(src, arg) or ""
    for kw in call.keywords:
        if kw.arg is None:
            raise XmixError(_loc(path, call),
                            f"{cls_name}: **kwargs is not supported")
        if kw.arg not in schema:
            raise XmixError(_loc(path, kw.value),
                            f"unknown argument '{kw.arg}' for {cls_name}",
                            alternatives=keys)
        if kw.arg in provided:
            raise XmixError(_loc(path, kw.value),
                            f"{cls_name}: argument '{kw.arg}' given twice")
        provided[kw.arg] = ast.get_source_segment(src, kw.value) or ""

    missing = [k for k, d in schema.items() if d is None and k not in provided]
    if missing:
        raise XmixError(
            _loc(path, call),
            f"{cls_name}: missing required argument(s): {', '.join(missing)}")

    if all(k in provided for k in keys):
        return None  # nothing to inject — preserve the user's verbatim text

    target = ast.get_source_segment(src, node.targets[0]) \
        if len(node.targets) == 1 else None
    if target is None:
        raise XmixError(
            _loc(path, node),
            f"{cls_name}: cannot inject defaults into a multi-target assignment")
    # Canonical keyword form, in schema order: provided as written, else default.
    parts = [f"{k}={provided[k] if k in provided else schema[k]}" for k in keys]
    return f"{target} = {cls_name}({', '.join(parts)})"


def _is_m_statement(node: ast.AST) -> bool:
    """True for a top-level ``m.<op>(...)...`` chain (vs raw-Python preamble)."""
    if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
        return False
    base, _ = _flatten_chain(node.value)
    return isinstance(base, ast.Name) and base.id == "m"


def _collect_assignment(node: ast.AST, class_map: dict[str, str]) -> None:
    """Record ``<dotted target> = <ClassName>(...)`` assignments by class name."""
    if not (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)):
        return
    cls = node.value.func.id
    for target in node.targets:
        name = _dotted_name(target)
        if name is not None:
            class_map[name] = cls


def _segment_targets(node: ast.AST) -> tuple[str, ...]:
    """Dotted assignment targets of a top-level node (empty for non-Assign)."""
    if not isinstance(node, ast.Assign):
        return ()
    names = [_dotted_name(t) for t in node.targets]
    return tuple(n for n in names if n is not None)


def _dotted_name(node: ast.AST) -> str | None:
    """Reconstruct a pure dotted name (``self.steer_refusal_vec``) or None."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted_name(node.value)
        return None if base is None else f"{base}.{node.attr}"
    return None


# ──────────────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────────────

def _loc(path: str, node: ast.AST) -> SourceLoc:
    return SourceLoc(path, getattr(node, "lineno", 1), getattr(node, "col_offset", 0))


def _flatten_chain(node: ast.AST) -> tuple[ast.AST, list[tuple[str, ast.AST]]]:
    """Flatten ``base.a(...).b[..].c(...)`` into (base_node, [(a, node), ...]).

    Each link node is a ``Call`` (``.a(...)``) or a ``Subscript`` (``.a[..]`` —
    used by the layer-range slice form ``.layer[15:23]``). Links are returned in
    *source order* (left to right).
    """
    links: list[tuple[str, ast.AST]] = []
    cur: ast.AST = node
    while True:
        if isinstance(cur, ast.Call) and isinstance(cur.func, ast.Attribute):
            links.append((cur.func.attr, cur))
            cur = cur.func.value
        elif isinstance(cur, ast.Subscript) and isinstance(cur.value, ast.Attribute):
            links.append((cur.value.attr, cur))
            cur = cur.value.value
        else:
            break
    links.reverse()
    return cur, links


def _no_keywords(path: str, call: ast.AST, what: str) -> None:
    if not isinstance(call, ast.Call):
        raise XmixError(_loc(path, call), f"{what} must be a (...) call")
    if call.keywords:
        raise XmixError(_loc(path, call), f"{what} does not accept keyword arguments")


def _expect_name_args(path: str, call: ast.Call, what: str) -> tuple[str, ...]:
    """Each positional arg must be a bare Name (a registry symbol)."""
    _no_keywords(path, call, what)
    names: list[str] = []
    for arg in call.args:
        if not isinstance(arg, ast.Name):
            raise XmixError(
                _loc(path, arg),
                f"{what} argument must be a registry symbol name, not a literal",
            )
        names.append(arg.id)
    return tuple(names)


def _expect_int_list(path: str, call: ast.Call, what: str) -> tuple[int, ...]:
    _no_keywords(path, call, what)
    if len(call.args) != 1 or not isinstance(call.args[0], ast.List):
        raise XmixError(_loc(path, call), f"{what} expects a single list of ints")
    items: list[int] = []
    for el in call.args[0].elts:
        if not (isinstance(el, ast.Constant) and isinstance(el.value, int)
                and not isinstance(el.value, bool)):
            raise XmixError(_loc(path, el), f"{what} list elements must be ints")
        items.append(el.value)
    if not items:
        raise XmixError(_loc(path, call), f"{what} list must not be empty")
    return tuple(items)


def _expect_layers(path: str, link: ast.AST, what: str) -> tuple[tuple[int, ...], bool]:
    """Parse a layer spec. Three forms, returning ``(layers, all_layers)``:

      * ``.layer[15:23]``  — inclusive Python slice ⇒ (15, 16, ..., 23)
      * ``.layer([1, 2])`` — explicit int list
      * ``.layer("all")``  — every layer ⇒ ((), True)
    """
    # Slice form: .layer[lo:hi] (a Subscript link).
    if isinstance(link, ast.Subscript):
        return _layers_from_slice(path, link, what), False

    call = link
    _no_keywords(path, call, what)
    if len(call.args) != 1:
        raise XmixError(_loc(path, call),
                        f'{what} expects [lo:hi], a list of ints, or "all"')
    arg = call.args[0]
    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
        if arg.value != "all":
            raise XmixError(_loc(path, arg), f'{what} string must be "all"',
                            alternatives=("all",))
        return (), True
    return _expect_int_list(path, call, what), False


def _int_const(node: ast.AST) -> int | None:
    """Return the int value of an int-literal node (rejecting bool), else None."""
    if isinstance(node, ast.Constant) and isinstance(node.value, int) \
            and not isinstance(node.value, bool):
        return node.value
    return None


def _layers_from_slice(path: str, sub: ast.Subscript, what: str) -> tuple[int, ...]:
    """Expand an inclusive ``[lo:hi]`` / ``[lo:hi:step]`` slice into a layer tuple."""
    sl = sub.slice
    if not isinstance(sl, ast.Slice) or sl.lower is None or sl.upper is None:
        raise XmixError(_loc(path, sub),
                        f"{what} range must be [lo:hi] with both bounds, e.g. [15:23]")
    lo, hi = _int_const(sl.lower), _int_const(sl.upper)
    if lo is None or hi is None:
        raise XmixError(_loc(path, sub), f"{what} range bounds must be int literals")
    step = 1
    if sl.step is not None:
        step = _int_const(sl.step)
        if step is None or step <= 0:
            raise XmixError(_loc(path, sub), f"{what} range step must be a positive int")
    if hi < lo:
        raise XmixError(_loc(path, sub),
                        f"{what} range upper ({hi}) must be >= lower ({lo})")
    # Inclusive of the upper bound: [15:23] -> 15..23.
    return tuple(range(lo, hi + 1, step))


def _expect_str(path: str, call: ast.Call, what: str) -> str:
    _no_keywords(path, call, what)
    if (len(call.args) != 1 or not isinstance(call.args[0], ast.Constant)
            or not isinstance(call.args[0].value, str)):
        raise XmixError(_loc(path, call), f"{what} expects a single string literal")
    return call.args[0].value


def _validate_submodule(path: str, call: ast.Call) -> str:
    name = _expect_str(path, call, ".submodule(...)")
    if name not in SPEC["submodules"]:
        raise XmixError(
            _loc(path, call), f"unknown submodule '{name}'",
            alternatives=SPEC["submodules"].keys(),
        )
    return name


# ──────────────────────────────────────────────────────────────────────────
# statement parsing
# ──────────────────────────────────────────────────────────────────────────

def _parse_statement(path: str, call: ast.Call,
                     class_map: dict[str, str]) -> Statement:
    base, links = _flatten_chain(call)
    if not (isinstance(base, ast.Name) and base.id == "m"):
        raise XmixError(_loc(path, call), "statement must start with 'm'")
    if not links:
        raise XmixError(_loc(path, call), "empty statement")

    # First link is the head op (write/read).
    verb, op_call = links[0]
    op_kind = _resolve_verb(path, op_call, verb)
    op_name, op_args, op_method, instance_expr = _parse_op_arg(
        path, op_call, class_map)

    layers: tuple[int, ...] = ()
    all_layers = False
    seen_layers = False
    submodule: str | None = None
    cond: CondClause | None = None

    for attr, link_call in links[1:]:
        if attr in ("layer", "layers"):
            layers, all_layers = _expect_layers(path, link_call, f".{attr}(...)")
            seen_layers = True
        elif attr == "submodule":
            submodule = _validate_submodule(path, link_call)
        elif attr == "cond":
            cond = _parse_cond(path, link_call, class_map)
        else:
            raise XmixError(
                _loc(path, link_call), f"unknown method '.{attr}'",
                alternatives=_STMT_METHODS,
            )

    if not seen_layers:
        raise XmixError(_loc(path, call), "statement missing .layers([...])")
    if submodule is None:
        raise XmixError(_loc(path, call), "statement missing .submodule(\"...\")")

    return Statement(
        op=op_kind, op_name=op_name, args=op_args, method=op_method,
        layers=layers, submodule=submodule, cond=cond, loc=_loc(path, call),
        instance_expr=instance_expr, all_layers=all_layers,
    )


def _resolve_verb(path: str, op_call: ast.Call, verb: str) -> OpKind:
    if verb == "write":
        return OpKind.WRITE
    if verb == "read":
        return OpKind.READ
    raise XmixError(
        _loc(path, op_call), f"unknown op '.{verb}'", alternatives=("write", "read"),
    )


def _parse_op_arg(
    path: str, op_call: ast.Call, class_map: dict[str, str],
) -> tuple[str, tuple[str, ...], str, str | None]:
    """Parse the single argument of ``m.write(...)``.

    Two accepted forms, returning ``(op_name, args, method, instance_expr)``:
      * ``<Class>(args).<method>`` — synthesized op; instance_expr is None.
      * ``<var>.<method>`` / ``<self.var>.<method>`` — variable reference to a
        preamble-built instance; instance_expr is the dotted name, args is ().
    """
    _no_keywords(path, op_call, ".write(...)")
    if len(op_call.args) != 1:
        raise XmixError(_loc(path, op_call), ".write(...) expects exactly one argument")
    arg = op_call.args[0]

    if not isinstance(arg, ast.Attribute):
        raise XmixError(
            _loc(path, arg),
            "op argument must be <Class>(args).<method> or <var>.<method>",
        )
    method = arg.attr

    # Synthesized form: <Class>(args).<method>
    if isinstance(arg.value, ast.Call) and isinstance(arg.value.func, ast.Name):
        cls_name = arg.value.func.id
        spec = _lookup_op(path, arg, cls_name)
        args = _expect_name_args(path, arg.value, cls_name)
        if len(args) != spec.arity:
            raise XmixError(
                _loc(path, arg.value),
                f"{cls_name} expects {spec.arity} arg(s), got {len(args)}",
            )
        _check_method(path, arg, cls_name, method, spec.method)
        return cls_name, args, method, None

    # Variable-reference form: <var>.<method> — resolve class from the preamble.
    instance_expr = _dotted_name(arg.value)
    if instance_expr is None:
        raise XmixError(
            _loc(path, arg),
            "op argument must be <Class>(args).<method> or <var>.<method>",
        )
    cls_name = class_map.get(instance_expr)
    if cls_name is None:
        raise XmixError(
            _loc(path, arg),
            f"cannot resolve class of '{instance_expr}'; assign it in the "
            f"preamble as '{instance_expr} = <Class>(...)'",
        )
    spec = _lookup_op(path, arg, cls_name)
    _check_method(path, arg, cls_name, method, spec.method)
    return cls_name, (), method, instance_expr


def _lookup_op(path: str, node: ast.AST, cls_name: str):
    spec = SPEC["ops"].get(cls_name)
    if spec is None:
        raise XmixError(
            _loc(path, node), f"unknown op '{cls_name}'",
            alternatives=SPEC["ops"].keys(),
        )
    return spec


def _check_method(path: str, node: ast.AST, cls_name: str,
                  method: str, expected: str) -> None:
    if method != expected:
        raise XmixError(
            _loc(path, node), f"unknown method '.{method}' on {cls_name}",
            alternatives=(expected,),
        )


def _parse_cond(path: str, cond_call: ast.Call,
                class_map: dict[str, str]) -> CondClause:
    _no_keywords(path, cond_call, ".cond(...)")
    if len(cond_call.args) != 1 or not isinstance(cond_call.args[0], ast.Call):
        raise XmixError(_loc(path, cond_call), ".cond(...) expects one chained expression")

    base, links = _flatten_chain(cond_call.args[0])
    if not links:
        raise XmixError(_loc(path, cond_call), "empty .cond expression")

    # Dispatch on the base: a probe *class* (Name in cond_ops) → synthesized
    # probe form; a prebuilt *instance* (preamble var) → gates form.
    if isinstance(base, ast.Name) and base.id in SPEC["cond_ops"]:
        return _parse_cond_probe(path, cond_call, base, links)

    instance_expr = _dotted_name(base)
    if instance_expr is not None and instance_expr in class_map:
        return _parse_cond_instance(path, cond_call, instance_expr, links,
                                    class_map[instance_expr])

    # Neither: produce the most helpful located error we can.
    if isinstance(base, ast.Name):
        raise XmixError(
            _loc(path, base), f"unknown cond op '{base.id}'",
            alternatives=SPEC["cond_ops"].keys(),
        )
    raise XmixError(
        _loc(path, cond_call),
        f"cond instance '{instance_expr}' is not built in the preamble; "
        f"assign it as '{instance_expr} = <Class>(...)'"
        if instance_expr is not None else
        ".cond expression must start with a probe class or a prebuilt instance",
    )


def _parse_cond_tail(
    path: str, cond_call: ast.Call, links: list[tuple[str, ast.Call]],
) -> tuple[tuple[int, ...], str]:
    """Parse the shared ``.layer([..]).submodule("..")`` tail of a cond expr."""
    layers: tuple[int, ...] | None = None
    submodule: str | None = None
    for attr, link_call in links:
        if attr == "layer":
            layers, all_layers = _expect_layers(path, link_call, ".layer(...)")
            if all_layers:
                raise XmixError(_loc(path, link_call),
                                'cond .layer does not support "all"')
        elif attr == "submodule":
            submodule = _validate_submodule(path, link_call)
        else:
            raise XmixError(
                _loc(path, link_call), f"unknown cond method '.{attr}'",
                alternatives=_COND_METHODS,
            )
    if layers is None:
        raise XmixError(_loc(path, cond_call), ".cond missing .layer([...])")
    if submodule is None:
        raise XmixError(_loc(path, cond_call), ".cond missing .submodule(\"...\")")
    return layers, submodule


def _parse_cond_probe(path: str, cond_call: ast.Call, base: ast.Name,
                      links: list[tuple[str, ast.Call]]) -> CondClause:
    """Probe-class form: ``LinearProbe.biggerThan(a, b).layer([..]).submodule("..")``."""
    probe_op = base.id
    cspec = SPEC["cond_ops"][probe_op]

    # First link is the compare op (e.g. biggerThan).
    compare_op, compare_call = links[0]
    compares = cspec["compares"]
    compare_spec = compares.get(compare_op)
    if compare_spec is None:
        raise XmixError(
            _loc(path, compare_call), f"unknown compare op '.{compare_op}' on {probe_op}",
            alternatives=compares.keys(),
        )
    args = _expect_name_args(path, compare_call, f"{probe_op}.{compare_op}")
    if len(args) != compare_spec.arity:
        raise XmixError(
            _loc(path, compare_call),
            f"{probe_op}.{compare_op} expects {compare_spec.arity} arg(s), got {len(args)}",
        )
    if compare_spec.builder_path is None:
        warnings.warn(
            f"xmix: compare op '{probe_op}.{compare_op}' has no bridge builder "
            f"registered in SPEC; conditional mask wiring will be skipped at "
            f"install time (probe runs read-only). Define a factory and set its "
            f"dotted path in SPEC to enable conditional steering.",
            stacklevel=2,
        )

    layers, submodule = _parse_cond_tail(path, cond_call, links[1:])
    return CondClause(
        layers=layers, submodule=submodule, loc=_loc(path, cond_call),
        probe_op=probe_op, compare_op=compare_op, args=args,
    )


def _parse_cond_instance(path: str, cond_call: ast.Call, instance_expr: str,
                         links: list[tuple[str, ast.Call]],
                         cond_cls: str) -> CondClause:
    """Gates form: ``<var>.gates("buffer"[, via=, kind=]).layer([..]).submodule("..")``."""
    cspec = SPEC["cond_instances"].get(cond_cls)
    if cspec is None:
        raise XmixError(
            _loc(path, cond_call),
            f"'{instance_expr}' is a {cond_cls}, which is not a registered cond "
            f"instance", alternatives=SPEC["cond_instances"].keys(),
        )

    # First link must be the gating verb.
    verb, gate_call = links[0]
    if verb != "gates":
        raise XmixError(
            _loc(path, gate_call),
            f"unknown cond method '.{verb}' on instance '{instance_expr}'",
            alternatives=("gates",),
        )
    # .gates("buffer", via="...", kind="...") — one string positional, two
    # optional string kwargs overriding the SPEC plug default.
    if len(gate_call.args) != 1 or not _is_str_const(gate_call.args[0]):
        raise XmixError(_loc(path, gate_call),
                        '.gates(...) expects a single buffer-name string')
    gate_buffer = gate_call.args[0].value
    plug_fn, plug_kind = None, None
    for kw in gate_call.keywords:
        if kw.arg == "via" and _is_str_const(kw.value):
            plug_fn = kw.value.value
        elif kw.arg == "kind" and _is_str_const(kw.value):
            plug_kind = kw.value.value
        else:
            raise XmixError(
                _loc(path, gate_call),
                f".gates(...) accepts only via=/kind= string kwargs, got "
                f"'{kw.arg}'")

    layers, submodule, model_level = _parse_cond_instance_tail(
        path, cond_call, links[1:])

    # Scope must match the class: a model-scope class (e.g. EOLTokenDetector)
    # requires .layer("model"); a per-layer class rejects it.
    scope = cspec.get("scope", "layer")
    if model_level and scope != "model":
        raise XmixError(
            _loc(path, cond_call),
            f"{cond_cls} is a per-layer condition; .layer(\"model\") is only "
            f"for model-scope conditions")
    if not model_level and scope == "model":
        raise XmixError(
            _loc(path, cond_call),
            f"{cond_cls} runs at model scope; use .layer(\"model\") "
            f"(no .submodule)")

    return CondClause(
        layers=layers, submodule=submodule, loc=_loc(path, cond_call),
        instance_expr=instance_expr, cond_cls=cond_cls,
        method=cspec["method"], gate_buffer=gate_buffer,
        plug_fn=plug_fn, plug_kind=plug_kind, model_level=model_level,
    )


def _parse_cond_instance_tail(
    path: str, cond_call: ast.Call, links: list[tuple[str, ast.Call]],
) -> tuple[tuple[int, ...], str, bool]:
    """Parse the tail of a gates-form cond, allowing the model-scope sentinel.

    Two shapes, returning ``(layers, submodule, model_level)``:
      * ``.layer("model")``                — model scope; no ``.submodule`` allowed.
      * ``.layer([..]).submodule("..")``   — per-layer; both required (as today).
    """
    layers: tuple[int, ...] | None = None
    submodule: str | None = None
    model_level = False
    for attr, link_call in links:
        if attr == "layer":
            if _is_model_sentinel(link_call):
                model_level = True
                layers = ()
                continue
            layers, all_layers = _expect_layers(path, link_call, ".layer(...)")
            if all_layers:
                raise XmixError(_loc(path, link_call),
                                'cond .layer does not support "all"')
        elif attr == "submodule":
            submodule = _validate_submodule(path, link_call)
        else:
            raise XmixError(
                _loc(path, link_call), f"unknown cond method '.{attr}'",
                alternatives=_COND_METHODS,
            )

    if model_level:
        if submodule is not None:
            raise XmixError(
                _loc(path, cond_call),
                '.layer("model") (model scope) does not take a .submodule(...)')
        return (), "", True

    if layers is None:
        raise XmixError(_loc(path, cond_call), ".cond missing .layer([...])")
    if submodule is None:
        raise XmixError(_loc(path, cond_call), ".cond missing .submodule(\"...\")")
    return layers, submodule, False


def _is_model_sentinel(link: ast.AST) -> bool:
    """True for the ``.layer("model")`` model-scope sentinel."""
    return (isinstance(link, ast.Call) and not link.keywords
            and len(link.args) == 1 and _is_str_const(link.args[0])
            and link.args[0].value == "model")


def _is_str_const(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and isinstance(node.value, str)
