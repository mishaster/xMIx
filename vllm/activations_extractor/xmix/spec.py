# SPDX-License-Identifier: Apache-2.0
"""Data-driven SPEC for the xmix DSL.

This is the single extension point of the DSL: to support a new steering class,
submodule hook-point, comparison op, or model family, edit only this file. The
parser (Phase A) and lowering (Phase B) read everything they need from ``SPEC``.

Class/builder references are stored as *dotted import paths* (strings) so this
module stays torch-free and importable for CPU-only static validation; Phase B
resolves them lazily.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OpSpec:
    """A head op usable as ``m.<verb>(<Op>(args).<method>)``."""

    cls_path: str        # dotted path to the real class instantiated in Phase B
    flag: str            # hook flag: "w" or "r"
    arity: int           # number of positional symbolic args in the DSL
    method: str          # trailing chained attr that yields the installed callable
    build_kind: str | None = None  # key into lower.BUILDERS; None -> cls(*args)
    # Constructor arg schema for the variable-ref form (hand-built in the .xmix
    # preamble): ordered {name -> default}. None default ⇒ required (compile
    # error if omitted); a string ⇒ optional, injected verbatim as name=<expr>.
    # Authoritative over the real signature (may add defaults the class lacks).
    # Synthesized ops (build_kind set) are not preamble-constructed ⇒ leave None.
    args: dict[str, str | None] | None = None


@dataclass(frozen=True)
class CompareSpec:
    """A comparison op usable inside ``.cond(<Probe>.<compare>(args)...)``.

    ``builder_path`` points to a *user-defined* factory with the contract:

        factory(probe, threshold, target_buffer) -> Callable[[Tensor], Any]

    i.e. given the constructed probe, a threshold, and the steerer's
    ``input_map`` buffer, it returns the read-hook callable that runs the probe
    and writes the 0/1 mask into ``input_map``. The framework only constructs
    and installs this callable — it does NOT synthesize the probability->mask
    bridge itself. ``builder_path`` is None until such a factory is provided.
    """

    builder_path: str | None  # dotted path to the user factory, or None
    ctor_kind: str            # describes how the builder is invoked
    arity: int                # symbolic-arg count of the compare call


SPEC = {
    # ── head ops: m.write(<Op>(args).run) ──────────────────────────────────
    "ops": {
        "SteeringVector": OpSpec(
            cls_path="vllm.activations_extractor.write_activations.SteeringVectorAdder",
            flag="w",
            arity=1,
            method="run",
            build_kind="steering_vector_adder",
        ),
        # Hand-constructed op: the user builds the instance in the .xmix preamble
        # and references it by variable (m.write(self.steer_refusal_vec.run)).
        # build_kind=None ⇒ no synthesis; arity=0 ⇒ no DSL construction args.
        # cls_path/method still drive static .run validation and arity reconciling.
        "SteeringVectorDotSubtractNormalized": OpSpec(
            cls_path="vllm.activations_extractor.write_activations.SteeringVectorDotSubtractNormalized",
            flag="w",
            arity=0,
            method="run",
            build_kind=None,
            # SteeringVectorDotSubtractNormalized(hidden_size, max_tokens,
            #     steering_vector, vec_indices, dtype, device)
            args={
                "hidden_size": "self.model_config.hidden_size",
                "max_tokens": "2048",
                "steering_vector": None,
                "vec_indices": None,
                "dtype": "self.model_config.dtype",
                "device": "self.device",
            },
        ),
        # Hand-constructed variable-reference form of SteeringVectorAdder. The
        # synthesized form is keyed "SteeringVector" above; this entry is keyed by
        # the real class name so the variable-ref path (class resolved from the
        # preamble assignment) can look it up. build_kind=None ⇒ no synthesis.
        "SteeringVectorAdder": OpSpec(
            cls_path="vllm.activations_extractor.write_activations.SteeringVectorAdder",
            flag="w",
            arity=0,
            method="run",
            build_kind=None,
            # SteeringVectorAdder(r, scales, vec_indices, n_vecs_per_token)
            args={
                "r": None,
                "scales": None,
                "vec_indices": None,
                "n_vecs_per_token": None,
            },
        ),
        # More hand-constructed variable-reference ops (built in the .xmix
        # preamble, referenced by variable). All install bare at their site:
        # the three write ops' run(x, r, steered_tokens_num) is 3-arg = mlp.post
        # write site; LinearProbe.run(x) is 1-arg = mlp.post read site.
        "SteeringVectorScaledAdder": OpSpec(
            cls_path="vllm.activations_extractor.write_activations.SteeringVectorScaledAdder",
            flag="w",
            arity=0,
            method="run",
            build_kind=None,
            # SteeringVectorScaledAdder(r, coeff, max_tokens)
            args={"r": None, "coeff": None, "max_tokens": "2048"},
        ),
        "SteeringLinear": OpSpec(
            cls_path="vllm.activations_extractor.write_activations.SteeringLinear",
            flag="w",
            arity=0,
            method="run",
            build_kind=None,
            # SteeringLinear(W, b, vec_indices, max_tokens)
            args={"W": None, "b": None, "vec_indices": None,
                  "max_tokens": "2048"},
        ),
        # Read op used via m.read(self.truth_probe.run). Also present in
        # cond_ops below for the .cond(LinearProbe.<compare>...) form — separate
        # parser entry points, no conflict.
        "LinearProbe": OpSpec(
            cls_path="vllm.activations_extractor.read_activations.LinearProbe",
            flag="r",
            arity=0,
            method="run",
            build_kind=None,
            # LinearProbe(weights, bias, max_tokens)
            args={"weights": None, "bias": None, "max_tokens": "2048"},
        ),
    },

    # ── cond ops: .cond(<Probe>.<compare>(args).layer([..]).submodule("..")) ─
    "cond_ops": {
        "LinearProbe": {
            "cls_path": "vllm.activations_extractor.read_activations.LinearProbe",
            "flag": "r",
            "arity": 2,
            "method": "run",
            "compares": {
                # builder_path None: bridge not provided yet (see CompareSpec).
                "biggerThan": CompareSpec(
                    builder_path=None,
                    ctor_kind="probe_threshold_mask",
                    arity=2,
                ),
            },
        },
    },

    # ── cond instances: .cond(<prebuilt-var>.gates("buffer").layer(..).submodule(..)) ─
    # The condition evaluator (e.g. CondProjCosSim) is hand-built in the .xmix
    # preamble with its threshold baked in. The DSL only declares the gating
    # *relationship*: which steerer buffer the condition output drives, and
    # where the condition reads. The imperative plug mechanics live here as a
    # per-class default (overridable per-statement via .gates(.., via=, kind=)):
    #   <instance>.<plug_fn>(<plug_kind>=[<steerer>.<buffer>])
    # and the condition's read hook (<instance>.<method>) installs at .layer/.submodule.
    "cond_instances": {
        "CondProjCosSim": {
            "cls_path": "vllm.activations_extractor.cond_kernels.CondProjCosSim",
            "flag": "r",
            "method": "run",        # installed as the read hook
            "plug_fn": "set_output_buffers",  # default wiring function
            "plug_kind": "out_bool",          # default kwarg / sink-kind
            "scope": "layer",       # installed as a per-decoder-layer read hook
            # CondProjCosSim(condition_projector, threshold, max_tokens)
            "args": {"condition_projector": None, "threshold": None,
                     "max_tokens": "2048"},
        },
        # Model-scope condition: the EOL token detector runs once per forward at
        # LlamaModel.forward (gated idx==0), NOT as a per-layer hook. The DSL
        # writes .layer("model") to mark it; the steerer-gating relationship is
        # the per-token EOL mask fanned into the steerer's buffer. Its
        # set_output_buffers takes a POSITIONAL list (plug_kind=None), unlike the
        # kwarg form above. model_attr is the attribute on the inner model object
        # (model.model.<attr>) that the forward already invokes.
        "EOLTokenDetector": {
            "cls_path": "vllm.activations_extractor.write_activations.EOLTokenDetector",
            "flag": "r",
            "method": "run",
            "plug_fn": "set_output_buffers",
            "plug_kind": None,      # None ⇒ positional list: set_output_buffers([target])
            "scope": "model",       # runs at LlamaModel.forward (idx==0), not per-layer
            "model_attr": "token_detector",  # model.model.<attr> the forward invokes
            # EOLTokenDetector(eol_token_ids, vocab_size, max_tokens, device)
            "args": {"eol_token_ids": None, "vocab_size": "self.vocab_size",
                     "max_tokens": "2048", "device": "self.device"},
            # Placement markers: dest/run_dest are human banner text (echoed
            # verbatim into comments, never parsed); dest_anchor is the apply
            # landmark KEY whose concrete file is resolved per-model from
            # SPEC["models"][<model>]["anchors"] (→ llama.py / qwen3.py /
            # mixtral.py depending on the .xmix `#model:` pragma).
            "dest": "model file — model construction (e.g. init_token_detector)",
            "dest_anchor": "model_construct",
            "run_dest": "model file — Model.forward (injected at the model_forward anchor)",
            # Forward run-site: the call injected inside Model.forward's layer
            # loop. run_call lines are templates ({attr} -> model_attr); they are
            # emitted at the run_anchor (resolved per-model from the models table).
            "run_anchor": "model_forward",
            "run_call": (
                "if idx == 0 and self.{attr} is not None:",
                "    self.{attr}(input_ids)",
            ),
        },
    },

    # ── destination markers: where each emitted region belongs in real code ─
    # The codegen groups the synthesized text by these labels (see codegen._Sink)
    # so the reader knows which excerpt to plant where. Per-class model-scope
    # destinations live on the cond_instances entry above ("dest"/"run_dest").
    "destinations": {
        "runner_setup": "gpu_model_runner — constructed once (e.g. in __init__/setup)",
        "runner_postload": "gpu_model_runner — after load_model (install hooks + wire buffers)",
    },

    # ── submodule string -> layer setter + per-direction hook call arity ────
    # write_arity / read_arity are how many positional args the layer forward
    # passes to the installed hook at that point (drives signature reconciling).
    "submodules": {
        "attention.post": {
            "setter": "set_post_attn_pre_norm_hook",
            "write_arity": 3,
            "read_arity": 1,
        },
        "mlp.post": {
            "setter": "set_post_mlp_hook",
            "write_arity": 3,
            "read_arity": 1,
        },
        "norm.pre_mlp": {
            "setter": "set_post_norm_pre_mlp_hook",
            "write_arity": 3,
            "read_arity": 1,
        },
    },

    # ── model families: how to reach decoder layers + their index, and where
    # each apply-anchor lives for that model. The .xmix declares which model via
    # a `#model: <name>` pragma (default below); the compiler bakes that model's
    # concrete file per anchor into the synthesized XMIX-APPLY tags. The runner
    # file is shared across models; only the model file (model_construct) differs.
    # Add a new model by adding an entry here — no code change.
    "models": {
        "llama": {
            "arch": "LlamaForCausalLM",
            "layers_attr": "model.layers",
            "layer_idx_attr": "self_attn.layer_idx",
            "anchors": {
                "runner_setup": "vllm/v1/worker/gpu_model_runner.py",
                "runner_postload": "vllm/v1/worker/gpu_model_runner.py",
                "model_construct": "vllm/model_executor/models/llama.py",
                "model_forward": "vllm/model_executor/models/llama.py",
            },
        },
        "qwen": {
            "arch": "Qwen3ForCausalLM",
            "layers_attr": "model.layers",
            "layer_idx_attr": "self_attn.layer_idx",
            "anchors": {
                "runner_setup": "vllm/v1/worker/gpu_model_runner.py",
                "runner_postload": "vllm/v1/worker/gpu_model_runner.py",
                "model_construct": "vllm/model_executor/models/qwen3.py",
                "model_forward": "vllm/model_executor/models/qwen3.py",
            },
        },
        "mixtral": {
            "arch": "MixtralForCausalLM",
            "layers_attr": "model.layers",
            "layer_idx_attr": "self_attn.layer_idx",
            "anchors": {
                "runner_setup": "vllm/v1/worker/gpu_model_runner.py",
                "runner_postload": "vllm/v1/worker/gpu_model_runner.py",
                "model_construct": "vllm/model_executor/models/mixtral.py",
                "model_forward": "vllm/model_executor/models/mixtral.py",
            },
        },
    },

    # Model assumed when an .xmix omits the `#model: <name>` pragma.
    "default_model": "llama",
}
