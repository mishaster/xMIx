# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
SteerMoE-style MoE router-logit steering.

This module reproduces the routing intervention from
`SteerMoE/src/modeling_vllm/mixtral.py:118-136` (and the equivalent code in
`olmoe.py` / `qwen3_moe.py`) as a Triton kernel + nn.Module wrapper that
follows the same pattern as `SteeringVectorAdder` /
`SteeringVectorDotSubtractNormalized` in `write_activations.py`.

What the operation does
───────────────────────
Operates on **router logits** (output of `self.gate(hidden_states)`) of shape
(num_tokens, num_experts), in-place. For a per-expert sign vector
`weights ∈ {-1, 0, +1}` and offset `eps` (default 0.01):

    logits = log_softmax(logits, dim=-1)
    logits[:, weights > 0] = max_per_token + eps        # force-promote
    logits[:, weights < 0] = min_per_token - eps        # force-suppress

The +eps / -eps pushes selected experts above/below all others in the row,
forcing top-k selection to include / exclude them. Unsteered rows pass
through unchanged (no log_softmax applied).

Note on FusedMoE compatibility: feeding `log_softmax(logits)` to a downstream
softmax+top_k is equivalent to feeding raw logits, because softmax is
invariant under per-row constant offsets. The +/- eps outliers on steered
experts are the actual mechanism that changes routing.

Per-row gating (input_map)
──────────────────────────
The kernel includes an int32 `input_map` per-row gate that matches the
convention used by the existing steering classes. `input_map[row] == 0`
means the row is left untouched (no log_softmax, no scatter). Set
`input_map` to all-ones to recover SteerMoE's unconditional steering.

Usage example
─────────────
```python
import torch
from vllm.activations_extractor.moe_router_steering import MoERouterLogitSteerer

# Per-expert sign for this layer (e.g. promote experts 0 and 4, suppress 3).
# In the SteerMoE workflow this is one row of weights[L, E], built offline
# by sorting experts by |risk_diff| from a contrastive activation analysis
# pickle and picking top-num_pos_experts (sign +1) and top-num_neg_experts
# (sign -1). Every decoder layer gets its own row.
weights = torch.tensor([1, 0, 0, -1, 1, 0, 0, 0],
                       dtype=torch.int32, device="cuda")

# Per-token gate. All-ones reproduces SteerMoE's unconditional behaviour;
# pass `input_map=None` (or omit) to start zero-allocated and let an
# upstream condition kernel write into it each forward.
input_map = torch.ones(32, dtype=torch.int32, device="cuda")

steerer = MoERouterLogitSteerer(
    num_experts=8, max_tokens=32,
    weights=weights, input_map=input_map,
    eps=0.01, device="cuda",
)

# Caller-owned, address-stable scalar (mirrors SteeringVectorAdder).
n_valid_buf = torch.tensor([12], dtype=torch.int32, device="cuda")

# Synthetic raw router logits (in real use: `self.gate(hidden_states)`).
logits = torch.randn(32, 8, dtype=torch.float16, device="cuda")
steerer(logits, n_valid_buf)   # in-place; ready to feed FusedMoE
```

Warning: `forward` modifies `logits` in place. Calling it twice on the same
tensor would re-apply log_softmax to already-steered values; restore the
input first if you need to re-run.
"""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def moe_router_logit_steer_kernel(
    logits_ptr,         # (max_tokens, NUM_EXPERTS) — modified in-place
    weights_ptr,        # (NUM_EXPERTS,) int32 — values in {-1, 0, +1}
    input_map_ptr,      # (max_tokens,) int32 — 0 = pass row through
    n_valid_ptr,        # (1,) int32 — number of real rows this pass
    eps,                # fp32 scalar (default 0.01)
    NUM_EXPERTS: tl.constexpr,
    BLOCK_E:     tl.constexpr,   # next_power_of_2(NUM_EXPERTS)
):
    row = tl.program_id(0)

    n_valid = tl.load(n_valid_ptr)
    if row >= n_valid:
        return
    if tl.load(input_map_ptr + row) == 0:
        return

    e  = tl.arange(0, BLOCK_E)
    em = e < NUM_EXPERTS

    row_off = row * NUM_EXPERTS

    # Out-of-range lanes load as -inf so they vanish in tl.max and tl.exp.
    NEG_INF = float("-inf")
    POS_INF = float("inf")
    x_raw = tl.load(logits_ptr + row_off + e, mask=em, other=NEG_INF)
    x     = x_raw.to(tl.float32)

    # log_softmax(x) = (x - xmax) - log(sum(exp(x - xmax)))
    xmax    = tl.max(x, axis=0)
    shifted = x - xmax
    shifted = tl.where(em, shifted, NEG_INF)   # invalid lanes contribute 0 to exp
    e_exp   = tl.exp(shifted)
    denom   = tl.sum(e_exp, axis=0)
    log_sm  = shifted - tl.log(denom)          # invalid lanes still -inf

    # Per-row max/min over real experts only.
    row_max = tl.max(log_sm, axis=0)
    log_sm_for_min = tl.where(em, log_sm, POS_INF)
    row_min = tl.min(log_sm_for_min, axis=0)

    pos_val = row_max + eps
    neg_val = row_min - eps

    w = tl.load(weights_ptr + e, mask=em, other=0)

    out = tl.where(w > 0, pos_val,
          tl.where(w < 0, neg_val, log_sm))

    tl.store(logits_ptr + row_off + e, out.to(x_raw.dtype), mask=em)


class MoERouterLogitSteerer(torch.nn.Module):
    """
    nn.Module wrapper around `moe_router_logit_steer_kernel`.

    Reproduces SteerMoE's router-logit intervention. One instance per MoE
    layer: each instance owns a `weights[num_experts]` int32 buffer with
    values in {-1, 0, +1}.

    All persistent buffers are registered as `nn.Parameter(..., requires_grad=False)`
    so Dynamo treats them as module state (no shape guards, CUDA-graph stable).

    Args:
        num_experts: number of experts in this layer (e.g. 8 for Mixtral, 128 for Qwen3).
        max_tokens:  maximum number of token rows the kernel will see in a batch.
                     Used as the kernel grid size and the `input_map` buffer length.
        weights:     (num_experts,) int32 tensor of signs in {-1, 0, +1}. Required.
                     Caller is responsible for dtype and device — registered as-is.
        input_map:   optional (max_tokens,) int32 tensor — per-row gate
                     (0 = pass row through, non-zero = steer). If None, allocated
                     as zeros on `device`. Typically written by an upstream
                     condition kernel (EOLTokenDetector / ConditionEvaluator /
                     CondProjCosSim) before forward().
        eps:         offset added/subtracted from the row max/min (default 0.01).
                     A Python scalar — not CUDA-graph-mutable. Reconstruct the
                     module to change eps mid-graph.
        device:      device used to zero-allocate `input_map` if it is not passed.
    """

    def __init__(
        self,
        num_experts: int,
        max_tokens:  int,
        weights:     torch.Tensor,
        input_map:   torch.Tensor | None = None,
        eps:         float = 0.01,
        device:      str = "cuda",
    ):
        super().__init__()
        assert num_experts >= 1
        assert max_tokens  >= 1

        assert weights.shape == (num_experts,), \
            f"weights shape {weights.shape} must be ({num_experts},)"
        assert weights.dtype == torch.int32, \
            f"weights dtype {weights.dtype} must be torch.int32"

        if input_map is None:
            input_map = torch.zeros(max_tokens, dtype=torch.int32, device=device)
        else:
            assert input_map.shape == (max_tokens,), \
                f"input_map shape {input_map.shape} must be ({max_tokens},)"
            assert input_map.dtype == torch.int32, \
                f"input_map dtype {input_map.dtype} must be torch.int32"

        self.num_experts = num_experts
        self.max_tokens  = max_tokens
        self.eps         = float(eps)
        self._block_e    = triton.next_power_of_2(num_experts)

        # Steering payload — registered as-is from caller.
        self.register_parameter('weights',
            torch.nn.Parameter(weights, requires_grad=False))

        # Per-row gate. Either caller-supplied or zero-allocated.
        # Read directly by the kernel each forward; CUDA-graph stable.
        self.register_parameter('input_map',
            torch.nn.Parameter(input_map, requires_grad=False))

    def load_weights(self, weights: torch.Tensor) -> None:
        """Copy per-expert sign vector into the pre-allocated buffer.

        Args:
            weights: (num_experts,) — values in {-1, 0, +1}. Cast to int32.
        """
        assert weights.shape == self.weights.shape, \
            f"Shape mismatch: expected {self.weights.shape}, got {weights.shape}"
        self.weights.copy_(weights.to(torch.int32))

    def load_input_map(self, input_map: torch.Tensor) -> None:
        """Copy per-row gate into the pre-allocated buffer.

        Args:
            input_map: (max_tokens,) — 0 = pass row through, non-zero = steer.
        """
        assert input_map.shape == self.input_map.shape, \
            f"Shape mismatch: expected {self.input_map.shape}, got {input_map.shape}"
        self.input_map.copy_(input_map.to(torch.int32))

    def forward(
        self,
        logits:      torch.Tensor,   # (n_rows, num_experts) — modified in-place
        n_valid_buf: torch.Tensor,   # (1,) int32 GPU tensor — real row count
    ) -> torch.Tensor:
        """
        Apply SteerMoE router-logit steering in place.

        For each row r where `r < n_valid_buf[0]` and `input_map[r] != 0`:
            - log_softmax across experts
            - replace logits[r, e] with row_max + eps where weights[e] > 0
            - replace logits[r, e] with row_min - eps where weights[e] < 0
        Other rows are left untouched (no log_softmax applied).

        Args:
            logits:      (n_rows, num_experts) router logits — modified in-place.
                         Last dim must equal `num_experts`. Any floating dtype.
            n_valid_buf: (1,) int32 GPU tensor; number of real rows in `logits`
                         this pass. Caller-owned, address-stable for CUDA graph.

        Returns:
            The same `logits` tensor (modified in place).
        """
        #assert logits.shape[-1] == self.num_experts, \
        #    f"logits last dim {logits.shape[-1]} != num_experts {self.num_experts}"
        moe_router_logit_steer_kernel[(self.max_tokens,)](
            logits, self.weights, self.input_map, n_valid_buf,
            self.eps,
            NUM_EXPERTS=self.num_experts,
            BLOCK_E=self._block_e,
        )
        return logits

    def run(
        self,
        logits:      torch.Tensor,
        n_valid_buf: torch.Tensor,
    ) -> torch.Tensor:
        return self(logits, n_valid_buf)
