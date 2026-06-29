# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Tests for MoERouterLogitSteerer and moe_router_logit_steer_kernel in
vllm/activations_extractor/moe_router_steering.py.

Compares the kernel output against a pure-PyTorch reference function
transcribed verbatim from
SteerMoE/src/modeling_vllm/mixtral.py:118-136 (with the per-row gate added).

Run with:
    pytest vllm/activations_extractor/test_moe_router_steering.py -v
"""

import pytest
import torch

from vllm.activations_extractor.moe_router_steering import (
    MoERouterLogitSteerer,
)


# ── reference (pure PyTorch) ──────────────────────────────────────────────────

def reference_steer_router_logits(
    logits:    torch.Tensor,   # (n_tokens, num_experts) — original dtype
    weights:   torch.Tensor,   # (num_experts,) int32, values in {-1, 0, +1}
    input_map: torch.Tensor,   # (n_tokens,) int32 — 0 = pass through
    eps:       float = 0.01,
) -> torch.Tensor:
    """Mirrors SteerMoE/src/modeling_vllm/mixtral.py:118-136 with per-row gate."""
    out    = logits.clone()
    active = input_map.bool()
    if active.any():
        rows = torch.nn.functional.log_softmax(out[active], dim=-1)
        max_per_tok = rows.max(dim=-1, keepdim=True).values
        min_per_tok = rows.min(dim=-1, keepdim=True).values
        pos_mask = (weights > 0)
        neg_mask = (weights < 0)
        n_pos = int(pos_mask.sum().item())
        n_neg = int(neg_mask.sum().item())
        if n_pos > 0:
            rows[:, pos_mask] = (max_per_tok + eps).expand(-1, n_pos).to(rows.dtype)
        if n_neg > 0:
            rows[:, neg_mask] = (min_per_tok - eps).expand(-1, n_neg).to(rows.dtype)
        out[active] = rows
    return out


# ── helpers ───────────────────────────────────────────────────────────────────

DTYPES_FLOAT = [torch.float32, torch.float16, torch.bfloat16]


def _tolerance(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.float32:
        return 1e-4, 1e-4
    if dtype == torch.float16:
        return 2e-3, 2e-3
    # bfloat16 has only a 7-bit mantissa; 1 ULP at |x|~6 is ~0.047, so allow
    # slack for the inevitable last-cast rounding differences between Triton
    # and PyTorch's log_softmax.
    return 5e-2, 5e-2


def _make_steerer(num_experts: int, max_tokens: int, eps: float = 0.01,
                  device: str = "cuda",
                  weights: torch.Tensor | None = None,
                  input_map: torch.Tensor | None = None,
                  ) -> MoERouterLogitSteerer:
    if weights is None:
        weights = torch.zeros(num_experts, dtype=torch.int32, device=device)
    return MoERouterLogitSteerer(
        num_experts=num_experts,
        max_tokens=max_tokens,
        weights=weights,
        input_map=input_map,
        eps=eps,
        device=device,
    )


def _random_weights(num_experts: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(seed)
    cpu_w = torch.randint(-1, 2, (num_experts,), generator=g, dtype=torch.int32)
    return cpu_w.to("cuda")


# ── parametrized parity vs reference ──────────────────────────────────────────

class TestKernelCorrectness:

    @pytest.mark.parametrize("n_tokens,num_experts", [
        (1, 8), (4, 8), (16, 8),       # Mixtral-shape
        (1, 128), (8, 128), (16, 128), # Qwen3-MoE-shape
        (8, 32),                       # mid size
        (4, 12),                       # non-power-of-two num_experts
    ])
    @pytest.mark.parametrize("dtype", DTYPES_FLOAT)
    def test_matches_reference(self, n_tokens, num_experts, dtype):
        torch.manual_seed(0)
        steerer = _make_steerer(num_experts, n_tokens)

        logits     = torch.randn(n_tokens, num_experts, dtype=dtype, device="cuda")
        logits_ref = logits.clone()
        weights    = _random_weights(num_experts, seed=42)
        input_map  = torch.ones(n_tokens, dtype=torch.int32, device="cuda")

        steerer.load_weights(weights)
        steerer.load_input_map(input_map)
        n_valid_buf = torch.tensor([n_tokens], dtype=torch.int32, device="cuda")
        steerer(logits, n_valid_buf)

        expected = reference_steer_router_logits(logits_ref, weights, input_map)

        atol, rtol = _tolerance(dtype)
        torch.testing.assert_close(
            logits.float(), expected.float(),
            atol=atol, rtol=rtol,
            msg=f"Mismatch for n_tokens={n_tokens}, num_experts={num_experts}, "
                f"dtype={dtype}",
        )

    def test_input_map_all_zeros_is_noop(self):
        """input_map=0 everywhere: output bitwise identical to input."""
        n_tokens, num_experts = 8, 8
        steerer = _make_steerer(num_experts, n_tokens)

        torch.manual_seed(1)
        logits   = torch.randn(n_tokens, num_experts, dtype=torch.float32, device="cuda")
        original = logits.clone()

        steerer.load_weights(torch.tensor([1, -1, 0, 1, -1, 0, 1, -1],
                                          dtype=torch.int32, device="cuda"))
        steerer.load_input_map(torch.zeros(n_tokens, dtype=torch.int32, device="cuda"))
        n_valid_buf = torch.tensor([n_tokens], dtype=torch.int32, device="cuda")
        steerer(logits, n_valid_buf)

        assert torch.equal(logits, original), \
            "input_map=0 must leave logits bitwise unchanged"

    def test_weights_all_zero_is_logsoftmax_on_active_rows(self):
        """weights=0 everywhere on active rows: output equals log_softmax(input)."""
        n_tokens, num_experts = 6, 8
        steerer = _make_steerer(num_experts, n_tokens)

        torch.manual_seed(2)
        logits   = torch.randn(n_tokens, num_experts, dtype=torch.float32, device="cuda")
        original = logits.clone()

        steerer.load_weights(torch.zeros(num_experts, dtype=torch.int32, device="cuda"))
        # First half active, second half passthrough
        input_map = torch.tensor([1, 1, 1, 0, 0, 0], dtype=torch.int32, device="cuda")
        steerer.load_input_map(input_map)
        n_valid_buf = torch.tensor([n_tokens], dtype=torch.int32, device="cuda")
        steerer(logits, n_valid_buf)

        expected_active = torch.nn.functional.log_softmax(original[:3], dim=-1)
        torch.testing.assert_close(logits[:3], expected_active, atol=1e-5, rtol=1e-5)
        assert torch.equal(logits[3:], original[3:]), \
            "Inactive rows must remain bitwise unchanged"

    def test_all_positive_weights(self):
        """weights=+1 everywhere: every active expert entry equals row_max+eps."""
        n_tokens, num_experts = 4, 8
        eps = 0.01
        steerer = _make_steerer(num_experts, n_tokens, eps=eps)

        torch.manual_seed(3)
        logits = torch.randn(n_tokens, num_experts, dtype=torch.float32, device="cuda")

        steerer.load_weights(torch.ones(num_experts, dtype=torch.int32, device="cuda"))
        steerer.load_input_map(torch.ones(n_tokens, dtype=torch.int32, device="cuda"))
        n_valid_buf = torch.tensor([n_tokens], dtype=torch.int32, device="cuda")

        log_sm   = torch.nn.functional.log_softmax(logits, dim=-1)
        row_max  = log_sm.max(dim=-1, keepdim=True).values
        expected = (row_max + eps).expand_as(logits).contiguous()

        steerer(logits, n_valid_buf)

        torch.testing.assert_close(logits, expected, atol=1e-5, rtol=1e-5)

    def test_all_negative_weights(self):
        """weights=-1 everywhere: every active expert entry equals row_min-eps.

        Pins the lane masking: invalid lanes must be +inf for the min reduction,
        otherwise an all-negative all-positive-weights case would still pass but
        an all-negative-weights row could pick up a fake 0 minimum from invalid
        lanes when num_experts is not a power of two.
        """
        n_tokens, num_experts = 4, 12   # non-power-of-two on purpose
        eps = 0.01
        steerer = _make_steerer(num_experts, n_tokens, eps=eps)

        torch.manual_seed(4)
        logits = torch.randn(n_tokens, num_experts, dtype=torch.float32, device="cuda")

        steerer.load_weights(-torch.ones(num_experts, dtype=torch.int32, device="cuda"))
        steerer.load_input_map(torch.ones(n_tokens, dtype=torch.int32, device="cuda"))
        n_valid_buf = torch.tensor([n_tokens], dtype=torch.int32, device="cuda")

        log_sm   = torch.nn.functional.log_softmax(logits, dim=-1)
        row_min  = log_sm.min(dim=-1, keepdim=True).values
        expected = (row_min - eps).expand_as(logits).contiguous()

        steerer(logits, n_valid_buf)

        torch.testing.assert_close(logits, expected, atol=1e-5, rtol=1e-5)

    def test_mixed_gating(self):
        """Off rows untouched bitwise; on rows match reference."""
        n_tokens, num_experts = 8, 8
        steerer = _make_steerer(num_experts, n_tokens)

        torch.manual_seed(5)
        logits     = torch.randn(n_tokens, num_experts, dtype=torch.float32, device="cuda")
        logits_ref = logits.clone()
        original   = logits.clone()

        weights   = torch.tensor([1, 0, -1, 0, 1, 0, -1, 0],
                                 dtype=torch.int32, device="cuda")
        input_map = torch.tensor([1, 0, 1, 0, 1, 0, 1, 0],
                                 dtype=torch.int32, device="cuda")
        steerer.load_weights(weights)
        steerer.load_input_map(input_map)
        n_valid_buf = torch.tensor([n_tokens], dtype=torch.int32, device="cuda")
        steerer(logits, n_valid_buf)

        expected = reference_steer_router_logits(logits_ref, weights, input_map)

        # Inactive rows bitwise unchanged
        for row in [1, 3, 5, 7]:
            assert torch.equal(logits[row], original[row]), \
                f"Row {row} (input_map=0) was modified"
        # Active rows match reference
        for row in [0, 2, 4, 6]:
            torch.testing.assert_close(logits[row], expected[row],
                                       atol=1e-5, rtol=1e-5,
                                       msg=f"Row {row} mismatch")

    def test_padded_batch(self):
        """Rows past n_valid_buf must be untouched even if input_map=1."""
        max_tokens, num_experts = 16, 8
        n_real = 8
        steerer = _make_steerer(num_experts, max_tokens)

        torch.manual_seed(6)
        logits   = torch.randn(max_tokens, num_experts, dtype=torch.float32, device="cuda")
        original = logits.clone()

        steerer.load_weights(_random_weights(num_experts, seed=7))
        steerer.load_input_map(torch.ones(max_tokens, dtype=torch.int32, device="cuda"))
        n_valid_buf = torch.tensor([n_real], dtype=torch.int32, device="cuda")
        steerer(logits, n_valid_buf)

        # Rows >= n_real must be bitwise unchanged
        assert torch.equal(logits[n_real:], original[n_real:]), \
            "Padded rows past n_valid_buf must remain unchanged"
        # Rows < n_real must have changed (random weights are nonzero with high prob;
        # repeat seed if this becomes flaky)
        assert not torch.equal(logits[:n_real], original[:n_real]), \
            "Real rows should have been steered"

    def test_custom_eps(self):
        """eps argument controls the offset magnitude."""
        n_tokens, num_experts = 4, 8
        eps = 0.5
        steerer = _make_steerer(num_experts, n_tokens, eps=eps)

        torch.manual_seed(8)
        logits     = torch.randn(n_tokens, num_experts, dtype=torch.float32, device="cuda")
        logits_ref = logits.clone()

        weights = torch.tensor([1, -1, 0, 1, -1, 0, 1, -1],
                               dtype=torch.int32, device="cuda")
        input_map = torch.ones(n_tokens, dtype=torch.int32, device="cuda")
        steerer.load_weights(weights)
        steerer.load_input_map(input_map)
        n_valid_buf = torch.tensor([n_tokens], dtype=torch.int32, device="cuda")
        steerer(logits, n_valid_buf)

        expected = reference_steer_router_logits(logits_ref, weights, input_map, eps=eps)
        torch.testing.assert_close(logits, expected, atol=1e-5, rtol=1e-5)


# ── module API ────────────────────────────────────────────────────────────────

class TestModuleAPI:

    def test_load_weights_shape_check(self):
        steerer = _make_steerer(num_experts=8, max_tokens=4)
        with pytest.raises(AssertionError):
            steerer.load_weights(torch.zeros(7, dtype=torch.int32, device="cuda"))

    def test_load_input_map_shape_check(self):
        steerer = _make_steerer(num_experts=8, max_tokens=4)
        with pytest.raises(AssertionError):
            steerer.load_input_map(torch.zeros(5, dtype=torch.int32, device="cuda"))

    def test_default_buffers_are_zeros(self):
        """Unconfigured layer is a no-op: input_map=0 by default → no change."""
        steerer = _make_steerer(num_experts=8, max_tokens=4)
        torch.manual_seed(9)
        logits   = torch.randn(4, 8, dtype=torch.float32, device="cuda")
        original = logits.clone()
        n_valid_buf = torch.tensor([4], dtype=torch.int32, device="cuda")
        steerer(logits, n_valid_buf)
        assert torch.equal(logits, original), \
            "Default-constructed steerer should be a no-op (input_map=0)"

    def test_run_alias(self):
        """run() is an alias for forward()."""
        steerer = _make_steerer(num_experts=8, max_tokens=4)
        steerer.load_weights(_random_weights(8, seed=10))
        steerer.load_input_map(torch.ones(4, dtype=torch.int32, device="cuda"))
        n_valid_buf = torch.tensor([4], dtype=torch.int32, device="cuda")

        torch.manual_seed(11)
        logits_a = torch.randn(4, 8, dtype=torch.float32, device="cuda")
        logits_b = logits_a.clone()
        steerer(logits_a, n_valid_buf)
        steerer.run(logits_b, n_valid_buf)

        torch.testing.assert_close(logits_a, logits_b, atol=0.0, rtol=0.0)
