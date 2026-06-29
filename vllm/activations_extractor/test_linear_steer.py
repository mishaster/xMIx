# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Tests for SteeringLinear and linear_select_write_kernel in claude_app.py.

Run with:
    pytest vllm/activations_extractor/test_linear_steer.py -v
"""

import pytest
import torch

from vllm.activations_extractor.claude_app import SteeringLinear


# ── helpers ───────────────────────────────────────────────────────────────────

def make_steerer(
    hidden_size: int = 64,
    max_tokens:  int = 16,
) -> SteeringLinear:
    W  = torch.zeros(hidden_size, hidden_size, device="cuda", dtype=torch.float32)
    b  = torch.zeros(hidden_size,              device="cuda", dtype=torch.float32)
    vi = torch.zeros(max_tokens,               device="cuda", dtype=torch.int32)
    return SteeringLinear(W, b, vi, max_tokens)


def set_active_tokens(s: SteeringLinear, indices: list[int]) -> None:
    n = len(indices)
    s.token_indices[:n].copy_(
        torch.tensor(indices, dtype=torch.int32, device="cuda")
    )
    s.n_valid_buf.fill_(n)


def reference_linear(
    x:             torch.Tensor,   # (n_rows, hidden_size)
    W:             torch.Tensor,   # (hidden_size, hidden_size)
    b:             torch.Tensor,   # (hidden_size,)
    token_indices: list[int],
    vec_indices:   torch.Tensor,   # (n_rows,) int32
) -> torch.Tensor:
    """Pure PyTorch reference: h' = W @ h + b for selected+gated rows."""
    x = x.clone()
    for row in token_indices:
        if vec_indices[row].item() != 0:
            x[row] = (W.float() @ x[row].float() + b.float()).to(x.dtype)
    return x


# ── kernel correctness ────────────────────────────────────────────────────────

class TestKernelCorrectness:

    @pytest.mark.parametrize("n_tokens,hidden_size", [
        (1,   64),
        (4,   64),
        (8,   128),
        (8,   100),   # non-power-of-2 hidden
    ])
    def test_matches_pytorch_reference(self, n_tokens, hidden_size):
        torch.manual_seed(0)
        s = make_steerer(hidden_size=hidden_size, max_tokens=n_tokens)

        W  = torch.randn(hidden_size, hidden_size, device="cuda", dtype=torch.float32)
        b  = torch.randn(hidden_size,              device="cuda", dtype=torch.float32)
        vi = torch.ones(n_tokens,                  device="cuda", dtype=torch.int32)
        x  = torch.randn(n_tokens, hidden_size,    device="cuda", dtype=torch.float32)

        s.load_weights(W, b)
        s.load_vec_indices(vi)
        set_active_tokens(s, list(range(n_tokens)))

        x_ref = reference_linear(x, W, b, list(range(n_tokens)), vi)
        s.run(x)

        torch.testing.assert_close(x.float(), x_ref.float(), atol=1e-4, rtol=1e-4,
            msg=f"Mismatch for n_tokens={n_tokens}, hidden={hidden_size}")

    def test_identity_W_zero_b_is_noop(self):
        """W=I, b=0 → output equals input for all steered rows."""
        hidden, n_tok = 64, 4
        s = make_steerer(hidden_size=hidden, max_tokens=n_tok)

        W  = torch.eye(hidden, device="cuda", dtype=torch.float32)
        b  = torch.zeros(hidden, device="cuda", dtype=torch.float32)
        vi = torch.ones(n_tok, device="cuda", dtype=torch.int32)

        s.load_weights(W, b)
        s.load_vec_indices(vi)
        set_active_tokens(s, list(range(n_tok)))

        x        = torch.randn(n_tok, hidden, device="cuda", dtype=torch.float32)
        x_before = x.clone()
        s.run(x)

        torch.testing.assert_close(x, x_before, atol=1e-5, rtol=0,
            msg="W=I, b=0 must leave x unchanged")

    def test_zero_W_leaves_only_bias(self):
        """W=0, b=c → output should be all-c for steered rows."""
        hidden, n_tok = 64, 4
        s = make_steerer(hidden_size=hidden, max_tokens=n_tok)

        W  = torch.zeros(hidden, hidden, device="cuda", dtype=torch.float32)
        b  = torch.ones(hidden, device="cuda", dtype=torch.float32) * 2.5
        vi = torch.ones(n_tok, device="cuda", dtype=torch.int32)

        s.load_weights(W, b)
        s.load_vec_indices(vi)
        set_active_tokens(s, list(range(n_tok)))

        x = torch.randn(n_tok, hidden, device="cuda", dtype=torch.float32)
        s.run(x)

        expected = torch.full((n_tok, hidden), 2.5, device="cuda", dtype=torch.float32)
        torch.testing.assert_close(x, expected, atol=1e-5, rtol=0)

    def test_hot_swap_weights(self):
        """Loading different weights changes the output on the next run."""
        hidden, n_tok = 64, 4
        x = torch.randn(n_tok, hidden, device="cuda", dtype=torch.float32)
        vi = torch.ones(n_tok, device="cuda", dtype=torch.int32)

        s = make_steerer(hidden_size=hidden, max_tokens=n_tok)
        s.load_vec_indices(vi)
        set_active_tokens(s, list(range(n_tok)))

        W1 = torch.eye(hidden, device="cuda", dtype=torch.float32)
        b1 = torch.ones(hidden, device="cuda", dtype=torch.float32)
        s.load_weights(W1, b1)
        x1 = x.clone()
        s.run(x1)

        W2 = torch.eye(hidden, device="cuda", dtype=torch.float32) * (-1.0)
        b2 = torch.zeros(hidden, device="cuda", dtype=torch.float32)
        s.load_weights(W2, b2)
        x2 = x.clone()
        s.run(x2)

        assert not torch.allclose(x1, x2), \
            "Different weights produced identical output"


# ── vec_indices gating ────────────────────────────────────────────────────────

class TestVecIndicesGating:

    def test_zero_flag_skips_token(self):
        hidden, n_tok = 64, 4
        s = make_steerer(hidden_size=hidden, max_tokens=n_tok)

        W  = torch.randn(hidden, hidden, device="cuda", dtype=torch.float32)
        b  = torch.randn(hidden,         device="cuda", dtype=torch.float32)
        vi = torch.zeros(n_tok,          device="cuda", dtype=torch.int32)

        s.load_weights(W, b)
        s.load_vec_indices(vi)
        set_active_tokens(s, list(range(n_tok)))

        x        = torch.randn(n_tok, hidden, device="cuda", dtype=torch.float32)
        x_before = x.clone()
        s.run(x)

        torch.testing.assert_close(x, x_before,
            msg="Tokens with vec_indices=0 must not be modified")

    def test_mixed_flags(self):
        """Rows 0,2,4 steered; rows 1,3,5 untouched."""
        hidden, n_tok = 64, 6
        s = make_steerer(hidden_size=hidden, max_tokens=n_tok)

        W  = torch.randn(hidden, hidden, device="cuda", dtype=torch.float32)
        b  = torch.randn(hidden,         device="cuda", dtype=torch.float32)
        vi = torch.tensor([1, 0, 1, 0, 1, 0], device="cuda", dtype=torch.int32)

        s.load_weights(W, b)
        s.load_vec_indices(vi)
        set_active_tokens(s, list(range(n_tok)))

        x        = torch.randn(n_tok, hidden, device="cuda", dtype=torch.float32)
        x_before = x.clone()
        x_ref    = reference_linear(x, W, b, list(range(n_tok)), vi)
        s.run(x)

        for row in [1, 3, 5]:
            torch.testing.assert_close(x[row], x_before[row],
                msg=f"Row {row} has vec_indices=0 but was modified")
        for row in [0, 2, 4]:
            torch.testing.assert_close(x[row].float(), x_ref[row].float(),
                atol=1e-4, rtol=1e-4,
                msg=f"Row {row} has vec_indices=1 but result is wrong")


# ── n_valid_buf gating ────────────────────────────────────────────────────────

class TestNValidBuf:

    def test_n_valid_zero_leaves_x_unchanged(self):
        hidden, n_tok = 64, 4
        s = make_steerer(hidden_size=hidden, max_tokens=n_tok)
        s.load_weights(
            torch.randn(hidden, hidden, device="cuda", dtype=torch.float32),
            torch.randn(hidden,         device="cuda", dtype=torch.float32),
        )
        s.load_vec_indices(torch.ones(n_tok, device="cuda", dtype=torch.int32))
        s.token_indices[:n_tok].copy_(torch.arange(n_tok, dtype=torch.int32, device="cuda"))
        s.n_valid_buf.fill_(0)

        x        = torch.randn(n_tok, hidden, device="cuda", dtype=torch.float32)
        x_before = x.clone()
        s.run(x)

        torch.testing.assert_close(x, x_before,
            msg="n_valid_buf=0 must leave x unchanged")

    def test_only_n_valid_tokens_processed(self):
        hidden, n_tok = 64, 4
        s = make_steerer(hidden_size=hidden, max_tokens=n_tok)

        W  = torch.randn(hidden, hidden, device="cuda", dtype=torch.float32)
        b  = torch.randn(hidden,         device="cuda", dtype=torch.float32)
        vi = torch.ones(n_tok, device="cuda", dtype=torch.int32)

        s.load_weights(W, b)
        s.load_vec_indices(vi)
        s.token_indices[:n_tok].copy_(torch.arange(n_tok, dtype=torch.int32, device="cuda"))
        s.n_valid_buf.fill_(2)

        x        = torch.randn(n_tok, hidden, device="cuda", dtype=torch.float32)
        x_before = x.clone()
        x_ref    = reference_linear(x, W, b, [0, 1], vi)
        s.run(x)

        torch.testing.assert_close(x[0].float(), x_ref[0].float(), atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(x[1].float(), x_ref[1].float(), atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(x[2], x_before[2], atol=0, rtol=0)
        torch.testing.assert_close(x[3], x_before[3], atol=0, rtol=0)


# ── buffer stability (CUDA-graph safety) ──────────────────────────────────────

class TestBufferStability:

    def test_load_weights_does_not_change_addresses(self):
        """load_weights() must not reallocate — buffer addresses must be stable."""
        hidden = 64
        s = make_steerer(hidden_size=hidden, max_tokens=8)
        addr_W = s.W.data_ptr()
        addr_b = s.b.data_ptr()

        s.load_weights(
            torch.randn(hidden, hidden, device="cuda", dtype=torch.float32),
            torch.randn(hidden,         device="cuda", dtype=torch.float32),
        )

        assert s.W.data_ptr() == addr_W, "load_weights() changed W buffer address"
        assert s.b.data_ptr() == addr_b, "load_weights() changed b buffer address"


# ── unselected rows are never modified ────────────────────────────────────────

class TestUnselectedRowsUnchanged:

    def test_unselected_rows_not_modified(self):
        """GEMM runs on all rows, but only selected+gated rows of x are updated."""
        hidden  = 64
        n_rows  = 10
        n_steer = 3
        # max_tokens must be >= n_rows so output_buf can hold all GEMM results
        s = make_steerer(hidden_size=hidden, max_tokens=n_rows)

        W  = torch.randn(hidden, hidden, device="cuda", dtype=torch.float32)
        b  = torch.randn(hidden,         device="cuda", dtype=torch.float32)
        vi = torch.ones(n_rows,          device="cuda", dtype=torch.int32)

        s.load_weights(W, b)
        s.load_vec_indices(vi)
        steer_rows = [2, 5, 9]
        set_active_tokens(s, steer_rows)

        x        = torch.randn(n_rows, hidden, device="cuda", dtype=torch.float32)
        x_before = x.clone()
        s.run(x)

        for row in range(n_rows):
            if row not in steer_rows:
                torch.testing.assert_close(x[row], x_before[row],
                    msg=f"Row {row} was modified but should be untouched")
