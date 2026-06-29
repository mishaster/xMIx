# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Tests for SteeringLMSteer and lm_steer_residual_kernel in claude_app.py.

Run with:
    pytest vllm/activations_extractor/test_lm_steer.py -v
"""

import pytest
import torch

from vllm.activations_extractor.claude_app import SteeringLMSteer


# ── helpers ───────────────────────────────────────────────────────────────────

def make_steerer(
    hidden_size: int = 64,
    rank:        int = 8,
    max_tokens:  int = 16,
) -> SteeringLMSteer:
    P1 = torch.zeros(hidden_size, rank, device="cuda", dtype=torch.float32)
    P2 = torch.zeros(hidden_size, rank, device="cuda", dtype=torch.float32)
    vi = torch.zeros(max_tokens,        device="cuda", dtype=torch.int32)
    return SteeringLMSteer(P1, P2, 1.0, vi, max_tokens)


def set_active_tokens(s: SteeringLMSteer, indices: list[int]) -> None:
    n = len(indices)
    s.token_indices[:n].copy_(
        torch.tensor(indices, dtype=torch.int32, device="cuda")
    )
    s.n_valid_buf.fill_(n)


def reference_lm_steer(
    x:             torch.Tensor,   # (n_rows, hidden_size)
    P1:            torch.Tensor,   # (hidden_size, rank)
    P2:            torch.Tensor,   # (hidden_size, rank)
    alpha:         float,
    token_indices: list[int],
    vec_indices:   torch.Tensor,   # (n_rows,) int32
) -> torch.Tensor:
    """Pure PyTorch reference: h += alpha * (h @ P1) @ P2^T for selected+gated rows."""
    x = x.clone().float()
    P1f = P1.float()
    P2f = P2.float()
    for row in token_indices:
        if vec_indices[row].item() != 0:
            delta   = (x[row] @ P1f) @ P2f.T
            x[row] += alpha * delta
    return x


# ── kernel correctness ────────────────────────────────────────────────────────

class TestKernelCorrectness:

    @pytest.mark.parametrize("n_tokens,hidden_size,rank", [
        (1,   64,   4),
        (4,   64,   8),
        (8,   512,  16),
        (8,   100,  4),    # non-power-of-2 hidden
    ])
    def test_matches_pytorch_reference(self, n_tokens, hidden_size, rank):
        torch.manual_seed(0)
        s = make_steerer(hidden_size=hidden_size, rank=rank, max_tokens=n_tokens)

        P1 = torch.randn(hidden_size, rank,     device="cuda", dtype=torch.float32)
        P2 = torch.randn(hidden_size, rank,     device="cuda", dtype=torch.float32)
        vi = torch.ones(n_tokens,               device="cuda", dtype=torch.int32)
        x  = torch.randn(n_tokens, hidden_size, device="cuda", dtype=torch.float32)

        s.load_projectors(P1, P2, alpha=1.0)
        s.load_vec_indices(vi)
        set_active_tokens(s, list(range(n_tokens)))

        x_ref = reference_lm_steer(x, P1, P2, 1.0, list(range(n_tokens)), vi)
        s.run(x)

        torch.testing.assert_close(x.float(), x_ref.float(), atol=1e-4, rtol=1e-4,
            msg=f"Mismatch for n_tokens={n_tokens}, hidden={hidden_size}, rank={rank}")

    def test_alpha_zero_is_noop(self):
        """alpha=0 → delta is added with scale 0 → x must be unchanged."""
        hidden, rank, n_tok = 64, 8, 4
        s = make_steerer(hidden_size=hidden, rank=rank, max_tokens=n_tok)
        P1 = torch.randn(hidden, rank, device="cuda", dtype=torch.float32)
        P2 = torch.randn(hidden, rank, device="cuda", dtype=torch.float32)
        vi = torch.ones(n_tok, device="cuda", dtype=torch.int32)

        s.load_projectors(P1, P2, alpha=0.0)
        s.load_vec_indices(vi)
        set_active_tokens(s, list(range(n_tok)))

        x        = torch.randn(n_tok, hidden, device="cuda", dtype=torch.float32)
        x_before = x.clone()
        s.run(x)

        torch.testing.assert_close(x, x_before, atol=1e-5, rtol=0,
            msg="alpha=0 must leave x unchanged")

    def test_negative_alpha_subtracts(self):
        """alpha=-1 should subtract the low-rank residual."""
        hidden, rank, n_tok = 64, 8, 4
        torch.manual_seed(3)
        s_pos = make_steerer(hidden_size=hidden, rank=rank, max_tokens=n_tok)
        s_neg = make_steerer(hidden_size=hidden, rank=rank, max_tokens=n_tok)

        P1 = torch.randn(hidden, rank, device="cuda", dtype=torch.float32)
        P2 = torch.randn(hidden, rank, device="cuda", dtype=torch.float32)
        vi = torch.ones(n_tok, device="cuda", dtype=torch.int32)
        x_orig = torch.randn(n_tok, hidden, device="cuda", dtype=torch.float32)

        for s, alpha in [(s_pos, 1.0), (s_neg, -1.0)]:
            s.load_projectors(P1, P2, alpha=alpha)
            s.load_vec_indices(vi)
            set_active_tokens(s, list(range(n_tok)))

        x_pos = x_orig.clone()
        s_pos.run(x_pos)
        x_neg = x_orig.clone()
        s_neg.run(x_neg)

        # x_pos + x_neg = 2 * x_orig + delta - delta = 2 * x_orig
        torch.testing.assert_close(
            x_pos + x_neg,
            2.0 * x_orig.float(),
            atol=1e-4, rtol=1e-4,
            msg="alpha=+1 and alpha=-1 should produce symmetric deltas",
        )

    @pytest.mark.parametrize("alpha", [0.5, 2.0, -0.3])
    def test_alpha_scaling(self, alpha):
        """Result must scale linearly with alpha."""
        hidden, rank, n_tok = 64, 8, 4
        torch.manual_seed(4)
        P1 = torch.randn(hidden, rank, device="cuda", dtype=torch.float32)
        P2 = torch.randn(hidden, rank, device="cuda", dtype=torch.float32)
        vi = torch.ones(n_tok, device="cuda", dtype=torch.int32)
        x_orig = torch.randn(n_tok, hidden, device="cuda", dtype=torch.float32)

        def run_with_alpha(a):
            s = make_steerer(hidden_size=hidden, rank=rank, max_tokens=n_tok)
            s.load_projectors(P1, P2, alpha=a)
            s.load_vec_indices(vi)
            set_active_tokens(s, list(range(n_tok)))
            x = x_orig.clone()
            s.run(x)
            return x

        x_ref = reference_lm_steer(x_orig, P1, P2, alpha, list(range(n_tok)), vi)
        x_out = run_with_alpha(alpha)

        torch.testing.assert_close(x_out.float(), x_ref.float(), atol=1e-4, rtol=1e-4,
            msg=f"Mismatch with alpha={alpha}")


# ── vec_indices gating ────────────────────────────────────────────────────────

class TestVecIndicesGating:

    def test_zero_flag_skips_token(self):
        hidden, rank, n_tok = 64, 8, 4
        s = make_steerer(hidden_size=hidden, rank=rank, max_tokens=n_tok)
        s.load_projectors(
            torch.randn(hidden, rank, device="cuda", dtype=torch.float32),
            torch.randn(hidden, rank, device="cuda", dtype=torch.float32),
        )
        s.load_vec_indices(torch.zeros(n_tok, device="cuda", dtype=torch.int32))
        set_active_tokens(s, list(range(n_tok)))

        x        = torch.randn(n_tok, hidden, device="cuda", dtype=torch.float32)
        x_before = x.clone()
        s.run(x)

        torch.testing.assert_close(x, x_before,
            msg="Tokens with vec_indices=0 must not be modified")

    def test_mixed_flags(self):
        """Rows 0,2,4 steered; rows 1,3,5 untouched."""
        hidden, rank, n_tok = 64, 8, 6
        s = make_steerer(hidden_size=hidden, rank=rank, max_tokens=n_tok)

        P1 = torch.randn(hidden, rank, device="cuda", dtype=torch.float32)
        P2 = torch.randn(hidden, rank, device="cuda", dtype=torch.float32)
        vi = torch.tensor([1, 0, 1, 0, 1, 0], device="cuda", dtype=torch.int32)

        s.load_projectors(P1, P2, alpha=1.0)
        s.load_vec_indices(vi)
        set_active_tokens(s, list(range(n_tok)))

        x        = torch.randn(n_tok, hidden, device="cuda", dtype=torch.float32)
        x_before = x.clone()
        x_ref    = reference_lm_steer(x, P1, P2, 1.0, list(range(n_tok)), vi)
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
        hidden, rank, n_tok = 64, 8, 4
        s = make_steerer(hidden_size=hidden, rank=rank, max_tokens=n_tok)
        s.load_projectors(
            torch.randn(hidden, rank, device="cuda", dtype=torch.float32),
            torch.randn(hidden, rank, device="cuda", dtype=torch.float32),
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
        hidden, rank, n_tok = 64, 8, 4
        s = make_steerer(hidden_size=hidden, rank=rank, max_tokens=n_tok)

        P1 = torch.randn(hidden, rank, device="cuda", dtype=torch.float32)
        P2 = torch.randn(hidden, rank, device="cuda", dtype=torch.float32)
        vi = torch.ones(n_tok, device="cuda", dtype=torch.int32)

        s.load_projectors(P1, P2, alpha=1.0)
        s.load_vec_indices(vi)
        s.token_indices[:n_tok].copy_(torch.arange(n_tok, dtype=torch.int32, device="cuda"))
        s.n_valid_buf.fill_(2)

        x        = torch.randn(n_tok, hidden, device="cuda", dtype=torch.float32)
        x_before = x.clone()
        x_ref    = reference_lm_steer(x, P1, P2, 1.0, [0, 1], vi)
        s.run(x)

        torch.testing.assert_close(x[0].float(), x_ref[0].float(), atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(x[1].float(), x_ref[1].float(), atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(x[2], x_before[2], atol=0, rtol=0)
        torch.testing.assert_close(x[3], x_before[3], atol=0, rtol=0)


# ── buffer stability (CUDA-graph safety) ──────────────────────────────────────

class TestBufferStability:

    def test_load_projectors_does_not_change_addresses(self):
        hidden, rank = 64, 8
        s = make_steerer(hidden_size=hidden, rank=rank, max_tokens=8)
        addr_P1    = s.P1.data_ptr()
        addr_P2    = s.P2.data_ptr()
        addr_alpha = s.alpha.data_ptr()

        s.load_projectors(
            torch.randn(hidden, rank, device="cuda", dtype=torch.float32),
            torch.randn(hidden, rank, device="cuda", dtype=torch.float32),
            alpha=2.0,
        )

        assert s.P1.data_ptr()    == addr_P1,    "P1 buffer address changed"
        assert s.P2.data_ptr()    == addr_P2,    "P2 buffer address changed"
        assert s.alpha.data_ptr() == addr_alpha, "alpha buffer address changed"


# ── unselected rows are never modified ────────────────────────────────────────

class TestUnselectedRowsUnchanged:

    def test_unselected_rows_not_modified(self):
        """GEMMs run on all rows, but only selected+gated rows of x are updated."""
        hidden, rank = 64, 8
        n_rows  = 10
        n_steer = 3
        # max_tokens must be >= n_rows so tmp_buf/delta_buf can hold all GEMM results
        s = make_steerer(hidden_size=hidden, rank=rank, max_tokens=n_rows)

        P1 = torch.randn(hidden, rank, device="cuda", dtype=torch.float32)
        P2 = torch.randn(hidden, rank, device="cuda", dtype=torch.float32)
        vi = torch.ones(n_rows, device="cuda", dtype=torch.int32)

        s.load_projectors(P1, P2, alpha=1.5)
        s.load_vec_indices(vi)
        steer_rows = [0, 3, 8]
        set_active_tokens(s, steer_rows)

        x        = torch.randn(n_rows, hidden, device="cuda", dtype=torch.float32)
        x_before = x.clone()
        s.run(x)

        for row in range(n_rows):
            if row not in steer_rows:
                torch.testing.assert_close(x[row], x_before[row],
                    msg=f"Row {row} was modified but should be untouched")
