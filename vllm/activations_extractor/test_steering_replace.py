# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Tests for SteeringVectorReplacer and replace_selected_kernel in claude_app.py.

Run with:
    pytest vllm/activations_extractor/test_steering_replace.py -v
"""

import pytest
import torch

from vllm.activations_extractor.claude_app import SteeringVectorReplacer


# ── helpers ───────────────────────────────────────────────────────────────────

def make_replacer(
    hidden_size: int = 64,
    max_tokens:  int = 16,
    dtype:       torch.dtype = torch.float32,
) -> SteeringVectorReplacer:
    v  = torch.zeros(hidden_size, device="cuda", dtype=dtype)
    vi = torch.zeros(max_tokens,  device="cuda", dtype=torch.int32)
    return SteeringVectorReplacer(v, vi, max_tokens)


def set_active_tokens(rep: SteeringVectorReplacer, indices: list[int]) -> None:
    n = len(indices)
    rep.token_indices[:n].copy_(
        torch.tensor(indices, dtype=torch.int32, device="cuda")
    )
    rep.n_valid_buf.fill_(n)


def reference_replace(
    x:             torch.Tensor,   # (n_rows, hidden_size)
    v:             torch.Tensor,   # (hidden_size,)
    token_indices: list[int],
    vec_indices:   torch.Tensor,   # (n_rows,) int32
) -> torch.Tensor:
    x = x.clone()
    for row in token_indices:
        if vec_indices[row].item() != 0:
            x[row] = v.to(x.dtype)
    return x


# ── kernel correctness ────────────────────────────────────────────────────────

class TestKernelCorrectness:

    @pytest.mark.parametrize("n_tokens,hidden_size", [
        (1,   64),
        (4,   64),
        (8,   512),
        (16,  4096),
        (8,   100),   # non-power-of-2 hidden
    ])
    def test_matches_pytorch_reference(self, n_tokens, hidden_size):
        torch.manual_seed(0)
        rep = make_replacer(hidden_size=hidden_size, max_tokens=n_tokens)

        v  = torch.randn(hidden_size, device="cuda", dtype=torch.float32)
        vi = torch.ones(n_tokens, device="cuda", dtype=torch.int32)

        rep.load_vector(v)
        rep.load_vec_indices(vi)
        set_active_tokens(rep, list(range(n_tokens)))

        x     = torch.randn(n_tokens, hidden_size, device="cuda", dtype=torch.float32)
        x_ref = reference_replace(x, v, list(range(n_tokens)), vi)

        rep.run(x)

        torch.testing.assert_close(x.float(), x_ref.float(), atol=1e-5, rtol=0,
            msg=f"Mismatch for n_tokens={n_tokens}, hidden={hidden_size}")

    def test_replaces_with_exact_vector(self):
        """After replacement, steered rows must equal v exactly."""
        hidden, n_tok = 64, 4
        rep = make_replacer(hidden_size=hidden, max_tokens=n_tok)

        v  = torch.ones(hidden, device="cuda", dtype=torch.float32) * 3.14
        vi = torch.ones(n_tok, device="cuda", dtype=torch.int32)

        rep.load_vector(v)
        rep.load_vec_indices(vi)
        set_active_tokens(rep, list(range(n_tok)))

        x = torch.zeros(n_tok, hidden, device="cuda", dtype=torch.float32)
        rep.run(x)

        for row in range(n_tok):
            torch.testing.assert_close(x[row], v, atol=1e-6, rtol=0,
                msg=f"Row {row} was not replaced with v")

    def test_hot_swap_vector(self):
        """Loading a new vector changes the replacement on the next run."""
        hidden, n_tok = 64, 2
        rep = make_replacer(hidden_size=hidden, max_tokens=n_tok)
        vi  = torch.ones(n_tok, device="cuda", dtype=torch.int32)
        set_active_tokens(rep, list(range(n_tok)))
        rep.load_vec_indices(vi)

        v1 = torch.ones(hidden, device="cuda", dtype=torch.float32)
        rep.load_vector(v1)
        x1 = torch.zeros(n_tok, hidden, device="cuda", dtype=torch.float32)
        rep.run(x1)

        v2 = torch.ones(hidden, device="cuda", dtype=torch.float32) * (-1.0)
        rep.load_vector(v2)
        x2 = torch.zeros(n_tok, hidden, device="cuda", dtype=torch.float32)
        rep.run(x2)

        torch.testing.assert_close(x1, torch.ones_like(x1),    atol=1e-6, rtol=0)
        torch.testing.assert_close(x2, torch.full_like(x2, -1.0), atol=1e-6, rtol=0)


# ── vec_indices gating ────────────────────────────────────────────────────────

class TestVecIndicesGating:

    def test_zero_flag_skips_token(self):
        """vec_indices=0 must leave the row unchanged."""
        hidden, n_tok = 64, 4
        rep = make_replacer(hidden_size=hidden, max_tokens=n_tok)

        rep.load_vector(torch.ones(hidden, device="cuda", dtype=torch.float32))
        rep.load_vec_indices(torch.zeros(n_tok, device="cuda", dtype=torch.int32))
        set_active_tokens(rep, list(range(n_tok)))

        x        = torch.randn(n_tok, hidden, device="cuda", dtype=torch.float32)
        x_before = x.clone()
        rep.run(x)

        torch.testing.assert_close(x, x_before,
            msg="Tokens with vec_indices=0 must not be modified")

    def test_nonzero_flag_replaces_token(self):
        """vec_indices != 0 must replace the row with v."""
        hidden, n_tok = 64, 4
        rep = make_replacer(hidden_size=hidden, max_tokens=n_tok)

        v  = torch.randn(hidden, device="cuda", dtype=torch.float32)
        vi = torch.ones(n_tok, device="cuda", dtype=torch.int32)

        rep.load_vector(v)
        rep.load_vec_indices(vi)
        set_active_tokens(rep, list(range(n_tok)))

        x = torch.randn(n_tok, hidden, device="cuda", dtype=torch.float32)
        rep.run(x)

        for row in range(n_tok):
            torch.testing.assert_close(x[row], v, atol=1e-5, rtol=0)

    def test_mixed_flags_steers_only_marked_rows(self):
        """Only rows with vec_indices != 0 should be replaced."""
        hidden, n_tok = 64, 6
        rep = make_replacer(hidden_size=hidden, max_tokens=n_tok)

        v  = torch.ones(hidden, device="cuda", dtype=torch.float32)
        vi = torch.tensor([1, 0, 1, 0, 1, 0], device="cuda", dtype=torch.int32)

        rep.load_vector(v)
        rep.load_vec_indices(vi)
        set_active_tokens(rep, list(range(n_tok)))

        x        = torch.randn(n_tok, hidden, device="cuda", dtype=torch.float32)
        x_before = x.clone()
        rep.run(x)

        for row in [1, 3, 5]:
            torch.testing.assert_close(x[row], x_before[row],
                msg=f"Row {row} has vec_indices=0 but was modified")
        for row in [0, 2, 4]:
            torch.testing.assert_close(x[row], v, atol=1e-5, rtol=0,
                msg=f"Row {row} has vec_indices=1 but was not replaced")

    def test_any_nonzero_value_replaces(self):
        """Any nonzero vec_indices value (1, 7, 255) should trigger replacement."""
        hidden, n_tok = 64, 3
        torch.manual_seed(5)
        v      = torch.randn(hidden, device="cuda", dtype=torch.float32)
        x_orig = torch.randn(n_tok, hidden, device="cuda", dtype=torch.float32)

        results = []
        for flag_val in [1, 7, 255]:
            rep = make_replacer(hidden_size=hidden, max_tokens=n_tok)
            vi  = torch.full((n_tok,), flag_val, dtype=torch.int32, device="cuda")
            rep.load_vector(v.clone())
            rep.load_vec_indices(vi)
            set_active_tokens(rep, list(range(n_tok)))
            x = x_orig.clone()
            rep.run(x)
            results.append(x.clone())

        torch.testing.assert_close(results[0], results[1], atol=1e-6, rtol=0)
        torch.testing.assert_close(results[0], results[2], atol=1e-6, rtol=0)


# ── n_valid_buf gating ────────────────────────────────────────────────────────

class TestNValidBuf:

    def test_n_valid_zero_leaves_x_unchanged(self):
        """n_valid_buf=0 → no tokens processed, x must be untouched."""
        hidden, n_tok = 64, 4
        rep = make_replacer(hidden_size=hidden, max_tokens=n_tok)

        rep.load_vector(torch.ones(hidden, device="cuda", dtype=torch.float32))
        rep.load_vec_indices(torch.ones(n_tok, device="cuda", dtype=torch.int32))

        rep.token_indices[:n_tok].copy_(torch.arange(n_tok, dtype=torch.int32, device="cuda"))
        rep.n_valid_buf.fill_(0)

        x        = torch.randn(n_tok, hidden, device="cuda", dtype=torch.float32)
        x_before = x.clone()
        rep.run(x)

        torch.testing.assert_close(x, x_before,
            msg="n_valid_buf=0 must leave x completely unchanged")

    def test_only_n_valid_tokens_processed(self):
        """Only the first n_valid entries in token_indices should be replaced."""
        hidden, n_tok = 64, 4
        rep = make_replacer(hidden_size=hidden, max_tokens=n_tok)

        v  = torch.ones(hidden, device="cuda", dtype=torch.float32)
        vi = torch.ones(n_tok, device="cuda", dtype=torch.int32)

        rep.load_vector(v)
        rep.load_vec_indices(vi)
        rep.token_indices[:n_tok].copy_(
            torch.arange(n_tok, dtype=torch.int32, device="cuda")
        )
        rep.n_valid_buf.fill_(2)   # only process first 2 tokens

        x = torch.zeros(n_tok, hidden, device="cuda", dtype=torch.float32)
        rep.run(x)

        torch.testing.assert_close(x[0], v, atol=1e-6, rtol=0)
        torch.testing.assert_close(x[1], v, atol=1e-6, rtol=0)
        torch.testing.assert_close(x[2], torch.zeros(hidden, device="cuda"), atol=0, rtol=0)
        torch.testing.assert_close(x[3], torch.zeros(hidden, device="cuda"), atol=0, rtol=0)


# ── unselected rows are never modified ────────────────────────────────────────

class TestUnselectedRowsUnchanged:

    def test_unselected_rows_not_modified(self):
        """Rows not listed in token_indices must be bit-identical after run()."""
        hidden    = 64
        n_rows    = 10
        n_steer   = 3
        rep = make_replacer(hidden_size=hidden, max_tokens=n_steer)

        rep.load_vector(torch.ones(hidden, device="cuda", dtype=torch.float32))
        rep.load_vec_indices(torch.ones(n_steer, device="cuda", dtype=torch.int32))
        steer_rows = [0, 4, 7]
        set_active_tokens(rep, steer_rows)

        x        = torch.randn(n_rows, hidden, device="cuda", dtype=torch.float32)
        x_before = x.clone()
        rep.run(x)

        for row in range(n_rows):
            if row not in steer_rows:
                torch.testing.assert_close(x[row], x_before[row],
                    msg=f"Row {row} was modified but should be untouched")
