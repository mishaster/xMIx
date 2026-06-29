# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Tests for SteeringConceptReplace and concept_replace_kernel in claude_app.py.

Run with:
    pytest vllm/activations_extractor/test_concept_replace.py -v
"""

import pytest
import torch

from vllm.activations_extractor.claude_app import SteeringConceptReplace


# ── helpers ───────────────────────────────────────────────────────────────────

def make_steerer(
    hidden_size: int = 64,
    max_tokens:  int = 16,
    dtype:       torch.dtype = torch.float32,
) -> SteeringConceptReplace:
    h1_raw = torch.randn(hidden_size, device="cuda", dtype=dtype)
    h1 = h1_raw / h1_raw.norm()   # caller must normalize h1
    h2 = torch.randn(hidden_size,  device="cuda", dtype=dtype)
    vi = torch.zeros(max_tokens,   device="cuda", dtype=torch.int32)
    return SteeringConceptReplace(h1, h2, vi, max_tokens)


def set_active_tokens(s: SteeringConceptReplace, indices: list[int]) -> None:
    n = len(indices)
    s.token_indices[:n].copy_(
        torch.tensor(indices, dtype=torch.int32, device="cuda")
    )
    s.n_valid_buf.fill_(n)


def reference_concept_replace(
    x:             torch.Tensor,   # (n_rows, hidden_size)
    h1:            torch.Tensor,   # (hidden_size,) — must be unit-norm
    h2:            torch.Tensor,   # (hidden_size,)
    token_indices: list[int],
    vec_indices:   torch.Tensor,   # (n_rows,) int32
) -> torch.Tensor:
    """Pure PyTorch reference: h += dot(h, h1) * (h2 - h1).  h1 assumed unit-norm."""
    x         = x.clone().float()
    h1_f      = h1.float()
    h2_f      = h2.float()
    direction = h2_f - h1_f

    for row in token_indices:
        if vec_indices[row].item() != 0:
            lam     = (x[row] * h1_f).sum()
            x[row] += lam * direction

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
        s = make_steerer(hidden_size=hidden_size, max_tokens=n_tokens)

        h1_raw = torch.randn(hidden_size, device="cuda", dtype=torch.float32)
        h1     = h1_raw / h1_raw.norm()   # normalize before passing
        h2     = torch.randn(hidden_size, device="cuda", dtype=torch.float32)
        vi     = torch.ones(n_tokens, device="cuda", dtype=torch.int32)
        x      = torch.randn(n_tokens, hidden_size, device="cuda", dtype=torch.float32)

        s.load_vectors(h1, h2)
        s.load_vec_indices(vi)
        set_active_tokens(s, list(range(n_tokens)))

        x_ref = reference_concept_replace(x, h1, h2, list(range(n_tokens)), vi)
        s.run(x)

        torch.testing.assert_close(x.float(), x_ref.float(), atol=1e-4, rtol=1e-4,
            msg=f"Mismatch for n_tokens={n_tokens}, hidden={hidden_size}")

    def test_orthogonal_h1_gives_zero_shift(self):
        """If x is orthogonal to h1, lambda=0 and x must be unchanged."""
        hidden, n_tok = 64, 4
        s = make_steerer(hidden_size=hidden, max_tokens=n_tok)

        # h1 = e_0 (first basis vector)
        h1_raw = torch.zeros(hidden, device="cuda", dtype=torch.float32)
        h1_raw[0] = 1.0
        # h2 = e_1 (anything works)
        h2_raw = torch.zeros(hidden, device="cuda", dtype=torch.float32)
        h2_raw[1] = 1.0

        vi = torch.ones(n_tok, device="cuda", dtype=torch.int32)
        s.load_vectors(h1_raw, h2_raw)
        s.load_vec_indices(vi)
        set_active_tokens(s, list(range(n_tok)))

        # x has zero component along e_0
        x = torch.randn(n_tok, hidden, device="cuda", dtype=torch.float32)
        x[:, 0] = 0.0
        x_before = x.clone()
        s.run(x)

        torch.testing.assert_close(x, x_before, atol=1e-5, rtol=0,
            msg="x orthogonal to h1 must be unchanged")

    def test_projection_coefficient_uses_unit_h1(self):
        """lambda = dot(x, h1) where h1 is the unit-norm vector passed by caller."""
        hidden, n_tok = 64, 4
        torch.manual_seed(2)
        h1_raw = torch.randn(hidden, device="cuda", dtype=torch.float32)
        h1     = h1_raw / h1_raw.norm()   # normalize before passing
        h2     = torch.randn(hidden, device="cuda", dtype=torch.float32)
        x_orig = torch.randn(n_tok, hidden, device="cuda", dtype=torch.float32)

        s = make_steerer(hidden_size=hidden, max_tokens=n_tok)
        s.load_vectors(h1.clone(), h2.clone())
        s.load_vec_indices(torch.ones(n_tok, device="cuda", dtype=torch.int32))
        set_active_tokens(s, list(range(n_tok)))
        x = x_orig.clone()
        s.run(x)

        x_ref = reference_concept_replace(
            x_orig, h1, h2, list(range(n_tok)),
            torch.ones(n_tok, device="cuda", dtype=torch.int32),
        )
        torch.testing.assert_close(x.float(), x_ref.float(), atol=1e-4, rtol=1e-4,
            msg="Kernel result does not match reference")

    def test_h1_equals_h2_gives_zero_shift(self):
        """When h2 == h1, direction = 0 → x must be unchanged."""
        hidden, n_tok = 64, 4
        s = make_steerer(hidden_size=hidden, max_tokens=n_tok)
        v_raw = torch.randn(hidden, device="cuda", dtype=torch.float32)
        v = v_raw / v_raw.norm()   # normalize before passing
        vi = torch.ones(n_tok, device="cuda", dtype=torch.int32)
        s.load_vectors(v, v.clone())
        s.load_vec_indices(vi)
        set_active_tokens(s, list(range(n_tok)))

        x        = torch.randn(n_tok, hidden, device="cuda", dtype=torch.float32)
        x_before = x.clone()
        s.run(x)

        torch.testing.assert_close(x, x_before, atol=1e-5, rtol=0,
            msg="h1==h2 → zero direction → x must be unchanged")


# ── stored h1 is unit norm ────────────────────────────────────────────────────

class TestNormalisation:

    def test_stored_h1_is_unchanged(self):
        """load_vectors must store h1 as-is — no modification."""
        hidden = 64
        s = make_steerer(hidden_size=hidden, max_tokens=4)
        h1_raw = torch.randn(hidden, device="cuda", dtype=torch.float32) * 7.0
        h2_raw = torch.randn(hidden, device="cuda", dtype=torch.float32)
        s.load_vectors(h1_raw, h2_raw)
        torch.testing.assert_close(s.h1.float(), h1_raw.float(), atol=0, rtol=0,
            msg="Stored h1 was modified — must equal exactly what was passed")


# ── vec_indices gating ────────────────────────────────────────────────────────

class TestVecIndicesGating:

    def test_zero_flag_skips_token(self):
        hidden, n_tok = 64, 4
        s = make_steerer(hidden_size=hidden, max_tokens=n_tok)
        h1_raw = torch.randn(hidden, device="cuda", dtype=torch.float32)
        s.load_vectors(
            h1_raw / h1_raw.norm(),
            torch.randn(hidden, device="cuda", dtype=torch.float32),
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
        hidden, n_tok = 64, 6
        s = make_steerer(hidden_size=hidden, max_tokens=n_tok)
        h1_raw = torch.randn(hidden, device="cuda", dtype=torch.float32)
        h1     = h1_raw / h1_raw.norm()
        h2     = torch.randn(hidden, device="cuda", dtype=torch.float32)
        vi = torch.tensor([1, 0, 1, 0, 1, 0], device="cuda", dtype=torch.int32)

        s.load_vectors(h1, h2)
        s.load_vec_indices(vi)
        set_active_tokens(s, list(range(n_tok)))

        x        = torch.randn(n_tok, hidden, device="cuda", dtype=torch.float32)
        x_before = x.clone()
        x_ref    = reference_concept_replace(x, h1, h2, list(range(n_tok)), vi)
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
        h1_raw = torch.randn(hidden, device="cuda", dtype=torch.float32)
        s.load_vectors(
            h1_raw / h1_raw.norm(),
            torch.randn(hidden, device="cuda", dtype=torch.float32),
        )
        s.load_vec_indices(torch.ones(n_tok, device="cuda", dtype=torch.int32))
        s.token_indices[:n_tok].copy_(torch.arange(n_tok, dtype=torch.int32, device="cuda"))
        s.n_valid_buf.fill_(0)

        x        = torch.randn(n_tok, hidden, device="cuda", dtype=torch.float32)
        x_before = x.clone()
        s.run(x)

        torch.testing.assert_close(x, x_before,
            msg="n_valid_buf=0 must leave x completely unchanged")

    def test_only_n_valid_tokens_processed(self):
        hidden, n_tok = 64, 4
        s = make_steerer(hidden_size=hidden, max_tokens=n_tok)
        h1_raw = torch.randn(hidden, device="cuda", dtype=torch.float32)
        h1     = h1_raw / h1_raw.norm()
        h2     = torch.randn(hidden, device="cuda", dtype=torch.float32)
        s.load_vectors(h1, h2)
        s.load_vec_indices(torch.ones(n_tok, device="cuda", dtype=torch.int32))
        s.token_indices[:n_tok].copy_(torch.arange(n_tok, dtype=torch.int32, device="cuda"))
        s.n_valid_buf.fill_(2)

        x        = torch.randn(n_tok, hidden, device="cuda", dtype=torch.float32)
        x_before = x.clone()
        x_ref    = reference_concept_replace(x, h1, h2, [0, 1],
                       torch.ones(n_tok, device="cuda", dtype=torch.int32))
        s.run(x)

        torch.testing.assert_close(x[0].float(), x_ref[0].float(), atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(x[1].float(), x_ref[1].float(), atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(x[2], x_before[2], atol=0, rtol=0)
        torch.testing.assert_close(x[3], x_before[3], atol=0, rtol=0)


# ── unselected rows are never modified ────────────────────────────────────────

class TestUnselectedRowsUnchanged:

    def test_unselected_rows_not_modified(self):
        hidden  = 64
        n_rows  = 10
        n_steer = 3
        s = make_steerer(hidden_size=hidden, max_tokens=n_steer)

        h1_raw = torch.randn(hidden, device="cuda", dtype=torch.float32)
        s.load_vectors(
            h1_raw / h1_raw.norm(),
            torch.randn(hidden, device="cuda", dtype=torch.float32),
        )
        s.load_vec_indices(torch.ones(n_steer, device="cuda", dtype=torch.int32))
        steer_rows = [1, 5, 8]
        set_active_tokens(s, steer_rows)

        x        = torch.randn(n_rows, hidden, device="cuda", dtype=torch.float32)
        x_before = x.clone()
        s.run(x)

        for row in range(n_rows):
            if row not in steer_rows:
                torch.testing.assert_close(x[row], x_before[row],
                    msg=f"Row {row} was modified but should be untouched")
