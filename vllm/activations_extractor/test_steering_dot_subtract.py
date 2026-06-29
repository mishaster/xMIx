# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Tests for SteeringVectorDotSubtractNormalized and
fused_subtract_projection_selected_kernel in write_activations.py.

Run with:
    pytest vllm/activations_extractor/test_steering_dot_subtract.py -v
"""

import pytest
import torch

from vllm.activations_extractor.write_activations import (
    SteeringVectorDotSubtractNormalized,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def make_subtractor(
    hidden_size: int = 64,
    max_tokens:  int = 16,
    dtype:       torch.dtype = torch.float32,
) -> SteeringVectorDotSubtractNormalized:
    return SteeringVectorDotSubtractNormalized(
        hidden_size=hidden_size,
        max_tokens=max_tokens,
        dtype=dtype,
        device="cuda",
    )


def set_active_tokens(
    sub: SteeringVectorDotSubtractNormalized,
    indices: list[int],
) -> None:
    """Populate token_indices / n_valid_buf (normally done by EOLTokenDetector)."""
    n = len(indices)
    sub.token_indices[:n].copy_(
        torch.tensor(indices, dtype=torch.int32, device="cuda")
    )
    sub.n_valid_buf.fill_(n)


def reference_subtract_projection(
    x:             torch.Tensor,   # (n_rows, hidden_size)
    r_raw:         torch.Tensor,   # (1, hidden_size) — unnormalised
    token_indices: list[int],
    vec_indices:   torch.Tensor,   # (n_rows,) int32
) -> torch.Tensor:
    """Pure PyTorch reference: normalise r, then subtract projection per token."""
    x      = x.clone().float()
    r_norm = r_raw.float().flatten()
    r_norm = r_norm / r_norm.norm()

    for row in token_indices:
        if vec_indices[row].item() == 0:
            continue
        dot    = (x[row] * r_norm).sum()
        x[row] = x[row] - dot * r_norm

    return x


# ── kernel correctness ────────────────────────────────────────────────────────

class TestKernelCorrectness:

    @pytest.mark.parametrize("n_tokens,hidden_size", [
        (1,   64),
        (4,   64),
        (8,   512),
        (16,  4096),
        (8,   100),    # non-power-of-2 hidden
    ])
    def test_matches_pytorch_reference(self, n_tokens, hidden_size):
        torch.manual_seed(0)
        sub = make_subtractor(hidden_size=hidden_size, max_tokens=n_tokens)

        r_raw = torch.randn(1, hidden_size, device="cuda", dtype=torch.float32)
        x     = torch.randn(n_tokens, hidden_size, device="cuda", dtype=torch.float32)

        sub.load_vector(r_raw)
        # Enable all tokens
        vi = torch.ones(n_tokens, dtype=torch.int32, device="cuda")
        sub.load_vec_indices(vi)
        set_active_tokens(sub, list(range(n_tokens)))

        x_ref = reference_subtract_projection(x, r_raw, list(range(n_tokens)), vi)
        sub.run(x)

        torch.testing.assert_close(x.float(), x_ref.float(), atol=1e-4, rtol=1e-4,
            msg=f"Mismatch for n_tokens={n_tokens}, hidden={hidden_size}")

    def test_projection_removed(self):
        """After subtraction, dot(x_result, r_norm) must be ~0 for each steered token."""
        hidden, n_tok = 64, 8
        sub = make_subtractor(hidden_size=hidden, max_tokens=n_tok)

        r_raw = torch.randn(1, hidden, device="cuda", dtype=torch.float32)
        x     = torch.randn(n_tok, hidden, device="cuda", dtype=torch.float32)

        sub.load_vector(r_raw)
        sub.load_vec_indices(torch.ones(n_tok, dtype=torch.int32, device="cuda"))
        set_active_tokens(sub, list(range(n_tok)))
        sub.run(x)

        r_norm = (r_raw / r_raw.norm()).flatten()
        residual_dots = (x.float() @ r_norm).abs()
        assert residual_dots.max().item() < 1e-4, \
            f"Projection not fully removed; max residual dot = {residual_dots.max().item()}"

    def test_x_norm_does_not_increase(self):
        """Subtracting a projection should never increase the norm of a token."""
        hidden, n_tok = 64, 16
        sub = make_subtractor(hidden_size=hidden, max_tokens=n_tok)

        r_raw = torch.randn(1, hidden, device="cuda", dtype=torch.float32)
        x     = torch.randn(n_tok, hidden, device="cuda", dtype=torch.float32)
        norms_before = x.norm(dim=-1).clone()

        sub.load_vector(r_raw)
        sub.load_vec_indices(torch.ones(n_tok, dtype=torch.int32, device="cuda"))
        set_active_tokens(sub, list(range(n_tok)))
        sub.run(x)

        norms_after = x.norm(dim=-1)
        assert (norms_after <= norms_before + 1e-5).all(), \
            "Projection subtraction increased a token's norm"


# ── normalisation ─────────────────────────────────────────────────────────────

class TestNormalisation:

    def test_unnormalised_r_gives_same_result_as_normalised(self):
        """load_vector must normalise r; passing 2*r should give the same result as r."""
        hidden, n_tok = 64, 4
        torch.manual_seed(1)

        r_unit = torch.randn(1, hidden, device="cuda", dtype=torch.float32)
        r_unit = r_unit / r_unit.norm()
        x_orig = torch.randn(n_tok, hidden, device="cuda", dtype=torch.float32)

        vi = torch.ones(n_tok, dtype=torch.int32, device="cuda")

        # Run with unit r
        sub_a = make_subtractor(hidden_size=hidden, max_tokens=n_tok)
        sub_a.load_vector(r_unit.clone())
        sub_a.load_vec_indices(vi)
        set_active_tokens(sub_a, list(range(n_tok)))
        x_a = x_orig.clone()
        sub_a.run(x_a)

        # Run with 2 * r — load_vector must normalise it to the same unit vector
        sub_b = make_subtractor(hidden_size=hidden, max_tokens=n_tok)
        sub_b.load_vector(r_unit.clone() * 2.0)
        sub_b.load_vec_indices(vi)
        set_active_tokens(sub_b, list(range(n_tok)))
        x_b = x_orig.clone()
        sub_b.run(x_b)

        torch.testing.assert_close(x_a, x_b, atol=1e-5, rtol=1e-5,
            msg="load_vector does not normalise r: scaling r changed the result")

    def test_stored_r_is_unit_norm(self):
        """After load_vector, self.r must have norm ≈ 1."""
        hidden = 64
        sub = make_subtractor(hidden_size=hidden, max_tokens=4)
        r_raw = torch.randn(1, hidden, device="cuda", dtype=torch.float32) * 5.0
        sub.load_vector(r_raw)
        assert abs(sub.r.norm().item() - 1.0) < 1e-5, \
            f"Stored r has norm {sub.r.norm().item():.6f}, expected 1.0"


# ── vec_indices gating ────────────────────────────────────────────────────────

class TestVecIndicesGating:

    def test_zero_flag_skips_token(self):
        """vec_indices=0 for a token must leave that row unchanged."""
        hidden, n_tok = 64, 4
        sub = make_subtractor(hidden_size=hidden, max_tokens=n_tok)

        sub.load_vector(torch.randn(1, hidden, device="cuda", dtype=torch.float32))
        sub.load_vec_indices(torch.zeros(n_tok, dtype=torch.int32, device="cuda"))
        set_active_tokens(sub, list(range(n_tok)))

        x        = torch.randn(n_tok, hidden, device="cuda", dtype=torch.float32)
        x_before = x.clone()
        sub.run(x)

        torch.testing.assert_close(x, x_before,
            msg="Tokens with vec_indices=0 must not be modified")

    def test_nonzero_flag_steers_token(self):
        """vec_indices=1 for a token must modify that row."""
        hidden, n_tok = 64, 4
        sub = make_subtractor(hidden_size=hidden, max_tokens=n_tok)

        r_raw = torch.randn(1, hidden, device="cuda", dtype=torch.float32)
        x     = torch.randn(n_tok, hidden, device="cuda", dtype=torch.float32)
        x_before = x.clone()

        sub.load_vector(r_raw)
        sub.load_vec_indices(torch.ones(n_tok, dtype=torch.int32, device="cuda"))
        set_active_tokens(sub, list(range(n_tok)))
        sub.run(x)

        assert not torch.allclose(x, x_before), \
            "Tokens with vec_indices=1 were not modified"

    def test_mixed_flags_steers_only_marked_tokens(self):
        """Only rows where vec_indices != 0 should be modified."""
        hidden, n_tok = 64, 6
        sub = make_subtractor(hidden_size=hidden, max_tokens=n_tok)

        r_raw = torch.randn(1, hidden, device="cuda", dtype=torch.float32)
        # Alternate: steer rows 0, 2, 4 — skip rows 1, 3, 5
        vi = torch.tensor([1, 0, 1, 0, 1, 0], dtype=torch.int32, device="cuda")
        steer_rows = [0, 2, 4]
        skip_rows  = [1, 3, 5]

        sub.load_vector(r_raw)
        sub.load_vec_indices(vi)
        set_active_tokens(sub, list(range(n_tok)))

        x        = torch.randn(n_tok, hidden, device="cuda", dtype=torch.float32)
        x_before = x.clone()
        x_ref    = reference_subtract_projection(x, r_raw, list(range(n_tok)), vi)
        sub.run(x)

        for row in skip_rows:
            torch.testing.assert_close(x[row], x_before[row],
                msg=f"Row {row} has vec_indices=0 but was modified")

        for row in steer_rows:
            torch.testing.assert_close(x[row].float(), x_ref[row].float(),
                atol=1e-4, rtol=1e-4,
                msg=f"Row {row} has vec_indices=1 but result is incorrect")

    def test_vec_indices_nonzero_value_does_not_matter(self):
        """Any nonzero value in vec_indices (1, 5, 255) should steer identically."""
        hidden, n_tok = 64, 4
        torch.manual_seed(3)
        r_raw  = torch.randn(1, hidden, device="cuda", dtype=torch.float32)
        x_orig = torch.randn(n_tok, hidden, device="cuda", dtype=torch.float32)

        results = []
        for flag_value in [1, 5, 255]:
            vi  = torch.full((n_tok,), flag_value, dtype=torch.int32, device="cuda")
            sub = make_subtractor(hidden_size=hidden, max_tokens=n_tok)
            sub.load_vector(r_raw.clone())
            sub.load_vec_indices(vi)
            set_active_tokens(sub, list(range(n_tok)))
            x = x_orig.clone()
            sub.run(x)
            results.append(x.clone())

        torch.testing.assert_close(results[0], results[1], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(results[0], results[2], atol=1e-5, rtol=1e-5)
