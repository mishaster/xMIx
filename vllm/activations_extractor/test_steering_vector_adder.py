# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Tests for SteeringVectorAdder and add_scaled_summed_vectors_kernel
in write_activations.py.

Run with:
    pytest vllm/activations_extractor/test_steering_vector_adder.py -v
"""

import pytest
import torch

from vllm.activations_extractor.write_activations import SteeringVectorAdder


# ── helpers ───────────────────────────────────────────────────────────────────

def make_adder(
    max_vecs:    int = 4,
    hidden_size: int = 64,
    max_tokens:  int = 16,
    dtype:       torch.dtype = torch.float32,
) -> SteeringVectorAdder:
    device = "cuda"
    r                = torch.zeros(max_vecs, hidden_size, dtype=dtype,        device=device)
    scales           = torch.zeros(max_tokens, max_vecs, dtype=torch.float32, device=device)
    vec_indices      = torch.zeros(max_tokens, max_vecs, dtype=torch.int32,   device=device)
    n_vecs_per_token = torch.zeros(max_tokens,           dtype=torch.int32,   device=device)
    return SteeringVectorAdder(r, scales, vec_indices, n_vecs_per_token)


def set_active_tokens(adder: SteeringVectorAdder, indices: list[int]) -> None:
    """Manually populate token_indices / n_valid_buf (normally done by EOLTokenDetector)."""
    n = len(indices)
    adder.token_indices[:n].copy_(
        torch.tensor(indices, dtype=torch.int32, device="cuda")
    )
    adder.n_valid_buf.fill_(n)


def reference_steer(
    x:               torch.Tensor,   # (n_rows, hidden_size)
    r:               torch.Tensor,   # (max_vecs, hidden_size)
    scales:          torch.Tensor,   # (max_tokens, max_vecs) float32
    token_indices:   list[int],
    vec_indices:     torch.Tensor,   # (max_tokens, max_vecs) int32
    n_vecs_per_token: torch.Tensor,  # (max_tokens,) int32
) -> torch.Tensor:
    """Pure PyTorch reference: for each selected token add scaled vector sum."""
    x = x.clone().float()
    for sel_idx, row in enumerate(token_indices):
        n_vecs = n_vecs_per_token[sel_idx].item()
        combined = torch.zeros(x.shape[-1], device=x.device, dtype=torch.float32)
        for v in range(n_vecs):
            scale   = scales[sel_idx, v].item()
            vec_idx = vec_indices[sel_idx, v].item()
            combined += scale * r[vec_idx].float()
        x[row] += combined
    return x


# ── kernel correctness ────────────────────────────────────────────────────────

class TestKernelCorrectness:

    @pytest.mark.parametrize("n_tokens,hidden_size,max_vecs", [
        (1,  64,  1),
        (4,  64,  2),
        (8,  512, 4),
        (16, 4096, 1),   # realistic hidden size
    ])
    def test_matches_pytorch_reference(self, n_tokens, hidden_size, max_vecs):
        torch.manual_seed(0)
        adder = make_adder(max_vecs=max_vecs, hidden_size=hidden_size,
                           max_tokens=n_tokens)

        r      = torch.randn(max_vecs, hidden_size, device="cuda", dtype=torch.float32)
        scales = torch.randn(n_tokens, max_vecs,    device="cuda", dtype=torch.float32)
        vi     = torch.zeros(n_tokens, max_vecs,    device="cuda", dtype=torch.int32)
        nvpt   = torch.ones(n_tokens,               device="cuda", dtype=torch.int32)

        # Each token uses vector 0 only
        adder.load_vectors(r)
        adder.load_scales(scales)
        adder.load_vec_indices(vi, nvpt)
        set_active_tokens(adder, list(range(n_tokens)))

        x     = torch.randn(n_tokens, hidden_size, device="cuda", dtype=torch.float32)
        x_ref = reference_steer(x, r, scales, list(range(n_tokens)), vi, nvpt)

        adder.run(x)

        torch.testing.assert_close(x.float(), x_ref, atol=1e-4, rtol=1e-4,
            msg=f"Mismatch for n_tokens={n_tokens}, hidden={hidden_size}")

    def test_multiple_vectors_summed_correctly(self):
        """n_vecs_per_token=2: combined = scale0*r[vi0] + scale1*r[vi1]."""
        hidden, max_vecs, n_tok = 64, 2, 4
        adder = make_adder(max_vecs=max_vecs, hidden_size=hidden, max_tokens=n_tok)

        r      = torch.randn(max_vecs, hidden, device="cuda", dtype=torch.float32)
        scales = torch.ones(n_tok, max_vecs,  device="cuda", dtype=torch.float32)
        vi     = torch.tensor([[0, 1]] * n_tok, device="cuda", dtype=torch.int32)
        nvpt   = torch.full((n_tok,), 2,        device="cuda", dtype=torch.int32)

        adder.load_vectors(r)
        adder.load_scales(scales)
        adder.load_vec_indices(vi, nvpt)
        set_active_tokens(adder, list(range(n_tok)))

        x     = torch.zeros(n_tok, hidden, device="cuda", dtype=torch.float32)
        x_ref = reference_steer(x, r, scales, list(range(n_tok)), vi, nvpt)
        adder.run(x)

        torch.testing.assert_close(x.float(), x_ref, atol=1e-4, rtol=1e-4)

    def test_negative_scale_subtracts(self):
        """Negative scale should subtract the vector."""
        hidden, max_vecs, n_tok = 64, 1, 2
        adder = make_adder(max_vecs=max_vecs, hidden_size=hidden, max_tokens=n_tok)

        r      = torch.ones(max_vecs, hidden,  device="cuda", dtype=torch.float32)
        scales = torch.full((n_tok, max_vecs), -1.0, device="cuda", dtype=torch.float32)
        vi     = torch.zeros(n_tok, max_vecs,  device="cuda", dtype=torch.int32)
        nvpt   = torch.ones(n_tok,             device="cuda", dtype=torch.int32)

        adder.load_vectors(r)
        adder.load_scales(scales)
        adder.load_vec_indices(vi, nvpt)
        set_active_tokens(adder, list(range(n_tok)))

        x = torch.zeros(n_tok, hidden, device="cuda", dtype=torch.float32)
        adder.run(x)

        # x was 0, r=1, scale=-1 → result should be -1 everywhere
        torch.testing.assert_close(x, torch.full_like(x, -1.0), atol=1e-5, rtol=0)


# ── n_vecs_per_token = 0 vs 1 ─────────────────────────────────────────────────

class TestNVecsPerToken:

    def test_zero_vecs_skips_token(self):
        """n_vecs_per_token=0 → token must not be modified even if it is selected."""
        hidden, max_vecs, n_tok = 64, 2, 4
        adder = make_adder(max_vecs=max_vecs, hidden_size=hidden, max_tokens=n_tok)

        r      = torch.ones(max_vecs, hidden, device="cuda", dtype=torch.float32)
        scales = torch.ones(n_tok, max_vecs,  device="cuda", dtype=torch.float32)
        vi     = torch.zeros(n_tok, max_vecs, device="cuda", dtype=torch.int32)
        nvpt   = torch.zeros(n_tok,           device="cuda", dtype=torch.int32)  # all 0

        adder.load_vectors(r)
        adder.load_scales(scales)
        adder.load_vec_indices(vi, nvpt)
        set_active_tokens(adder, list(range(n_tok)))

        x        = torch.randn(n_tok, hidden, device="cuda", dtype=torch.float32)
        x_before = x.clone()
        adder.run(x)

        torch.testing.assert_close(x, x_before,
            msg="Tokens with n_vecs=0 must not be modified")

    def test_one_vec_applies_steering(self):
        """n_vecs_per_token=1 → exactly one vector added with given scale."""
        hidden, max_vecs, n_tok = 64, 2, 4
        adder = make_adder(max_vecs=max_vecs, hidden_size=hidden, max_tokens=n_tok)

        r      = torch.eye(max_vecs, hidden,  device="cuda", dtype=torch.float32)
        scales = torch.ones(n_tok, max_vecs,  device="cuda", dtype=torch.float32) * 2.0
        vi     = torch.zeros(n_tok, max_vecs, device="cuda", dtype=torch.int32)
        nvpt   = torch.ones(n_tok,            device="cuda", dtype=torch.int32)  # use 1 vec

        adder.load_vectors(r)
        adder.load_scales(scales)
        adder.load_vec_indices(vi, nvpt)
        set_active_tokens(adder, list(range(n_tok)))

        x = torch.zeros(n_tok, hidden, device="cuda", dtype=torch.float32)
        adder.run(x)

        # scale=2, r[0]=e_0 (first basis vector), so x[:,0] should be 2, rest 0
        expected = torch.zeros_like(x)
        expected[:, 0] = 2.0
        torch.testing.assert_close(x, expected, atol=1e-5, rtol=0)

    def test_mixed_zero_and_one_per_token(self):
        """Some tokens have n_vecs=0 (no steering), others n_vecs=1 (steered)."""
        hidden, max_vecs, n_tok = 64, 1, 4
        adder = make_adder(max_vecs=max_vecs, hidden_size=hidden, max_tokens=n_tok)

        r      = torch.ones(max_vecs, hidden, device="cuda", dtype=torch.float32)
        scales = torch.ones(n_tok, max_vecs,  device="cuda", dtype=torch.float32)
        vi     = torch.zeros(n_tok, max_vecs, device="cuda", dtype=torch.int32)
        # tokens 0 and 2 steered, tokens 1 and 3 not
        nvpt   = torch.tensor([1, 0, 1, 0], device="cuda", dtype=torch.int32)

        adder.load_vectors(r)
        adder.load_scales(scales)
        adder.load_vec_indices(vi, nvpt)
        set_active_tokens(adder, list(range(n_tok)))

        x = torch.zeros(n_tok, hidden, device="cuda", dtype=torch.float32)
        adder.run(x)

        # Steered tokens (0, 2) should be all-ones; unsteered (1, 3) should be zero
        torch.testing.assert_close(x[0], torch.ones(hidden, device="cuda"), atol=1e-5, rtol=0)
        torch.testing.assert_close(x[1], torch.zeros(hidden, device="cuda"), atol=1e-5, rtol=0)
        torch.testing.assert_close(x[2], torch.ones(hidden, device="cuda"), atol=1e-5, rtol=0)
        torch.testing.assert_close(x[3], torch.zeros(hidden, device="cuda"), atol=1e-5, rtol=0)


# ── n_valid_buf gates which tokens are processed ─────────────────────────────

class TestNValidBuf:

    def test_n_valid_zero_leaves_x_unchanged(self):
        """n_valid_buf=0 → no tokens processed, x must be untouched."""
        hidden, max_vecs, n_tok = 64, 1, 4
        adder = make_adder(max_vecs=max_vecs, hidden_size=hidden, max_tokens=n_tok)

        adder.load_vectors(torch.ones(max_vecs, hidden, device="cuda"))
        adder.load_scales(torch.ones(n_tok, max_vecs,   device="cuda", dtype=torch.float32))
        adder.load_vec_indices(
            torch.zeros(n_tok, max_vecs, device="cuda", dtype=torch.int32),
            torch.ones(n_tok,            device="cuda", dtype=torch.int32),
        )
        # Set token_indices but leave n_valid_buf = 0
        adder.token_indices[:n_tok].copy_(
            torch.arange(n_tok, dtype=torch.int32, device="cuda")
        )
        adder.n_valid_buf.fill_(0)

        x        = torch.randn(n_tok, hidden, device="cuda", dtype=torch.float32)
        x_before = x.clone()
        adder.run(x)

        torch.testing.assert_close(x, x_before,
            msg="n_valid_buf=0 must leave x completely unchanged")

    def test_only_n_valid_tokens_processed(self):
        """Only the first n_valid entries in token_indices should be steered."""
        hidden, max_vecs, n_tok = 64, 1, 4
        adder = make_adder(max_vecs=max_vecs, hidden_size=hidden, max_tokens=n_tok)

        r      = torch.ones(max_vecs, hidden, device="cuda", dtype=torch.float32)
        scales = torch.ones(n_tok, max_vecs,  device="cuda", dtype=torch.float32)
        vi     = torch.zeros(n_tok, max_vecs, device="cuda", dtype=torch.int32)
        nvpt   = torch.ones(n_tok,            device="cuda", dtype=torch.int32)

        adder.load_vectors(r)
        adder.load_scales(scales)
        adder.load_vec_indices(vi, nvpt)

        # Only activate the first 2 tokens out of 4
        adder.token_indices[:n_tok].copy_(
            torch.arange(n_tok, dtype=torch.int32, device="cuda")
        )
        adder.n_valid_buf.fill_(2)

        x        = torch.zeros(n_tok, hidden, device="cuda", dtype=torch.float32)
        x_before = x.clone()
        adder.run(x)

        # Rows 0 and 1 steered → all ones; rows 2 and 3 untouched → all zeros
        torch.testing.assert_close(x[0], torch.ones(hidden,  device="cuda"), atol=1e-5, rtol=0)
        torch.testing.assert_close(x[1], torch.ones(hidden,  device="cuda"), atol=1e-5, rtol=0)
        torch.testing.assert_close(x[2], torch.zeros(hidden, device="cuda"), atol=1e-5, rtol=0)
        torch.testing.assert_close(x[3], torch.zeros(hidden, device="cuda"), atol=1e-5, rtol=0)


# ── unselected rows are never modified ────────────────────────────────────────

class TestUnselectedRowsUnchanged:

    def test_unselected_rows_not_modified(self):
        """Rows not listed in token_indices must be bit-identical after forward()."""
        hidden, max_vecs = 64, 1
        n_rows, n_steer  = 10, 3
        adder = make_adder(max_vecs=max_vecs, hidden_size=hidden, max_tokens=n_steer)

        adder.load_vectors(torch.ones(max_vecs, hidden, device="cuda"))
        adder.load_scales(torch.ones(n_steer, max_vecs, device="cuda", dtype=torch.float32))
        adder.load_vec_indices(
            torch.zeros(n_steer, max_vecs, device="cuda", dtype=torch.int32),
            torch.ones(n_steer,            device="cuda", dtype=torch.int32),
        )
        steer_rows = [0, 4, 7]
        set_active_tokens(adder, steer_rows)

        x        = torch.randn(n_rows, hidden, device="cuda", dtype=torch.float32)
        x_before = x.clone()
        adder.run(x)

        for row in range(n_rows):
            if row not in steer_rows:
                torch.testing.assert_close(x[row], x_before[row],
                    msg=f"Row {row} was modified but should be untouched")
