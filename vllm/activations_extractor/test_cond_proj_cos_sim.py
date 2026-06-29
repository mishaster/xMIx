# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Tests for CondProjCosSim and its kernels in cond_kernels.py.

Run with:
    pytest vllm/activations_extractor/test_cond_proj_cos_sim.py -v
"""

import pytest
import torch

from vllm.activations_extractor.cond_kernels import (
    CondProjCosSim,
    _cond_proj_cos_sim_kernel,
)
from triton import next_power_of_2


# ── PyTorch reference ─────────────────────────────────────────────────────────

def reference_project(
    x:  torch.Tensor,   # (n_tokens, hidden_size)
    cp: torch.Tensor,   # (hidden_size, hidden_size)
) -> torch.Tensor:
    """tanh(x @ cp.T) — matches the cuBLAS step in _project()."""
    return torch.tanh(x.float() @ cp.float().T)


def reference_cos_sim(
    x:    torch.Tensor,   # (n_tokens, hidden_size)
    proj: torch.Tensor,   # (n_tokens, hidden_size)
) -> torch.Tensor:
    """Per-token cosine similarity between x and proj."""
    x    = x.float()
    proj = proj.float()
    dot    = (x * proj).sum(dim=-1)
    norm_x = x.norm(dim=-1)
    norm_y = proj.norm(dim=-1)
    return dot / (norm_x * norm_y + 1e-8)


# ── _cond_proj_cos_sim_kernel float correctness ───────────────────────────────

class TestKernelCorrectness:
    """Call _cond_proj_cos_sim_kernel directly and compare against the PyTorch reference.
    The kernel receives pre-projected y (tanh already applied), so we compute
    the projection with PyTorch first, then hand both x and y to the kernel."""

    @pytest.mark.parametrize("n_tokens,hidden_size", [
        (1,   64),
        (8,   64),
        (32,  512),
        (16,  100),    # not a power of 2 — tests masking
        (128, 4096),   # realistic LLM hidden size
        (64,  1536),   # Qwen hidden size
    ])
    def test_matches_pytorch_reference(self, n_tokens, hidden_size):
        torch.manual_seed(0)
        x   = torch.randn(n_tokens, hidden_size, device="cuda", dtype=torch.float32)
        cp  = torch.randn(hidden_size, hidden_size, device="cuda", dtype=torch.float32)
        out = torch.empty(n_tokens, device="cuda", dtype=torch.float32)

        proj = reference_project(x, cp)   # tanh(x @ cp.T)

        _cond_proj_cos_sim_kernel[(n_tokens,)](
            x, proj, out,
            hidden_size=hidden_size,
            BLOCK_H=next_power_of_2(hidden_size),
        )

        ref = reference_cos_sim(x, proj)
        torch.testing.assert_close(
            out, ref, atol=1e-5, rtol=1e-5,
            msg=f"Mismatch for n_tokens={n_tokens}, hidden_size={hidden_size}",
        )

    def test_output_range(self):
        """Cosine similarity must lie in [-1, 1]."""
        torch.manual_seed(1)
        hidden, n_tok = 128, 32
        x    = torch.randn(n_tok, hidden, device="cuda", dtype=torch.float32)
        cp   = torch.randn(hidden, hidden, device="cuda", dtype=torch.float32)
        proj = reference_project(x, cp)
        out  = torch.empty(n_tok, device="cuda", dtype=torch.float32)

        _cond_proj_cos_sim_kernel[(n_tok,)](
            x, proj, out,
            hidden_size=hidden,
            BLOCK_H=next_power_of_2(hidden),
        )
        assert out.min().item() >= -1.0 - 1e-5
        assert out.max().item() <=  1.0 + 1e-5

    def test_identical_x_and_proj_gives_one(self):
        """cos_sim(v, v) = 1 for any non-zero v."""
        hidden, n_tok = 64, 8
        x   = torch.randn(n_tok, hidden, device="cuda", dtype=torch.float32)
        out = torch.empty(n_tok, device="cuda", dtype=torch.float32)

        _cond_proj_cos_sim_kernel[(n_tok,)](
            x, x, out,
            hidden_size=hidden,
            BLOCK_H=next_power_of_2(hidden),
        )
        torch.testing.assert_close(out, torch.ones_like(out), atol=1e-5, rtol=0)


# ── threshold kernel correctness ──────────────────────────────────────────────

class TestThresholdKernel:
    """Verify forward() (→ _cond_proj_cos_sim_threshold_kernel) produces correct booleans."""

    @pytest.mark.parametrize("n_tokens,hidden_size,threshold", [
        (8,  64,    0.0),
        (8,  64,    0.5),
        (16, 512,  -0.3),
        (32, 4096,  0.9),
        (16, 100,   0.0),   # non-power-of-2 hidden
    ])
    def test_bool_matches_reference(self, n_tokens, hidden_size, threshold):
        torch.manual_seed(42)
        x  = torch.randn(n_tokens, hidden_size, device="cuda", dtype=torch.float32)
        cp = torch.randn(hidden_size, hidden_size, device="cuda", dtype=torch.float32)

        scorer = CondProjCosSim(condition_projector=torch.empty(hidden_size, hidden_size, device="cuda"), threshold=0.0, max_tokens=n_tokens)
        scorer.load_projector(cp)
        scorer.load_threshold(threshold)

        scorer.run(x)
        out_bool = scorer.out_bool[:n_tokens].clone()

        proj     = reference_project(x, cp)
        ref_bool = (reference_cos_sim(x, proj) > threshold).to(torch.int32)
        torch.testing.assert_close(out_bool, ref_bool,
            msg=f"Bool mismatch at threshold={threshold}")

    def test_threshold_above_all_gives_all_zeros(self):
        """threshold=2.0 → above max cos_sim of 1 → all False."""
        hidden, n_tok = 64, 8
        cp = torch.randn(hidden, hidden, device="cuda")
        scorer = CondProjCosSim(condition_projector=torch.empty(hidden, hidden, device="cuda"), threshold=0.0, max_tokens=n_tok)
        scorer.load_projector(cp)
        scorer.load_threshold(2.0)
        scorer.run(torch.randn(n_tok, hidden, device="cuda"))
        out = scorer.out_bool[:n_tok]
        assert out.sum().item() == 0

    def test_threshold_below_all_gives_all_ones(self):
        """threshold=-2.0 → below min cos_sim of -1 → all True."""
        hidden, n_tok = 64, 8
        cp = torch.randn(hidden, hidden, device="cuda")
        scorer = CondProjCosSim(condition_projector=torch.empty(hidden, hidden, device="cuda"), threshold=0.0, max_tokens=n_tok)
        scorer.load_projector(cp)
        scorer.load_threshold(-2.0)
        scorer.run(torch.randn(n_tok, hidden, device="cuda"))
        out = scorer.out_bool[:n_tok]
        assert out.sum().item() == n_tok

    def test_output_dtype_is_int32(self):
        hidden, n_tok = 64, 4
        scorer = CondProjCosSim(condition_projector=torch.empty(hidden, hidden, device="cuda"), threshold=0.0, max_tokens=n_tok)
        scorer.load_projector(torch.randn(hidden, hidden, device="cuda"))
        scorer.load_threshold(0.0)
        scorer.run(torch.randn(n_tok, hidden, device="cuda"))
        out = scorer.out_bool[:n_tok]
        assert out.dtype == torch.int32

    def test_output_shape(self):
        hidden, n_tok = 64, 10
        scorer = CondProjCosSim(condition_projector=torch.empty(hidden, hidden, device="cuda"), threshold=0.0, max_tokens=16)
        scorer.load_projector(torch.randn(hidden, hidden, device="cuda"))
        scorer.load_threshold(0.0)
        scorer.run(torch.randn(n_tok, hidden, device="cuda"))
        out = scorer.out_bool[:n_tok]
        assert out.shape == (n_tok,)


# ── set_output_buffers ────────────────────────────────────────────────────────

class TestSetOutputBuffers:

    def test_out_bool_redirected(self):
        """run() must write into the external buffer when set_output_buffers is called."""
        hidden, n_tok = 64, 8
        torch.manual_seed(7)
        x  = torch.randn(n_tok, hidden, device="cuda", dtype=torch.float32)
        cp = torch.randn(hidden, hidden, device="cuda", dtype=torch.float32)

        scorer = CondProjCosSim(condition_projector=torch.empty(hidden, hidden, device="cuda"), threshold=0.0, max_tokens=n_tok)
        scorer.load_projector(cp)
        scorer.load_threshold(0.0)

        scorer.run(x)
        expected = scorer.out_bool[:n_tok].clone()

        ext_buf = torch.full((n_tok,), -1, dtype=torch.int32, device="cuda")
        scorer.set_output_buffers(out_bool=[ext_buf])
        scorer.run(x)

        torch.testing.assert_close(ext_buf, expected,
            msg="External out_bool buffer does not match direct output")

    def test_own_out_bool_not_written_after_redirect(self):
        """After redirection, the class's own out_bool must not be written."""
        hidden, n_tok = 64, 8
        scorer = CondProjCosSim(condition_projector=torch.empty(hidden, hidden, device="cuda"), threshold=0.0, max_tokens=n_tok)
        scorer.load_projector(torch.randn(hidden, hidden, device="cuda"))
        scorer.load_threshold(0.0)

        ext_buf = torch.zeros(n_tok, dtype=torch.int32, device="cuda")
        scorer.set_output_buffers(out_bool=[ext_buf])

        scorer.out_bool.zero_()
        sentinel = scorer.out_bool.clone()

        scorer.run(torch.randn(n_tok, hidden, device="cuda"))

        torch.testing.assert_close(scorer.out_bool, sentinel,
            msg="Own out_bool was written despite external buffer being set")

    def test_external_buffer_address_is_stable(self):
        """set_output_buffers must not allocate — the external pointer must not change."""
        hidden, n_tok = 64, 8
        scorer  = CondProjCosSim(condition_projector=torch.empty(hidden, hidden, device="cuda"), threshold=0.0, max_tokens=n_tok)
        ext_buf = torch.zeros(n_tok, dtype=torch.int32, device="cuda")
        ptr_before = ext_buf.data_ptr()

        scorer.set_output_buffers(out_bool=[ext_buf])
        scorer.load_projector(torch.randn(hidden, hidden, device="cuda"))
        scorer.load_threshold(0.0)
        scorer.run(torch.randn(n_tok, hidden, device="cuda"))

        assert ext_buf.data_ptr() == ptr_before, \
            "External buffer address changed — unexpected allocation"


# ── fan-out to multiple sinks ────────────────────────────────────────────────

class TestFanOut:

    def test_out_bool_fanout_two_sinks(self):
        """forward() must populate every int32 sink with identical threshold flags."""
        hidden, n_tok = 64, 8
        torch.manual_seed(11)
        x  = torch.randn(n_tok, hidden, device="cuda", dtype=torch.float32)
        cp = torch.randn(hidden, hidden, device="cuda", dtype=torch.float32)

        scorer = CondProjCosSim(condition_projector=torch.empty(hidden, hidden, device="cuda"), threshold=0.0, max_tokens=n_tok)
        scorer.load_projector(cp)
        scorer.load_threshold(0.0)

        a = torch.full((n_tok,), -1, dtype=torch.int32, device="cuda")
        b = torch.full((n_tok,), -1, dtype=torch.int32, device="cuda")
        scorer.set_output_buffers(out_bool=[a, b])
        scorer.run(x)

        proj     = reference_project(x, cp)
        ref_bool = (reference_cos_sim(x, proj) > 0.0).to(torch.int32)
        torch.testing.assert_close(a, ref_bool,
            msg="First sink does not match reference threshold flags")
        torch.testing.assert_close(b, ref_bool,
            msg="Second sink does not match reference threshold flags")
        torch.testing.assert_close(a, b,
            msg="Sinks diverged — fan-out copy did not propagate")

    def test_out_fanout_two_sinks(self):
        """forward_backup() must populate every float32 sink with identical scores."""
        hidden, n_tok = 64, 8
        torch.manual_seed(13)
        x  = torch.randn(n_tok, hidden, device="cuda", dtype=torch.float32)
        cp = torch.randn(hidden, hidden, device="cuda", dtype=torch.float32)

        scorer = CondProjCosSim(condition_projector=torch.empty(hidden, hidden, device="cuda"), threshold=0.0, max_tokens=n_tok)
        scorer.load_projector(cp)
        scorer.load_threshold(0.0)

        a = torch.zeros(n_tok, dtype=torch.float32, device="cuda")
        b = torch.zeros(n_tok, dtype=torch.float32, device="cuda")
        scorer.set_output_buffers(out=[a, b])
        scorer.forward_backup(x)

        proj = reference_project(x, cp)
        ref  = reference_cos_sim(x, proj)
        torch.testing.assert_close(a, ref, atol=1e-5, rtol=1e-5,
            msg="First sink does not match reference cos_sim scores")
        torch.testing.assert_close(b, ref, atol=1e-5, rtol=1e-5,
            msg="Second sink does not match reference cos_sim scores")
        torch.testing.assert_close(a, b,
            msg="Sinks diverged — fan-out copy did not propagate")

    def test_set_output_buffers_validates_dtype(self):
        """set_output_buffers must reject sinks with the wrong dtype."""
        hidden, n_tok = 64, 8
        scorer = CondProjCosSim(condition_projector=torch.empty(hidden, hidden, device="cuda"), threshold=0.0, max_tokens=n_tok)

        wrong_dtype = torch.zeros(n_tok, dtype=torch.int32, device="cuda")
        with pytest.raises(AssertionError):
            scorer.set_output_buffers(out=[wrong_dtype])

        wrong_dtype = torch.zeros(n_tok, dtype=torch.float32, device="cuda")
        with pytest.raises(AssertionError):
            scorer.set_output_buffers(out_bool=[wrong_dtype])

    def test_set_output_buffers_validates_shape(self):
        """set_output_buffers must reject sinks with the wrong shape."""
        hidden, n_tok = 64, 8
        scorer = CondProjCosSim(condition_projector=torch.empty(hidden, hidden, device="cuda"), threshold=0.0, max_tokens=n_tok)

        wrong_shape = torch.zeros(n_tok + 1, dtype=torch.int32, device="cuda")
        with pytest.raises(AssertionError):
            scorer.set_output_buffers(out_bool=[wrong_shape])
