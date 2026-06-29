# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Tests for LinearProbe and MLPProbe in read_activations.py.

Run with:
    pytest vllm/activations_extractor/test_read_activations.py -v
"""

import pytest
import torch

from vllm.activations_extractor.read_activations import LinearProbe, MLPProbe


# ── LinearProbe helpers ───────────────────────────────────────────────────────

def make_linear_probe(
    hidden_size: int = 64,
    n_classes:   int = 1,
    max_tokens:  int = 32,
    dtype: torch.dtype = torch.float32,
) -> LinearProbe:
    weights = torch.randn(n_classes, hidden_size, device="cuda", dtype=dtype)
    bias    = torch.randn(n_classes,              device="cuda", dtype=dtype)
    return LinearProbe(weights, bias, max_tokens)


def reference_probe(
    x:       torch.Tensor,
    weights: torch.Tensor,
    bias:    torch.Tensor,
) -> torch.Tensor:
    logits = x.float() @ weights.float().T + bias.float()
    return torch.sigmoid(logits)


# ── LinearProbe correctness ───────────────────────────────────────────────────

class TestLinearProbeCorrectness:

    @pytest.mark.parametrize("n_tokens,hidden_size,n_classes", [
        (1,   64,   1),
        (8,   64,   1),
        (8,   64,   4),
        (32,  512,  1),
        (16,  512,  8),
        (128, 4096, 1),
        (64,  4096, 4),
    ])
    def test_matches_pytorch_reference(self, n_tokens, hidden_size, n_classes):
        torch.manual_seed(0)
        weights = torch.randn(n_classes, hidden_size, device="cuda", dtype=torch.float32)
        bias    = torch.randn(n_classes,              device="cuda", dtype=torch.float32)
        x       = torch.randn(n_tokens,  hidden_size, device="cuda", dtype=torch.float32)

        probe = LinearProbe(weights, bias, max_tokens=n_tokens)
        out   = probe.run(x).clone()
        ref   = reference_probe(x, weights, bias)

        torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5,
            msg=f"Mismatch for n_tokens={n_tokens}, hidden={hidden_size}, classes={n_classes}")

    def test_output_range_is_probability(self):
        probe = make_linear_probe(hidden_size=128, n_classes=3, max_tokens=16)
        out   = probe.run(torch.randn(16, 128, device="cuda"))
        assert out.min().item() >= 0.0
        assert out.max().item() <= 1.0

    def test_zero_weights_gives_half(self):
        weights = torch.zeros(2, 64, device="cuda")
        bias    = torch.zeros(2,     device="cuda")
        probe   = LinearProbe(weights, bias, max_tokens=8)
        out     = probe.run(torch.randn(8, 64, device="cuda"))
        torch.testing.assert_close(out, torch.full_like(out, 0.5), atol=1e-6, rtol=0)

    def test_large_positive_bias_gives_one(self):
        weights = torch.zeros(1, 64, device="cuda")
        bias    = torch.tensor([100.0], device="cuda")
        probe   = LinearProbe(weights, bias, max_tokens=8)
        out     = probe.run(torch.randn(8, 64, device="cuda"))
        torch.testing.assert_close(out, torch.ones_like(out), atol=1e-4, rtol=0)

    def test_large_negative_bias_gives_zero(self):
        weights = torch.zeros(1, 64, device="cuda")
        bias    = torch.tensor([-100.0], device="cuda")
        probe   = LinearProbe(weights, bias, max_tokens=8)
        out     = probe.run(torch.randn(8, 64, device="cuda"))
        torch.testing.assert_close(out, torch.zeros_like(out), atol=1e-4, rtol=0)

    def test_hidden_size_not_multiple_of_block(self):
        hidden, n_cls, n_tok = 100, 2, 5
        weights = torch.randn(n_cls, hidden, device="cuda")
        bias    = torch.randn(n_cls,         device="cuda")
        x       = torch.randn(n_tok, hidden, device="cuda")
        probe   = LinearProbe(weights, bias, max_tokens=n_tok)
        torch.testing.assert_close(probe.run(x), reference_probe(x, weights, bias),
                                   atol=1e-5, rtol=1e-5)

    def test_1d_weight_accepted_for_binary(self):
        """(hidden_size,) weight and scalar bias are accepted for n_classes=1."""
        probe = LinearProbe(
            torch.randn(64, device="cuda"),
            torch.tensor(0.0, device="cuda"),
            max_tokens=4,
        )
        assert probe.weights.shape == (1, 64)
        assert probe.bias.shape    == (1,)


# ── LinearProbe output shape ──────────────────────────────────────────────────

class TestLinearProbeOutputShape:

    def test_output_shape_binary(self):
        probe = make_linear_probe(hidden_size=64, n_classes=1, max_tokens=32)
        out   = probe.run(torch.randn(10, 64, device="cuda"))
        assert out.shape == (10, 1)

    def test_output_shape_multiclass(self):
        probe = make_linear_probe(hidden_size=64, n_classes=5, max_tokens=32)
        out   = probe.run(torch.randn(7, 64, device="cuda"))
        assert out.shape == (7, 5)

    def test_leading_dims_flattened(self):
        probe = make_linear_probe(hidden_size=64, n_classes=1, max_tokens=32)
        out   = probe.run(torch.randn(4, 8, 64, device="cuda"))
        assert out.shape == (32, 1)

    def test_output_backed_by_preallocated_buffer(self):
        probe = make_linear_probe(hidden_size=64, n_classes=1, max_tokens=32)
        addr  = probe.output.data_ptr()
        probe.run(torch.randn(8, 64, device="cuda"))
        assert probe.output.data_ptr() == addr


# ── LinearProbe buffer stability (CUDA graph safety) ─────────────────────────

class TestLinearProbeBufferStability:

    def test_weight_addresses_stable_after_load_weights(self):
        probe    = make_linear_probe()
        addr_w   = probe.weights.data_ptr()
        addr_b   = probe.bias.data_ptr()
        addr_out = probe.output.data_ptr()

        probe.load_weights(
            torch.randn(1, 64, device="cuda"),
            torch.randn(1,     device="cuda"),
        )

        assert probe.weights.data_ptr() == addr_w,   "weights address changed"
        assert probe.bias.data_ptr()    == addr_b,   "bias address changed"
        assert probe.output.data_ptr()  == addr_out, "output buffer address changed"

    def test_hot_swap_changes_output(self):
        hidden, n_tok = 64, 4
        x = torch.randn(n_tok, hidden, device="cuda")

        w1 = torch.ones(1, hidden, device="cuda")
        b1 = torch.zeros(1, device="cuda")
        probe = LinearProbe(w1, b1, max_tokens=n_tok)
        out1  = probe.run(x).clone()

        probe.load_weights(torch.ones(1, hidden, device="cuda") * (-1),
                           torch.zeros(1, device="cuda"))
        out2  = probe.run(x).clone()

        torch.testing.assert_close(out1 + out2, torch.ones_like(out1), atol=1e-5, rtol=0)


# ── LinearProbe multiple instances ───────────────────────────────────────────

class TestLinearProbeMultipleInstances:

    def test_independent_weights(self):
        hidden, n_tok = 64, 8
        x = torch.randn(n_tok, hidden, device="cuda")

        probe_a = LinearProbe(torch.ones(1, hidden, device="cuda"),
                              torch.zeros(1, device="cuda"), max_tokens=n_tok)
        probe_b = LinearProbe(torch.ones(1, hidden, device="cuda") * (-1),
                              torch.zeros(1, device="cuda"), max_tokens=n_tok)

        assert not torch.allclose(probe_a.run(x), probe_b.run(x))

    def test_probes_do_not_share_buffers(self):
        hidden, n_tok = 64, 4
        x = torch.randn(n_tok, hidden, device="cuda")

        w = torch.zeros(1, hidden, device="cuda")
        b = torch.zeros(1, device="cuda")
        probe_a = LinearProbe(w.clone(), b.clone(), max_tokens=n_tok)
        probe_b = LinearProbe(w.clone(), b.clone(), max_tokens=n_tok)

        out_b_before = probe_b.run(x).clone()
        probe_a.load_weights(torch.ones(1, hidden, device="cuda") * 999,
                             torch.ones(1, device="cuda") * 999)
        out_b_after  = probe_b.run(x).clone()
        torch.testing.assert_close(out_b_before, out_b_after)

    def test_different_hidden_sizes(self):
        probe_s = LinearProbe(torch.randn(1, 64,   device="cuda"),
                              torch.zeros(1,        device="cuda"), max_tokens=8)
        probe_l = LinearProbe(torch.randn(1, 4096,  device="cuda"),
                              torch.zeros(1,        device="cuda"), max_tokens=8)
        assert probe_s.run(torch.randn(4, 64,   device="cuda")).shape == (4, 1)
        assert probe_l.run(torch.randn(4, 4096, device="cuda")).shape == (4, 1)


# ── MLPProbe helpers ──────────────────────────────────────────────────────────

def make_mlp(
    input_dim:  int = 64,
    hidden_dim: int = 32,
    output_dim: int = 8,
    max_tokens: int = 16,
    dtype: torch.dtype = torch.float32,
) -> MLPProbe:
    w1 = torch.randn(hidden_dim, input_dim,  device="cuda", dtype=dtype)
    b1 = torch.randn(hidden_dim,              device="cuda", dtype=dtype)
    w2 = torch.randn(output_dim, hidden_dim, device="cuda", dtype=dtype)
    b2 = torch.randn(output_dim,              device="cuda", dtype=dtype)
    return MLPProbe(w1, b1, w2, b2, max_tokens)


def reference_mlp(x, w1, b1, w2, b2):
    h   = torch.relu(x.float() @ w1.float().T + b1.float())
    out = torch.sigmoid(h @ w2.float().T + b2.float())
    return out


# ── MLPProbe correctness ──────────────────────────────────────────────────────

class TestMLPProbeCorrectness:

    @pytest.mark.parametrize("n_tokens,input_dim,hidden_dim,output_dim", [
        (1,   64,   32,  8),
        (4,   64,   32,  8),
        (8,   128,  64,  16),
        (8,   100,  32,  8),
        (16,  1536, 256, 16),
    ])
    def test_matches_pytorch_reference(self, n_tokens, input_dim, hidden_dim, output_dim):
        torch.manual_seed(0)
        w1 = torch.randn(hidden_dim, input_dim,  device="cuda", dtype=torch.float32)
        b1 = torch.randn(hidden_dim,              device="cuda", dtype=torch.float32)
        w2 = torch.randn(output_dim, hidden_dim, device="cuda", dtype=torch.float32)
        b2 = torch.randn(output_dim,              device="cuda", dtype=torch.float32)
        x  = torch.randn(n_tokens, input_dim,    device="cuda", dtype=torch.float32)

        mlp = MLPProbe(w1, b1, w2, b2, max_tokens=n_tokens)
        out = mlp.run(x).clone()
        ref = reference_mlp(x, w1, b1, w2, b2)

        torch.testing.assert_close(out.float(), ref.float(), atol=1e-4, rtol=1e-4,
            msg=f"Mismatch for n_tokens={n_tokens}, input={input_dim}, "
                f"hidden={hidden_dim}, output={output_dim}")

    def test_zero_weights_gives_half(self):
        w1 = torch.zeros(32, 64, device="cuda")
        b1 = torch.zeros(32,     device="cuda")
        w2 = torch.zeros(8,  32, device="cuda")
        b2 = torch.zeros(8,      device="cuda")
        mlp = MLPProbe(w1, b1, w2, b2, max_tokens=4)
        out = mlp.run(torch.randn(4, 64, device="cuda"))
        torch.testing.assert_close(out, torch.full_like(out, 0.5), atol=1e-5, rtol=0)

    def test_output_shape(self):
        mlp = make_mlp(input_dim=64, hidden_dim=32, output_dim=8, max_tokens=16)
        assert mlp.run(torch.randn(5, 64, device="cuda")).shape == (5, 8)

    def test_output_range_is_probability(self):
        mlp = make_mlp()
        out = mlp.run(torch.randn(8, 64, device="cuda"))
        assert out.min().item() >= 0.0
        assert out.max().item() <= 1.0

    def test_leading_dims_flattened(self):
        w1 = torch.zeros(32, 64, device="cuda")
        b1 = torch.zeros(32,     device="cuda")
        w2 = torch.zeros(8,  32, device="cuda")
        b2 = torch.zeros(8,      device="cuda")
        mlp = MLPProbe(w1, b1, w2, b2, max_tokens=32)
        assert mlp.run(torch.randn(4, 8, 64, device="cuda")).shape == (32, 8)


# ── MLPProbe buffer stability (CUDA graph safety) ────────────────────────────

class TestMLPProbeBufferStability:

    def test_weight_addresses_stable_after_load_weights(self):
        mlp   = make_mlp()
        addrs = {n: p.data_ptr() for n, p in mlp.named_parameters()}

        mlp.load_weights(
            torch.randn(32, 64, device="cuda"), torch.randn(32, device="cuda"),
            torch.randn(8,  32, device="cuda"), torch.randn(8,  device="cuda"),
        )

        for name, addr in addrs.items():
            assert dict(mlp.named_parameters())[name].data_ptr() == addr, \
                f"{name} address changed after load_weights()"

    def test_hidden_buf_address_stable(self):
        mlp  = make_mlp()
        addr = mlp.hidden_buf.data_ptr()
        mlp.run(torch.randn(4, 64, device="cuda"))
        mlp.run(torch.randn(4, 64, device="cuda"))
        assert mlp.hidden_buf.data_ptr() == addr

    def test_output_buf_address_stable(self):
        mlp  = make_mlp()
        addr = mlp.output_buf.data_ptr()
        mlp.run(torch.randn(4, 64, device="cuda"))
        assert mlp.output_buf.data_ptr() == addr


# ── MLPProbe n_tokens slicing ─────────────────────────────────────────────────

class TestMLPProbeNTokens:

    def test_output_slice_length_equals_n_tokens(self):
        mlp = make_mlp(input_dim=64, hidden_dim=32, output_dim=8, max_tokens=16)
        for n in [1, 4, 7, 16]:
            out = mlp.run(torch.randn(n, 64, device="cuda"))
            assert out.shape[0] == n, f"Expected {n} rows, got {out.shape[0]}"
