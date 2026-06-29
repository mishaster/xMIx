# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from write_activations import subtract_refusal_projection_indexed

HIDDEN_SIZE = 4096
TOLERANCE = 1e-2  # float16 + atomic float32 accumulation


def pytorch_reference(x, r, indices):
    """Pure PyTorch equivalent for correctness comparison."""
    r_norm = r / r.norm()
    projection = x[indices] @ r_norm          # (n_selected,)
    x = x.clone()
    x[indices] -= torch.outer(projection, r_norm)
    return x


def make_buffers(max_tokens):
    indices_buf = torch.zeros(max_tokens, dtype=torch.int32, device='cuda')
    n_valid_buf  = torch.zeros(1,          dtype=torch.int32, device='cuda')
    return indices_buf, n_valid_buf


def set_buffers(indices_buf, n_valid_buf, indices):
    n = len(indices)
    indices_buf[:n] = torch.tensor(indices, dtype=torch.int32, device='cuda')
    n_valid_buf.fill_(n)


def test_correctness(n_tokens: int, n_selected: int):
    """Kernel output matches PyTorch reference for selected rows."""
    x = torch.randn(n_tokens, HIDDEN_SIZE, device='cuda', dtype=torch.float16)
    r = torch.randn(HIDDEN_SIZE, device='cuda', dtype=torch.float16)
    indices = list(range(n_selected))  # first n_selected rows

    ref = pytorch_reference(x, r, indices)

    indices_buf, n_valid_buf = make_buffers(max_tokens=n_tokens)
    set_buffers(indices_buf, n_valid_buf, indices)
    out = subtract_refusal_projection_indexed(x.clone(), r, indices_buf, n_valid_buf)

    assert torch.allclose(out, ref, atol=TOLERANCE), \
        f"Correctness fail: max diff = {(out - ref).abs().max()}"
    print(f"PASS correctness ({n_tokens} tokens, {n_selected} selected)")


def test_unselected_rows_unchanged(n_tokens: int, n_selected: int):
    """Rows NOT in the index list must be completely untouched."""
    x = torch.randn(n_tokens, HIDDEN_SIZE, device='cuda', dtype=torch.float16)
    r = torch.randn(HIDDEN_SIZE, device='cuda', dtype=torch.float16)
    indices = list(range(n_selected))

    x_orig = x.clone()
    indices_buf, n_valid_buf = make_buffers(max_tokens=n_tokens)
    set_buffers(indices_buf, n_valid_buf, indices)
    subtract_refusal_projection_indexed(x, r, indices_buf, n_valid_buf)

    unselected = list(range(n_selected, n_tokens))
    assert torch.equal(x[unselected], x_orig[unselected]), \
        "Unselected rows were modified"
    print(f"PASS unselected rows unchanged ({n_tokens} tokens, {n_selected} selected)")


def test_padding_slots_ignored():
    """Padding slots in indices_buf (beyond n_valid) must not affect output."""
    n_tokens, n_selected, max_tokens = 32, 4, 32
    x = torch.randn(n_tokens, HIDDEN_SIZE, device='cuda', dtype=torch.float16)
    r = torch.randn(HIDDEN_SIZE, device='cuda', dtype=torch.float16)
    indices = [0, 1, 2, 3]

    # Fill padding slots with row 10 — if padding were processed, row 10 would change
    indices_buf, n_valid_buf = make_buffers(max_tokens)
    set_buffers(indices_buf, n_valid_buf, indices)
    indices_buf[n_selected:] = 10  # poisoned padding

    x_orig = x.clone()
    subtract_refusal_projection_indexed(x, r, indices_buf, n_valid_buf)

    assert torch.equal(x[10], x_orig[10]), \
        "Padding slot was processed — row 10 was incorrectly modified"
    print("PASS padding slots ignored")


def test_n_valid_controls_selection():
    """Changing n_valid_buf selects different numbers of tokens without reallocating."""
    n_tokens, max_tokens = 32, 32
    r = torch.randn(HIDDEN_SIZE, device='cuda', dtype=torch.float16)
    indices_buf, n_valid_buf = make_buffers(max_tokens)
    indices_buf[:max_tokens] = torch.arange(max_tokens, dtype=torch.int32)

    for n in [1, 8, 16, 32]:
        x = torch.randn(n_tokens, HIDDEN_SIZE, device='cuda', dtype=torch.float16)
        x_orig = x.clone()
        n_valid_buf.fill_(n)

        subtract_refusal_projection_indexed(x, r, indices_buf, n_valid_buf)

        # rows 0..n-1 should change, rows n..31 should not
        assert not torch.equal(x[:n], x_orig[:n]), \
            f"Selected rows unchanged for n_valid={n}"
        if n < n_tokens:
            assert torch.equal(x[n:], x_orig[n:]), \
                f"Unselected rows changed for n_valid={n}"
        print(f"PASS n_valid={n} controls selection correctly")


def test_projection_removed(n_tokens: int):
    """After kernel, dot product of each selected row with r should be ~0."""
    x = torch.randn(n_tokens, HIDDEN_SIZE, device='cuda', dtype=torch.float16)
    r = torch.randn(HIDDEN_SIZE, device='cuda', dtype=torch.float16)
    indices = list(range(n_tokens))

    indices_buf, n_valid_buf = make_buffers(max_tokens=n_tokens)
    set_buffers(indices_buf, n_valid_buf, indices)
    subtract_refusal_projection_indexed(x, r, indices_buf, n_valid_buf)

    r_norm = r / r.norm()
    dots = (x.float() @ r_norm.float())  # (n_tokens,)
    assert dots.abs().max() < 0.1, \
        f"Projection not removed: max residual dot = {dots.abs().max()}"
    print(f"PASS projection removed ({n_tokens} tokens)")


if __name__ == "__main__":
    test_correctness(n_tokens=32,  n_selected=8)
    test_correctness(n_tokens=128, n_selected=64)
    test_correctness(n_tokens=512, n_selected=1)    # single token
    test_unselected_rows_unchanged(n_tokens=32,  n_selected=8)
    test_unselected_rows_unchanged(n_tokens=128, n_selected=1)
    test_padding_slots_ignored()
    test_n_valid_controls_selection()
    test_projection_removed(n_tokens=64)
    print("\nAll tests passed.")
