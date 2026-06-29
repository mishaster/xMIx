# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from write_activations import subtract_refusal_projection_multi_vec_per_token

HIDDEN_SIZE = 4096
MAX_VECS    = 4      # keep small for fast tests
TOLERANCE   = 1e-2   # float16 + atomic float32 accumulation


# ---------------------------------------------------------------------------
# PyTorch reference — pure Python, no Triton
# ---------------------------------------------------------------------------

def pytorch_reference(x, r, token_indices, vec_indices, n_valid, n_vecs_per_token):
    r_norm = r / r.norm(dim=1, keepdim=True)   # (max_vecs, hidden_size)
    x = x.clone()
    for i in range(n_valid):
        row    = token_indices[i].item()
        n_vecs = n_vecs_per_token[i].item()
        for v in range(n_vecs):
            vec_idx = vec_indices[i, v].item()
            r_v = r_norm[vec_idx]
            dot = x[row] @ r_v
            x[row] = x[row] - dot * r_v
    return x


# ---------------------------------------------------------------------------
# Buffer helpers
# ---------------------------------------------------------------------------

def make_buffers(max_tokens, max_vecs):
    token_indices    = torch.zeros(max_tokens,           dtype=torch.int32, device='cuda')
    vec_indices      = torch.zeros(max_tokens, max_vecs, dtype=torch.int32, device='cuda')
    n_valid_buf      = torch.zeros(1,                    dtype=torch.int32, device='cuda')
    n_vecs_per_token = torch.zeros(max_tokens,           dtype=torch.int32, device='cuda')
    return token_indices, vec_indices, n_valid_buf, n_vecs_per_token


def set_buffers(token_indices, vec_indices, n_valid_buf, n_vecs_per_token,
                rows, vecs_per_row):
    """
    rows:         list of row indices into x, one per selected token
    vecs_per_row: list of lists — vecs_per_row[i] = vector indices for token i
    """
    n = len(rows)
    token_indices[:n] = torch.tensor(rows, dtype=torch.int32, device='cuda')
    n_valid_buf.fill_(n)
    for i, vecs in enumerate(vecs_per_row):
        nv = len(vecs)
        vec_indices[i, :nv] = torch.tensor(vecs, dtype=torch.int32, device='cuda')
        n_vecs_per_token[i] = nv


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_correctness_single_vec_per_token(n_tokens, n_selected):
    """Each selected token uses exactly one (different) vector — matches reference."""
    x = torch.randn(n_tokens, HIDDEN_SIZE, device='cuda', dtype=torch.float16)
    r = torch.randn(MAX_VECS, HIDDEN_SIZE, device='cuda', dtype=torch.float16)

    rows         = list(range(n_selected))
    vecs_per_row = [[i % MAX_VECS] for i in range(n_selected)]   # one vec each

    ref = pytorch_reference(x, r,
                            torch.tensor(rows, dtype=torch.int32, device='cuda'),
                            torch.tensor(vecs_per_row, dtype=torch.int32, device='cuda'),
                            n_selected,
                            torch.tensor([1] * n_selected, dtype=torch.int32, device='cuda'))

    token_indices, vec_indices, n_valid_buf, n_vecs_per_token = make_buffers(n_tokens, MAX_VECS)
    set_buffers(token_indices, vec_indices, n_valid_buf, n_vecs_per_token, rows, vecs_per_row)
    out = subtract_refusal_projection_multi_vec_per_token(
        x.clone(), r, token_indices, vec_indices, n_valid_buf, n_vecs_per_token)

    assert torch.allclose(out, ref, atol=TOLERANCE), \
        f"Correctness fail (single vec): max diff = {(out - ref).abs().max()}"
    print(f"PASS correctness single vec ({n_tokens} tokens, {n_selected} selected)")


def test_correctness_multi_vec_per_token(n_tokens, n_selected, n_vecs):
    """Each selected token uses n_vecs vectors — all projections removed."""
    assert n_vecs <= MAX_VECS
    x = torch.randn(n_tokens, HIDDEN_SIZE, device='cuda', dtype=torch.float16)
    r = torch.randn(MAX_VECS, HIDDEN_SIZE, device='cuda', dtype=torch.float16)

    rows         = list(range(n_selected))
    vecs_per_row = [list(range(n_vecs)) for _ in range(n_selected)]

    ref = pytorch_reference(x, r,
                            torch.tensor(rows, dtype=torch.int32, device='cuda'),
                            torch.tensor(vecs_per_row, dtype=torch.int32, device='cuda'),
                            n_selected,
                            torch.tensor([n_vecs] * n_selected, dtype=torch.int32, device='cuda'))

    token_indices, vec_indices, n_valid_buf, n_vecs_per_token = make_buffers(n_tokens, MAX_VECS)
    set_buffers(token_indices, vec_indices, n_valid_buf, n_vecs_per_token, rows, vecs_per_row)
    out = subtract_refusal_projection_multi_vec_per_token(
        x.clone(), r, token_indices, vec_indices, n_valid_buf, n_vecs_per_token)

    assert torch.allclose(out, ref, atol=TOLERANCE), \
        f"Correctness fail (multi vec): max diff = {(out - ref).abs().max()}"
    print(f"PASS correctness multi vec ({n_tokens} tokens, {n_selected} selected, {n_vecs} vecs each)")


def test_different_vectors_per_token():
    """Each token uses a different number of vectors — per-token n_vecs respected."""
    n_tokens = 32
    x = torch.randn(n_tokens, HIDDEN_SIZE, device='cuda', dtype=torch.float16)
    r = torch.randn(MAX_VECS, HIDDEN_SIZE, device='cuda', dtype=torch.float16)

    rows         = [0, 1, 2, 3]
    vecs_per_row = [[0], [0, 1], [0, 1, 2], [0, 1, 2, 3]]   # 1, 2, 3, 4 vectors

    ref = pytorch_reference(
        x, r,
        torch.tensor(rows, dtype=torch.int32, device='cuda'),
        torch.tensor([v + [0] * (MAX_VECS - len(v)) for v in vecs_per_row],
                     dtype=torch.int32, device='cuda'),
        len(rows),
        torch.tensor([len(v) for v in vecs_per_row], dtype=torch.int32, device='cuda'))

    token_indices, vec_indices, n_valid_buf, n_vecs_per_token = make_buffers(n_tokens, MAX_VECS)
    set_buffers(token_indices, vec_indices, n_valid_buf, n_vecs_per_token, rows, vecs_per_row)
    out = subtract_refusal_projection_multi_vec_per_token(
        x.clone(), r, token_indices, vec_indices, n_valid_buf, n_vecs_per_token)

    assert torch.allclose(out, ref, atol=TOLERANCE), \
        f"Correctness fail (different vecs per token): max diff = {(out - ref).abs().max()}"
    print("PASS different vectors per token")


def test_unselected_rows_unchanged():
    """Rows not in token_indices must be completely untouched."""
    n_tokens, n_selected = 32, 4
    x = torch.randn(n_tokens, HIDDEN_SIZE, device='cuda', dtype=torch.float16)
    r = torch.randn(MAX_VECS, HIDDEN_SIZE, device='cuda', dtype=torch.float16)

    rows         = list(range(n_selected))
    vecs_per_row = [[0, 1]] * n_selected

    x_orig = x.clone()
    token_indices, vec_indices, n_valid_buf, n_vecs_per_token = make_buffers(n_tokens, MAX_VECS)
    set_buffers(token_indices, vec_indices, n_valid_buf, n_vecs_per_token, rows, vecs_per_row)
    subtract_refusal_projection_multi_vec_per_token(
        x, r, token_indices, vec_indices, n_valid_buf, n_vecs_per_token)

    unselected = list(range(n_selected, n_tokens))
    assert torch.equal(x[unselected], x_orig[unselected]), \
        "Unselected rows were modified"
    print("PASS unselected rows unchanged")


def test_padding_tokens_ignored():
    """Padding slots in token_indices (beyond n_valid) must not affect output."""
    n_tokens, n_selected = 32, 4
    x = torch.randn(n_tokens, HIDDEN_SIZE, device='cuda', dtype=torch.float16)
    r = torch.randn(MAX_VECS, HIDDEN_SIZE, device='cuda', dtype=torch.float16)

    rows         = [0, 1, 2, 3]
    vecs_per_row = [[0]] * 4

    token_indices, vec_indices, n_valid_buf, n_vecs_per_token = make_buffers(n_tokens, MAX_VECS)
    set_buffers(token_indices, vec_indices, n_valid_buf, n_vecs_per_token, rows, vecs_per_row)
    # Poison padding: point unused slots at row 10 — if processed, row 10 would change
    token_indices[n_selected:] = 10

    x_orig = x.clone()
    subtract_refusal_projection_multi_vec_per_token(
        x, r, token_indices, vec_indices, n_valid_buf, n_vecs_per_token)

    assert torch.equal(x[10], x_orig[10]), \
        "Padding token slot was processed — row 10 incorrectly modified"
    print("PASS padding tokens ignored")


def test_padding_vectors_ignored():
    """Extra vector slots beyond n_vecs_per_token must not affect output."""
    n_tokens = 32
    x = torch.randn(n_tokens, HIDDEN_SIZE, device='cuda', dtype=torch.float16)
    r = torch.randn(MAX_VECS, HIDDEN_SIZE, device='cuda', dtype=torch.float16)

    # Token 0 uses only 1 vector, but we poison the remaining slots with vector index 2
    rows         = [0]
    vecs_per_row = [[0]]   # only vec 0 is valid

    token_indices, vec_indices, n_valid_buf, n_vecs_per_token = make_buffers(n_tokens, MAX_VECS)
    set_buffers(token_indices, vec_indices, n_valid_buf, n_vecs_per_token, rows, vecs_per_row)
    vec_indices[0, 1:] = 2   # poison the unused vector slots with vec 2

    # Reference: only vec 0 applied
    ref = pytorch_reference(
        x, r,
        torch.tensor([0], dtype=torch.int32, device='cuda'),
        torch.tensor([[0, 0, 0, 0]], dtype=torch.int32, device='cuda'),
        1,
        torch.tensor([1], dtype=torch.int32, device='cuda'))

    out = subtract_refusal_projection_multi_vec_per_token(
        x.clone(), r, token_indices, vec_indices, n_valid_buf, n_vecs_per_token)

    assert torch.allclose(out, ref, atol=TOLERANCE), \
        f"Padding vector slot was processed: max diff = {(out - ref).abs().max()}"
    print("PASS padding vectors ignored")


def test_projection_removed_multi_vec():
    """After kernel, dot product of each selected row with each of its vectors should be ~0."""
    n_tokens = 32
    x = torch.randn(n_tokens, HIDDEN_SIZE, device='cuda', dtype=torch.float16)
    r = torch.randn(MAX_VECS, HIDDEN_SIZE, device='cuda', dtype=torch.float16)

    rows         = list(range(n_tokens))
    vecs_per_row = [list(range(MAX_VECS))] * n_tokens   # all vectors for every token

    token_indices, vec_indices, n_valid_buf, n_vecs_per_token = make_buffers(n_tokens, MAX_VECS)
    set_buffers(token_indices, vec_indices, n_valid_buf, n_vecs_per_token, rows, vecs_per_row)
    subtract_refusal_projection_multi_vec_per_token(
        x, r, token_indices, vec_indices, n_valid_buf, n_vecs_per_token)

    r_norm = r / r.norm(dim=1, keepdim=True)
    for vec_idx in range(MAX_VECS):
        dots = (x.float() @ r_norm[vec_idx].float())   # (n_tokens,)
        assert dots.abs().max() < 0.1, \
            f"Projection not removed for vec {vec_idx}: max residual = {dots.abs().max()}"
    print(f"PASS projection removed for all {MAX_VECS} vectors ({n_tokens} tokens)")


if __name__ == "__main__":
    test_correctness_single_vec_per_token(n_tokens=32,  n_selected=8)
    test_correctness_single_vec_per_token(n_tokens=128, n_selected=64)
    test_correctness_multi_vec_per_token(n_tokens=32,  n_selected=8,  n_vecs=2)
    test_correctness_multi_vec_per_token(n_tokens=64,  n_selected=16, n_vecs=MAX_VECS)
    test_different_vectors_per_token()
    test_unselected_rows_unchanged()
    test_padding_tokens_ignored()
    test_padding_vectors_ignored()
    test_projection_removed_multi_vec()
    print("\nAll tests passed.")
