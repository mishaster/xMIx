# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from write_activations_multi_vector import subtract_summed_vectors_per_token

HIDDEN_SIZE = 4096
MAX_VECS    = 4
TOLERANCE   = 1e-2   # float16


# ---------------------------------------------------------------------------
# PyTorch reference
# ---------------------------------------------------------------------------

def pytorch_reference(x, r, token_indices, vec_indices, n_valid, n_vecs_per_token):
    """For each selected token, sum its assigned vectors and subtract from x."""
    x = x.clone()
    for i in range(n_valid):
        row    = token_indices[i].item()
        n_vecs = n_vecs_per_token[i].item()
        combined = torch.zeros(r.shape[1], device=r.device, dtype=torch.float32)
        for v in range(n_vecs):
            vec_idx = vec_indices[i, v].item()
            combined += r[vec_idx].to(torch.float32)
        x[row] = (x[row].to(torch.float32) - combined).to(x.dtype)
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

def test_correctness_single_vec(n_tokens, n_selected):
    """Each token uses one vector — result equals x minus that vector."""
    x = torch.randn(n_tokens, HIDDEN_SIZE, device='cuda', dtype=torch.float16)
    r = torch.randn(MAX_VECS, HIDDEN_SIZE, device='cuda', dtype=torch.float16)

    rows         = list(range(n_selected))
    vecs_per_row = [[i % MAX_VECS] for i in range(n_selected)]

    ref = pytorch_reference(
        x, r,
        torch.tensor(rows, dtype=torch.int32, device='cuda'),
        torch.tensor(vecs_per_row, dtype=torch.int32, device='cuda'),
        n_selected,
        torch.tensor([1] * n_selected, dtype=torch.int32, device='cuda'),
    )

    token_indices, vec_indices, n_valid_buf, n_vecs_per_token = make_buffers(n_tokens, MAX_VECS)
    set_buffers(token_indices, vec_indices, n_valid_buf, n_vecs_per_token, rows, vecs_per_row)
    out = subtract_summed_vectors_per_token(
        x.clone(), r, token_indices, vec_indices, n_valid_buf, n_vecs_per_token)

    assert torch.allclose(out, ref, atol=TOLERANCE), \
        f"Correctness fail (single vec): max diff = {(out - ref).abs().max()}"
    print(f"PASS correctness single vec ({n_tokens} tokens, {n_selected} selected)")


def test_correctness_multi_vec(n_tokens, n_selected, n_vecs):
    """Each token uses n_vecs vectors — result equals x minus their sum."""
    assert n_vecs <= MAX_VECS
    x = torch.randn(n_tokens, HIDDEN_SIZE, device='cuda', dtype=torch.float16)
    r = torch.randn(MAX_VECS, HIDDEN_SIZE, device='cuda', dtype=torch.float16)

    rows         = list(range(n_selected))
    vecs_per_row = [list(range(n_vecs)) for _ in range(n_selected)]

    ref = pytorch_reference(
        x, r,
        torch.tensor(rows, dtype=torch.int32, device='cuda'),
        torch.tensor(vecs_per_row, dtype=torch.int32, device='cuda'),
        n_selected,
        torch.tensor([n_vecs] * n_selected, dtype=torch.int32, device='cuda'),
    )

    token_indices, vec_indices, n_valid_buf, n_vecs_per_token = make_buffers(n_tokens, MAX_VECS)
    set_buffers(token_indices, vec_indices, n_valid_buf, n_vecs_per_token, rows, vecs_per_row)
    out = subtract_summed_vectors_per_token(
        x.clone(), r, token_indices, vec_indices, n_valid_buf, n_vecs_per_token)

    assert torch.allclose(out, ref, atol=TOLERANCE), \
        f"Correctness fail (multi vec): max diff = {(out - ref).abs().max()}"
    print(f"PASS correctness multi vec ({n_tokens} tokens, {n_selected} selected, {n_vecs} vecs each)")


def test_different_vecs_per_token():
    """Each token uses a different number of vectors."""
    n_tokens = 32
    x = torch.randn(n_tokens, HIDDEN_SIZE, device='cuda', dtype=torch.float16)
    r = torch.randn(MAX_VECS, HIDDEN_SIZE, device='cuda', dtype=torch.float16)

    rows         = [0, 1, 2, 3]
    vecs_per_row = [[0], [0, 1], [0, 1, 2], [0, 1, 2, 3]]

    ref = pytorch_reference(
        x, r,
        torch.tensor(rows, dtype=torch.int32, device='cuda'),
        torch.tensor([v + [0] * (MAX_VECS - len(v)) for v in vecs_per_row],
                     dtype=torch.int32, device='cuda'),
        len(rows),
        torch.tensor([len(v) for v in vecs_per_row], dtype=torch.int32, device='cuda'),
    )

    token_indices, vec_indices, n_valid_buf, n_vecs_per_token = make_buffers(n_tokens, MAX_VECS)
    set_buffers(token_indices, vec_indices, n_valid_buf, n_vecs_per_token, rows, vecs_per_row)
    out = subtract_summed_vectors_per_token(
        x.clone(), r, token_indices, vec_indices, n_valid_buf, n_vecs_per_token)

    assert torch.allclose(out, ref, atol=TOLERANCE), \
        f"Correctness fail (different vecs per token): max diff = {(out - ref).abs().max()}"
    print("PASS different vectors per token")


def test_single_vec_equals_direct_subtraction():
    """With one vector per token, result must equal x[row] - r[vec_idx] exactly."""
    n_tokens, n_selected = 16, 4
    x = torch.randn(n_tokens, HIDDEN_SIZE, device='cuda', dtype=torch.float16)
    r = torch.randn(MAX_VECS, HIDDEN_SIZE, device='cuda', dtype=torch.float16)

    rows         = list(range(n_selected))
    vecs_per_row = [[i] for i in range(n_selected)]

    token_indices, vec_indices, n_valid_buf, n_vecs_per_token = make_buffers(n_tokens, MAX_VECS)
    set_buffers(token_indices, vec_indices, n_valid_buf, n_vecs_per_token, rows, vecs_per_row)
    out = subtract_summed_vectors_per_token(
        x.clone(), r, token_indices, vec_indices, n_valid_buf, n_vecs_per_token)

    for i, (row, vec_idx) in enumerate(zip(rows, range(n_selected))):
        expected = (x[row].to(torch.float32) - r[vec_idx].to(torch.float32)).to(torch.float16)
        assert torch.allclose(out[row], expected, atol=TOLERANCE), \
            f"Direct subtraction mismatch at row {row}: max diff = {(out[row] - expected).abs().max()}"
    print("PASS single vec equals direct subtraction")


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
    subtract_summed_vectors_per_token(
        x, r, token_indices, vec_indices, n_valid_buf, n_vecs_per_token)

    unselected = list(range(n_selected, n_tokens))
    assert torch.equal(x[unselected], x_orig[unselected]), \
        "Unselected rows were modified"
    print("PASS unselected rows unchanged")


def test_padding_tokens_ignored():
    """Padding slots in token_indices beyond n_valid must not affect output."""
    n_tokens, n_selected = 32, 4
    x = torch.randn(n_tokens, HIDDEN_SIZE, device='cuda', dtype=torch.float16)
    r = torch.randn(MAX_VECS, HIDDEN_SIZE, device='cuda', dtype=torch.float16)

    rows         = [0, 1, 2, 3]
    vecs_per_row = [[0]] * 4

    token_indices, vec_indices, n_valid_buf, n_vecs_per_token = make_buffers(n_tokens, MAX_VECS)
    set_buffers(token_indices, vec_indices, n_valid_buf, n_vecs_per_token, rows, vecs_per_row)
    token_indices[n_selected:] = 10   # poison: if processed, row 10 would change

    x_orig = x.clone()
    subtract_summed_vectors_per_token(
        x, r, token_indices, vec_indices, n_valid_buf, n_vecs_per_token)

    assert torch.equal(x[10], x_orig[10]), \
        "Padding token slot was processed — row 10 incorrectly modified"
    print("PASS padding tokens ignored")


def test_padding_vectors_ignored():
    """Vector slots beyond n_vecs_per_token must not affect output."""
    n_tokens = 32
    x = torch.randn(n_tokens, HIDDEN_SIZE, device='cuda', dtype=torch.float16)
    r = torch.randn(MAX_VECS, HIDDEN_SIZE, device='cuda', dtype=torch.float16)

    rows         = [0]
    vecs_per_row = [[0]]   # only vec 0 is valid

    token_indices, vec_indices, n_valid_buf, n_vecs_per_token = make_buffers(n_tokens, MAX_VECS)
    set_buffers(token_indices, vec_indices, n_valid_buf, n_vecs_per_token, rows, vecs_per_row)
    vec_indices[0, 1:] = 2   # poison unused slots with vec 2

    # Reference: only vec 0 subtracted
    expected = (x[0].to(torch.float32) - r[0].to(torch.float32)).to(torch.float16)
    out = subtract_summed_vectors_per_token(
        x.clone(), r, token_indices, vec_indices, n_valid_buf, n_vecs_per_token)

    assert torch.allclose(out[0], expected, atol=TOLERANCE), \
        f"Padding vector slot was processed: max diff = {(out[0] - expected).abs().max()}"
    print("PASS padding vectors ignored")


def test_zero_vecs_leaves_token_unchanged():
    """A token with n_vecs_per_token=0 must not be modified."""
    n_tokens = 16
    x = torch.randn(n_tokens, HIDDEN_SIZE, device='cuda', dtype=torch.float16)
    r = torch.randn(MAX_VECS, HIDDEN_SIZE, device='cuda', dtype=torch.float16)

    # Two tokens selected, but first has 0 vectors
    rows         = [0, 1]
    vecs_per_row = [[], [0]]   # token 0: no vecs, token 1: vec 0

    x_orig = x.clone()
    token_indices, vec_indices, n_valid_buf, n_vecs_per_token = make_buffers(n_tokens, MAX_VECS)
    set_buffers(token_indices, vec_indices, n_valid_buf, n_vecs_per_token, rows, vecs_per_row)
    subtract_summed_vectors_per_token(
        x, r, token_indices, vec_indices, n_valid_buf, n_vecs_per_token)

    assert torch.equal(x[0], x_orig[0]), \
        "Token with 0 vectors was modified"
    print("PASS zero vecs leaves token unchanged")


def test_subtraction_magnitude():
    """Subtracted combined vector has the expected magnitude."""
    n_tokens = 8
    x = torch.zeros(n_tokens, HIDDEN_SIZE, device='cuda', dtype=torch.float16)
    # Use identical unit vectors so combined = n_vecs * r[0] for easy checking
    r = torch.zeros(MAX_VECS, HIDDEN_SIZE, device='cuda', dtype=torch.float16)
    r[:, 0] = 1.0   # each vector is e_0 (unit along first dimension)

    rows         = [0]
    vecs_per_row = [[0, 1, 2]]   # 3 vectors, each is e_0

    token_indices, vec_indices, n_valid_buf, n_vecs_per_token = make_buffers(n_tokens, MAX_VECS)
    set_buffers(token_indices, vec_indices, n_valid_buf, n_vecs_per_token, rows, vecs_per_row)
    out = subtract_summed_vectors_per_token(
        x.clone(), r, token_indices, vec_indices, n_valid_buf, n_vecs_per_token)

    # x[0] was 0, combined = 3 * e_0, so out[0, 0] should be -3
    assert abs(out[0, 0].item() - (-3.0)) < TOLERANCE, \
        f"Magnitude wrong: expected -3.0, got {out[0, 0].item()}"
    assert torch.equal(out[0, 1:], x[0, 1:]), \
        "Non-zero dimension incorrectly modified"
    print("PASS subtraction magnitude correct")


if __name__ == "__main__":
    test_correctness_single_vec(n_tokens=32,  n_selected=8)
    test_correctness_single_vec(n_tokens=128, n_selected=64)
    test_correctness_multi_vec(n_tokens=32,  n_selected=8,  n_vecs=2)
    test_correctness_multi_vec(n_tokens=64,  n_selected=16, n_vecs=MAX_VECS)
    test_different_vecs_per_token()
    test_single_vec_equals_direct_subtraction()
    test_unselected_rows_unchanged()
    test_padding_tokens_ignored()
    test_padding_vectors_ignored()
    test_zero_vecs_leaves_token_unchanged()
    test_subtraction_magnitude()
    print("\nAll tests passed.")
