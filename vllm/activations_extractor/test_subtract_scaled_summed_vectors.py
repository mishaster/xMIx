# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import torch
from write_activations import add_scaled_summed_vectors_per_token

HIDDEN_SIZE = 4096
MAX_VECS    = 4
TOLERANCE   = 1e-2
DUMP_FILE   = "dump_scaled_summed_vectors.txt"


def dump(f, title, tensor, n_rows=None, n_cols=None):
    """
    Write a labelled tensor to file f with full precision, no truncation.
    n_rows / n_cols: if set, only print that many rows/columns (useful for
    large hidden sizes — set n_cols=16 to see just the first 16 values).
    """
    torch.set_printoptions(threshold=torch.inf, linewidth=200, precision=4)
    t = tensor.cpu().float()
    if t.dim() == 2:
        if n_rows is not None: t = t[:n_rows]
        if n_cols is not None: t = t[:, :n_cols]
    elif t.dim() == 1:
        if n_cols is not None: t = t[:n_cols]
    f.write(f"\n{'='*60}\n{title}\nshape={tensor.shape}  dtype={tensor.dtype}\n{'='*60}\n")
    f.write(str(t))
    f.write("\n")
    torch.set_printoptions()   # reset to defaults


def dump_test_case(label, x, r, scales, token_indices, vec_indices,
                   n_valid, n_vecs_per_token, ref, out, n_cols=16):
    """Write all tensors for one test case to DUMP_FILE."""
    with open(DUMP_FILE, "a") as f:
        f.write(f"\n\n{'#'*60}\n# {label}\n{'#'*60}\n")
        dump(f, "x (input)",            x,               n_cols=n_cols)
        dump(f, "r (steering library)", r,               n_cols=n_cols)
        dump(f, "scales",               scales)
        dump(f, "token_indices",        token_indices)
        dump(f, "vec_indices",          vec_indices)
        dump(f, "n_vecs_per_token",     n_vecs_per_token)
        dump(f, "n_valid",              n_valid)
        dump(f, "reference output",     ref,             n_cols=n_cols)
        dump(f, "kernel output",        out,             n_cols=n_cols)
        diff = (out.float() - ref.float())
        dump(f, "diff (kernel - ref)",  diff,            n_cols=n_cols)
        f.write(f"\nmax_abs_diff = {diff.abs().max().item():.6f}\n")


# ---------------------------------------------------------------------------
# PyTorch reference
# ---------------------------------------------------------------------------

def pytorch_reference(x, r, scales, token_indices, vec_indices, n_valid, n_vecs_per_token):
    """For each selected token, compute Σ_v scale_v * r_v and subtract from x."""
    x = x.clone()
    for i in range(n_valid):
        row    = token_indices[i].item()
        n_vecs = n_vecs_per_token[i].item()
        combined = torch.zeros(r.shape[1], device=r.device, dtype=torch.float32)
        for v in range(n_vecs):
            vec_idx = vec_indices[i, v].item()
            scale   = scales[i, v].item()
            combined += scale * r[vec_idx].to(torch.float32)
        x[row] = (x[row].to(torch.float32) + combined).to(x.dtype)
    return x


# ---------------------------------------------------------------------------
# Buffer helpers
# ---------------------------------------------------------------------------

def make_buffers(max_tokens, max_vecs):
    token_indices    = torch.zeros(max_tokens,           dtype=torch.int32,   device='cuda')
    vec_indices      = torch.zeros(max_tokens, max_vecs, dtype=torch.int32,   device='cuda')
    scales           = torch.zeros(max_tokens, max_vecs, dtype=torch.float32, device='cuda')
    n_valid_buf      = torch.zeros(1,                    dtype=torch.int32,   device='cuda')
    n_vecs_per_token = torch.zeros(max_tokens,           dtype=torch.int32,   device='cuda')
    return token_indices, vec_indices, scales, n_valid_buf, n_vecs_per_token


def set_buffers(token_indices, vec_indices, scales, n_valid_buf, n_vecs_per_token,
                rows, vecs_per_row, scales_per_row):
    n = len(rows)
    token_indices[:n] = torch.tensor(rows, dtype=torch.int32, device='cuda')
    n_valid_buf.fill_(n)
    for i, (vecs, sc) in enumerate(zip(vecs_per_row, scales_per_row)):
        nv = len(vecs)
        vec_indices[i, :nv]  = torch.tensor(vecs, dtype=torch.int32,   device='cuda')
        scales[i, :nv]       = torch.tensor(sc,   dtype=torch.float32, device='cuda')
        n_vecs_per_token[i]  = nv


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_correctness_single_vec(n_tokens, n_selected):
    """Single vector per token with a scalar scale — matches reference."""
    x = torch.randn(n_tokens, HIDDEN_SIZE, device='cuda', dtype=torch.float16)
    r = torch.randn(MAX_VECS, HIDDEN_SIZE, device='cuda', dtype=torch.float16)

    rows         = list(range(n_selected))
    vecs_per_row = [[i % MAX_VECS] for i in range(n_selected)]
    scales_vals  = [[float(i + 1) * 0.5] for i in range(n_selected)]  # 0.5, 1.0, 1.5, ...

    token_indices, vec_indices, scales, n_valid_buf, n_vecs_per_token = \
        make_buffers(n_tokens, MAX_VECS)
    set_buffers(token_indices, vec_indices, scales, n_valid_buf, n_vecs_per_token,
                rows, vecs_per_row, scales_vals)

    ref = pytorch_reference(x, r, scales,
                            token_indices, vec_indices, n_selected, n_vecs_per_token)
    out = add_scaled_summed_vectors_per_token(
        x.clone(), r, scales, token_indices, vec_indices, n_valid_buf, n_vecs_per_token)

    dump_test_case(
        f"correctness_single_vec ({n_tokens} tokens, {n_selected} selected)",
        x, r, scales, token_indices, vec_indices, n_valid_buf, n_vecs_per_token, ref, out)

    assert torch.allclose(out, ref, atol=TOLERANCE), \
        f"Correctness fail (single vec): max diff = {(out - ref).abs().max()}"
    print(f"PASS correctness single vec ({n_tokens} tokens, {n_selected} selected)")


def test_correctness_multi_vec(n_tokens, n_selected, n_vecs):
    """Multiple vectors per token each with its own scale — matches reference."""
    assert n_vecs <= MAX_VECS
    x = torch.randn(n_tokens, HIDDEN_SIZE, device='cuda', dtype=torch.float16)
    r = torch.randn(MAX_VECS, HIDDEN_SIZE, device='cuda', dtype=torch.float16)

    rows         = list(range(n_selected))
    vecs_per_row = [list(range(n_vecs)) for _ in range(n_selected)]
    scales_vals  = [[0.25 * (v + 1) for v in range(n_vecs)] for _ in range(n_selected)]

    token_indices, vec_indices, scales, n_valid_buf, n_vecs_per_token = \
        make_buffers(n_tokens, MAX_VECS)
    set_buffers(token_indices, vec_indices, scales, n_valid_buf, n_vecs_per_token,
                rows, vecs_per_row, scales_vals)

    ref = pytorch_reference(x, r, scales,
                            token_indices, vec_indices, n_selected, n_vecs_per_token)
    out = add_scaled_summed_vectors_per_token(
        x.clone(), r, scales, token_indices, vec_indices, n_valid_buf, n_vecs_per_token)

    dump_test_case(
        f"correctness_multi_vec ({n_tokens} tokens, {n_selected} selected, {n_vecs} vecs each)",
        x, r, scales, token_indices, vec_indices, n_valid_buf, n_vecs_per_token, ref, out)

    assert torch.allclose(out, ref, atol=TOLERANCE), \
        f"Correctness fail (multi vec): max diff = {(out - ref).abs().max()}"
    print(f"PASS correctness multi vec ({n_tokens} tokens, {n_selected} selected, {n_vecs} vecs each)")


def test_scale_zero_leaves_token_unchanged():
    """A token whose all scales are 0 must not be modified."""
    n_tokens = 16
    x = torch.randn(n_tokens, HIDDEN_SIZE, device='cuda', dtype=torch.float16)
    r = torch.randn(MAX_VECS, HIDDEN_SIZE, device='cuda', dtype=torch.float16)

    rows         = [0, 1]
    vecs_per_row = [[0, 1], [0, 1]]
    scales_vals  = [[0.0, 0.0], [1.0, 1.0]]  # token 0: all zeros, token 1: normal

    token_indices, vec_indices, scales, n_valid_buf, n_vecs_per_token = \
        make_buffers(n_tokens, MAX_VECS)
    set_buffers(token_indices, vec_indices, scales, n_valid_buf, n_vecs_per_token,
                rows, vecs_per_row, scales_vals)

    ref    = pytorch_reference(x, r, scales,
                               token_indices, vec_indices, len(rows), n_vecs_per_token)
    x_orig = x.clone()
    out    = add_scaled_summed_vectors_per_token(
        x.clone(), r, scales, token_indices, vec_indices, n_valid_buf, n_vecs_per_token)

    dump_test_case(
        "scale_zero_leaves_token_unchanged",
        x, r, scales, token_indices, vec_indices, n_valid_buf, n_vecs_per_token, ref, out)

    assert torch.equal(out[0], x_orig[0]), \
        "Token with all-zero scales was modified"
    print("PASS scale=0 leaves token unchanged")


def test_scale_one_matches_reference():
    """With all scales=1 the result must match the pytorch reference (pure addition)."""
    n_tokens, n_selected, n_vecs = 32, 8, 3
    x = torch.randn(n_tokens, HIDDEN_SIZE, device='cuda', dtype=torch.float16)
    r = torch.randn(MAX_VECS, HIDDEN_SIZE, device='cuda', dtype=torch.float16)

    rows         = list(range(n_selected))
    vecs_per_row = [list(range(n_vecs))] * n_selected
    scales_vals  = [[1.0] * n_vecs] * n_selected

    token_indices, vec_indices, scales, n_valid_buf, n_vecs_per_token = \
        make_buffers(n_tokens, MAX_VECS)
    set_buffers(token_indices, vec_indices, scales, n_valid_buf, n_vecs_per_token,
                rows, vecs_per_row, scales_vals)

    ref = pytorch_reference(x, r, scales,
                            token_indices, vec_indices, n_selected, n_vecs_per_token)
    out = add_scaled_summed_vectors_per_token(
        x.clone(), r, scales, token_indices, vec_indices, n_valid_buf, n_vecs_per_token)

    dump_test_case(
        "scale_one_matches_reference",
        x, r, scales, token_indices, vec_indices, n_valid_buf, n_vecs_per_token, ref, out)

    assert torch.allclose(out, ref, atol=TOLERANCE), \
        f"Scale=1 mismatch vs reference: max diff = {(out - ref).abs().max()}"
    print("PASS scale=1 matches reference")


def test_scale_magnitude():
    """Doubling all scales doubles the added amount."""
    n_tokens = 8
    x = torch.zeros(n_tokens, HIDDEN_SIZE, device='cuda', dtype=torch.float16)
    r = torch.zeros(MAX_VECS, HIDDEN_SIZE, device='cuda', dtype=torch.float16)
    r[0, 0] = 1.0   # r[0] = e_0

    rows         = [0]
    vecs_per_row = [[0]]

    # scale = 1.0 → out[0, 0] = +1.0
    token_indices, vec_indices, scales, n_valid_buf, n_vecs_per_token = \
        make_buffers(n_tokens, MAX_VECS)
    set_buffers(token_indices, vec_indices, scales, n_valid_buf, n_vecs_per_token,
                rows, vecs_per_row, [[1.0]])
    ref1 = pytorch_reference(x, r, scales,
                             token_indices, vec_indices, len(rows), n_vecs_per_token)
    out1 = add_scaled_summed_vectors_per_token(
        x.clone(), r, scales, token_indices, vec_indices, n_valid_buf, n_vecs_per_token)
    dump_test_case(
        "scale_magnitude scale=1.0",
        x, r, scales, token_indices, vec_indices, n_valid_buf, n_vecs_per_token, ref1, out1)

    # scale = 2.0 → out[0, 0] = +2.0
    scales.zero_()
    set_buffers(token_indices, vec_indices, scales, n_valid_buf, n_vecs_per_token,
                rows, vecs_per_row, [[2.0]])
    ref2 = pytorch_reference(x, r, scales,
                             token_indices, vec_indices, len(rows), n_vecs_per_token)
    out2 = add_scaled_summed_vectors_per_token(
        x.clone(), r, scales, token_indices, vec_indices, n_valid_buf, n_vecs_per_token)
    dump_test_case(
        "scale_magnitude scale=2.0",
        x, r, scales, token_indices, vec_indices, n_valid_buf, n_vecs_per_token, ref2, out2)

    assert abs(out1[0, 0].item() - 1.0) < TOLERANCE, \
        f"scale=1 wrong: expected +1.0, got {out1[0, 0].item()}"
    assert abs(out2[0, 0].item() - 2.0) < TOLERANCE, \
        f"scale=2 wrong: expected +2.0, got {out2[0, 0].item()}"
    print("PASS scale magnitude correct (1x and 2x)")


def test_different_scales_per_token():
    """Each token uses a different scale — all match reference independently."""
    n_tokens = 32
    x = torch.randn(n_tokens, HIDDEN_SIZE, device='cuda', dtype=torch.float16)
    r = torch.randn(MAX_VECS, HIDDEN_SIZE, device='cuda', dtype=torch.float16)

    rows         = [0, 1, 2, 3]
    vecs_per_row = [[0], [0], [0], [0]]
    scales_vals  = [[0.1], [0.5], [1.0], [2.0]]

    token_indices, vec_indices, scales, n_valid_buf, n_vecs_per_token = \
        make_buffers(n_tokens, MAX_VECS)
    set_buffers(token_indices, vec_indices, scales, n_valid_buf, n_vecs_per_token,
                rows, vecs_per_row, scales_vals)

    ref = pytorch_reference(x, r, scales,
                            token_indices, vec_indices, len(rows), n_vecs_per_token)
    out = add_scaled_summed_vectors_per_token(
        x.clone(), r, scales, token_indices, vec_indices, n_valid_buf, n_vecs_per_token)

    dump_test_case(
        "different_scales_per_token (scales: 0.1, 0.5, 1.0, 2.0)",
        x, r, scales, token_indices, vec_indices, n_valid_buf, n_vecs_per_token, ref, out)

    assert torch.allclose(out, ref, atol=TOLERANCE), \
        f"Different scales per token fail: max diff = {(out - ref).abs().max()}"
    print("PASS different scales per token")


def test_unselected_rows_unchanged():
    """Rows not in token_indices must be completely untouched."""
    n_tokens, n_selected = 32, 4
    x = torch.randn(n_tokens, HIDDEN_SIZE, device='cuda', dtype=torch.float16)
    r = torch.randn(MAX_VECS, HIDDEN_SIZE, device='cuda', dtype=torch.float16)

    rows         = list(range(n_selected))
    vecs_per_row = [[0, 1]] * n_selected
    scales_vals  = [[0.5, 1.5]] * n_selected

    token_indices, vec_indices, scales, n_valid_buf, n_vecs_per_token = \
        make_buffers(n_tokens, MAX_VECS)
    set_buffers(token_indices, vec_indices, scales, n_valid_buf, n_vecs_per_token,
                rows, vecs_per_row, scales_vals)

    ref    = pytorch_reference(x, r, scales,
                               token_indices, vec_indices, n_selected, n_vecs_per_token)
    x_orig = x.clone()
    out    = add_scaled_summed_vectors_per_token(
        x.clone(), r, scales, token_indices, vec_indices, n_valid_buf, n_vecs_per_token)

    dump_test_case(
        "unselected_rows_unchanged",
        x, r, scales, token_indices, vec_indices, n_valid_buf, n_vecs_per_token, ref, out)

    unselected = list(range(n_selected, n_tokens))
    assert torch.equal(out[unselected], x_orig[unselected]), \
        "Unselected rows were modified"
    print("PASS unselected rows unchanged")


def test_padding_tokens_ignored():
    """Padding slots in token_indices beyond n_valid must not affect output."""
    n_tokens, n_selected = 32, 4
    x = torch.randn(n_tokens, HIDDEN_SIZE, device='cuda', dtype=torch.float16)
    r = torch.randn(MAX_VECS, HIDDEN_SIZE, device='cuda', dtype=torch.float16)

    rows         = [0, 1, 2, 3]
    vecs_per_row = [[0]] * 4
    scales_vals  = [[1.0]] * 4

    token_indices, vec_indices, scales, n_valid_buf, n_vecs_per_token = \
        make_buffers(n_tokens, MAX_VECS)
    set_buffers(token_indices, vec_indices, scales, n_valid_buf, n_vecs_per_token,
                rows, vecs_per_row, scales_vals)
    token_indices[n_selected:] = 10   # poison

    ref    = pytorch_reference(x, r, scales,
                               token_indices, vec_indices, n_selected, n_vecs_per_token)
    x_orig = x.clone()
    out    = add_scaled_summed_vectors_per_token(
        x.clone(), r, scales, token_indices, vec_indices, n_valid_buf, n_vecs_per_token)

    dump_test_case(
        "padding_tokens_ignored (poison row=10 beyond n_valid=4)",
        x, r, scales, token_indices, vec_indices, n_valid_buf, n_vecs_per_token, ref, out)

    assert torch.equal(out[10], x_orig[10]), \
        "Padding token slot processed — row 10 incorrectly modified"
    print("PASS padding tokens ignored")


def test_padding_vectors_ignored():
    """Scale slots beyond n_vecs_per_token must not affect output."""
    n_tokens = 16
    x = torch.randn(n_tokens, HIDDEN_SIZE, device='cuda', dtype=torch.float16)
    r = torch.randn(MAX_VECS, HIDDEN_SIZE, device='cuda', dtype=torch.float16)

    rows         = [0]
    vecs_per_row = [[0]]
    scales_vals  = [[1.0]]

    token_indices, vec_indices, scales, n_valid_buf, n_vecs_per_token = \
        make_buffers(n_tokens, MAX_VECS)
    set_buffers(token_indices, vec_indices, scales, n_valid_buf, n_vecs_per_token,
                rows, vecs_per_row, scales_vals)
    # Poison unused slots with large scales and non-zero vector indices
    scales[0, 1:]      = 999.0
    vec_indices[0, 1:] = 2

    ref = pytorch_reference(x, r, scales,
                            token_indices, vec_indices, 1, n_vecs_per_token)
    out = add_scaled_summed_vectors_per_token(
        x.clone(), r, scales, token_indices, vec_indices, n_valid_buf, n_vecs_per_token)

    dump_test_case(
        "padding_vectors_ignored (poison scales[0,1:]=999, vec_indices[0,1:]=2)",
        x, r, scales, token_indices, vec_indices, n_valid_buf, n_vecs_per_token, ref, out)

    # ref uses only vec 0 with scale 1.0 (n_vecs_per_token[0]=1), so poisoned slots
    # should have no effect; ref == out for the whole tensor
    assert torch.allclose(out, ref, atol=TOLERANCE), \
        f"Padding vector slot processed: max diff = {(out - ref).abs().max()}"
    print("PASS padding vectors ignored")


if __name__ == "__main__":
    # Start each run with a fresh dump file
    if os.path.exists(DUMP_FILE):
        os.remove(DUMP_FILE)

    test_correctness_single_vec(n_tokens=32,  n_selected=8)
    test_correctness_single_vec(n_tokens=128, n_selected=64)
    test_correctness_multi_vec(n_tokens=32,  n_selected=8,  n_vecs=2)
    test_correctness_multi_vec(n_tokens=64,  n_selected=16, n_vecs=MAX_VECS)
    test_scale_zero_leaves_token_unchanged()
    test_scale_one_matches_reference()
    test_scale_magnitude()
    test_different_scales_per_token()
    test_unselected_rows_unchanged()
    test_padding_tokens_ignored()
    test_padding_vectors_ignored()
    print("\nAll tests passed.")
    print(f"Tensor dumps written to: {DUMP_FILE}")
