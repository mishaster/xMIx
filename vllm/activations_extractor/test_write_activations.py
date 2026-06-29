# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from write_activations import add_vector_to_activations
from vllm.activations_extractor.write_activations import (
    EOLTokenDetector,
    compact_mask_to_indices,
)

HIDDEN_SIZE = 4096
TOLERANCE = 1e-3  # float16 has limited precision


def test_2d(n_tokens: int):
    x = torch.zeros(n_tokens, HIDDEN_SIZE, device='cuda', dtype=torch.float16)
    y = torch.full((HIDDEN_SIZE,), 0.1, device='cuda', dtype=torch.float16)
    out = add_vector_to_activations(x, y)

    assert out.shape == (n_tokens, HIDDEN_SIZE), f"Shape mismatch: {out.shape}"
    assert torch.allclose(out, torch.full_like(out, 0.1), atol=TOLERANCE), \
        f"Values wrong: min={out.min()}, max={out.max()}"
    print(f"PASS 2D ({n_tokens}, {HIDDEN_SIZE})")


def test_3d(batch: int, seq: int):
    x = torch.zeros(batch, seq, HIDDEN_SIZE, device='cuda', dtype=torch.float16)
    y = torch.full((HIDDEN_SIZE,), 0.1, device='cuda', dtype=torch.float16)
    out = add_vector_to_activations(x, y)

    assert out.shape == (batch, seq, HIDDEN_SIZE), f"Shape mismatch: {out.shape}"
    assert torch.allclose(out, torch.full_like(out, 0.1), atol=TOLERANCE), \
        f"Values wrong: min={out.min()}, max={out.max()}"
    print(f"PASS 3D ({batch}, {seq}, {HIDDEN_SIZE})")


def test_all_tokens_touched(n_tokens: int):
    # Set each row to its row index, then verify every row was incremented
    x = torch.zeros(n_tokens, HIDDEN_SIZE, device='cuda', dtype=torch.float16)
    for i in range(n_tokens):
        x[i, :] = float(i)
    y = torch.full((HIDDEN_SIZE,), 0.1, device='cuda', dtype=torch.float16)
    out = add_vector_to_activations(x, y)

    for i in range(n_tokens):
        expected = i + 0.1
        assert torch.allclose(
            out[i],
            torch.full((HIDDEN_SIZE,), expected, device='cuda', dtype=torch.float16),
            atol=TOLERANCE
        ), f"Token {i} not correctly updated: got {out[i, 0]}, expected {expected}"
    print(f"PASS all {n_tokens} tokens touched")


# ── helpers ───────────────────────────────────────────────────────────────────

EOL_IDS = [10, 20, 30, 100, 200, 999]
VOCAB   = 1024
MAX_TOK = 128


def _make_detector():
    return EOLTokenDetector(EOL_IDS, vocab_size=VOCAB,
                            max_tokens=MAX_TOK, device="cuda")


def _mask_set(det):
    """Set of positions where the detector's owned mask is 1."""
    return {i for i, v in enumerate(det.mask.cpu().tolist()) if v}


def _buf_set(buf):
    """Set of positions where an external sink buffer is 1."""
    return {i for i, v in enumerate(buf.cpu().tolist()) if v}


def _compact(mask):
    n   = mask.shape[0]
    out = torch.zeros(n, dtype=torch.int32, device="cuda")
    cnt = torch.zeros(1, dtype=torch.int32, device="cuda")
    compact_mask_to_indices(mask, out, cnt)
    count  = cnt[0].item()
    result = set(out[:count].cpu().tolist())
    return result, count


# ── compact_mask_to_indices tests ─────────────────────────────────────────────

def test_compact_no_matches():
    mask = torch.zeros(16, dtype=torch.bool, device="cuda")
    _, count = _compact(mask)
    assert count == 0
    print("PASS compact_no_matches")


def test_compact_all_matches():
    mask = torch.ones(8, dtype=torch.bool, device="cuda")
    indices, count = _compact(mask)
    assert count == 8
    assert indices == set(range(8))
    print("PASS compact_all_matches")


def test_compact_single_match():
    mask = torch.zeros(16, dtype=torch.bool, device="cuda")
    mask[5] = True
    indices, count = _compact(mask)
    assert count == 1
    assert indices == {5}
    print("PASS compact_single_match")


def test_compact_boundary_positions():
    mask = torch.zeros(64, dtype=torch.bool, device="cuda")
    mask[0] = mask[63] = True
    indices, count = _compact(mask)
    assert count == 2
    assert indices == {0, 63}
    print("PASS compact_boundary_positions")


def test_compact_matches_reference():
    true_pos = [1, 7, 33, 63]
    mask = torch.zeros(64, dtype=torch.bool, device="cuda")
    mask[true_pos] = True
    indices, count = _compact(mask)
    ref = set(mask.cpu().nonzero(as_tuple=True)[0].tolist())
    assert count == len(ref)
    assert indices == ref
    print("PASS compact_matches_reference")


def test_compact_buffer_address_stable():
    mask = torch.zeros(8, dtype=torch.bool, device="cuda")
    mask[2] = mask[5] = True
    out  = torch.zeros(8, dtype=torch.int32, device="cuda")
    cnt  = torch.zeros(1, dtype=torch.int32, device="cuda")
    addr = cnt.data_ptr()
    compact_mask_to_indices(mask, out, cnt)
    assert cnt.data_ptr() == addr
    assert cnt[0].item() == 2
    print("PASS compact_buffer_address_stable")


# ── EOLTokenDetector tests ────────────────────────────────────────────────────

def test_eol_no_matches():
    det = _make_detector()
    det(torch.tensor([1, 2, 3], dtype=torch.int32, device="cuda"))
    assert _mask_set(det) == set()
    print("PASS eol_no_matches")


def test_eol_all_match():
    det = _make_detector()
    det(torch.tensor(EOL_IDS, dtype=torch.int32, device="cuda"))
    assert _mask_set(det) == set(range(len(EOL_IDS)))
    print("PASS eol_all_match")


def test_eol_mixed():
    det = _make_detector()
    # positions 1 and 3 are EOL
    det(torch.tensor([1, 10, 5, 20, 7], dtype=torch.int32, device="cuda"))
    assert _mask_set(det) == {1, 3}
    print("PASS eol_mixed")


def test_eol_boundary():
    det = _make_detector()
    det(torch.tensor([10, 1, 2, 999], dtype=torch.int32, device="cuda"))
    assert _mask_set(det) == {0, 3}
    print("PASS eol_boundary")


def test_eol_out_of_vocab():
    det = _make_detector()
    det(torch.tensor([VOCAB, VOCAB + 1, 5000], dtype=torch.int32, device="cuda"))
    assert _mask_set(det) == set()
    print("PASS eol_out_of_vocab")


def test_eol_matches_reference():
    torch.manual_seed(42)
    det = _make_detector()
    eol_set = set(EOL_IDS)
    for n in [1, 7, 16, 64, 128]:
        ids = torch.randint(0, VOCAB, (n,), dtype=torch.int32, device="cuda")
        det(ids)
        ref = {i for i, t in enumerate(ids.cpu().tolist()) if t in eol_set}
        assert _mask_set(det) == ref, f"n={n}: mask mismatch vs ref"
    print("PASS eol_matches_reference (sizes 1,7,16,64,128)")


def test_eol_buffer_address_stable():
    det = _make_detector()
    ids = torch.tensor([10, 1, 20], dtype=torch.int32, device="cuda")
    addr_mask = det.mask.data_ptr()
    for _ in range(5):
        det(ids)
    assert det.mask.data_ptr() == addr_mask
    print("PASS eol_buffer_address_stable")


def test_eol_padding_zeroed():
    """
    Padding slots [n_tokens, max_tokens) must be wiped on every call so
    stale 1s from a previous call cannot leak into the next forward pass.
    """
    det = _make_detector()

    # First call: every position is an EOL hit. Mask is dense 1s in [0, 6).
    det(torch.tensor(EOL_IDS, dtype=torch.int32, device="cuda"))
    assert _mask_set(det) == set(range(len(EOL_IDS)))
    # Padding past n_tokens must already be zero.
    assert all(v == 0 for v in det.mask[len(EOL_IDS):].cpu().tolist())

    # Second call with a shorter, all-non-EOL input. The previously-set 1s
    # in slots [0, 6) must be wiped, and the rest must remain zero.
    det(torch.tensor([1, 2, 3], dtype=torch.int32, device="cuda"))
    assert _mask_set(det) == set()
    print("PASS eol_padding_zeroed")


def test_eol_multi_sink():
    """
    set_output_buffers replaces the sink list. After it is called, every
    forward writes the same mask into each external buffer, and the
    detector's owned `mask` is no longer touched.
    """
    det = _make_detector()
    buf_a = torch.zeros(MAX_TOK, dtype=torch.int32, device="cuda")
    buf_b = torch.zeros(MAX_TOK, dtype=torch.int32, device="cuda")
    det.set_output_buffers([buf_a, buf_b])

    ids = torch.tensor([1, 10, 5, 20, 7], dtype=torch.int32, device="cuda")
    det(ids)

    expected = {1, 3}
    assert _buf_set(buf_a) == expected
    assert _buf_set(buf_b) == expected
    assert torch.equal(buf_a, buf_b)
    # Owned mask is no longer in the sink list, so it stays as-initialized.
    assert _mask_set(det) == set()
    print("PASS eol_multi_sink")


def test_eol_lookup_not_modified():
    det  = _make_detector()
    snap = det.lookup.clone()
    det(torch.tensor([10, 20, 999], dtype=torch.int32, device="cuda"))
    assert torch.equal(det.lookup, snap)
    print("PASS eol_lookup_not_modified")


if __name__ == "__main__":
    test_2d(n_tokens=1)        # decode: single token
    test_2d(n_tokens=128)      # small prefill
    test_2d(n_tokens=2048)     # large prefill
    test_3d(batch=4, seq=512)  # 3D input
    test_all_tokens_touched(n_tokens=32)
    print("\n--- compact_mask_to_indices ---")
    test_compact_no_matches()
    test_compact_all_matches()
    test_compact_single_match()
    test_compact_boundary_positions()
    test_compact_matches_reference()
    test_compact_buffer_address_stable()
    print("\n--- EOLTokenDetector ---")
    test_eol_no_matches()
    test_eol_all_match()
    test_eol_mixed()
    test_eol_boundary()
    test_eol_out_of_vocab()
    test_eol_matches_reference()
    test_eol_buffer_address_stable()
    test_eol_padding_zeroed()
    test_eol_multi_sink()
    test_eol_lookup_not_modified()
    print("\nAll tests passed.")
