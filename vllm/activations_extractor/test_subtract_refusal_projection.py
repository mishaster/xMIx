# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from vllm.activations_extractor.write_activations import (
    SteeringVectorDotSubtractNormalized,
    subtract_refusal_projection,
)

HIDDEN_SIZE = 4096
TOLERANCE = 1e-2  # float16 limited precision


def _run_module(x: torch.Tensor, r_normalized: torch.Tensor) -> torch.Tensor:
    """Run SteeringVectorDotSubtractNormalized on all rows of x."""
    n_rows = x.shape[0]
    mod = SteeringVectorDotSubtractNormalized(
        hidden_size=HIDDEN_SIZE,
        max_tokens=n_rows,
        dtype=x.dtype,
        device=x.device,
    )
    mod.r.copy_(r_normalized.unsqueeze(0))
    mod.token_indices.copy_(
        torch.arange(n_rows, dtype=torch.int32, device=x.device)
    )
    mod.n_valid_buf.copy_(
        torch.tensor([n_rows], dtype=torch.int32, device=x.device)
    )
    return mod(x)


def test_equivalence(n_rows: int, seed: int = 42):
    torch.manual_seed(seed)
    device = "cuda"
    dtype = torch.float16

    x_ref = torch.randn(n_rows, HIDDEN_SIZE, device=device, dtype=dtype)
    x_mod = x_ref.clone()
    r = torch.randn(HIDDEN_SIZE, device=device, dtype=dtype)
    r_normalized = r / r.norm()

    subtract_refusal_projection(x_ref, r)
    _run_module(x_mod, r_normalized)

    assert torch.allclose(x_ref, x_mod, atol=TOLERANCE), (
        f"n_rows={n_rows}: max diff = {(x_ref - x_mod).abs().max().item():.6f}"
    )
    print(f"PASS equivalence n_rows={n_rows}")


if __name__ == "__main__":
    for n in [1, 8, 64, 512]:
        test_equivalence(n)
    print("\nAll tests passed.")
