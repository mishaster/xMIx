# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
EasySteer-inspired Triton steering kernels.

Four CUDA-graph-safe activation-manipulation classes:
  - SteeringVectorReplacer   — replace selected tokens with a fixed vector
  - SteeringConceptReplace   — h += dot(h, h1_norm) * (h2 - h1)
  - SteeringLinear           — h = W @ h + b  (cuBLAS + Triton scatter)
  - SteeringLMSteer          — h += alpha * (h @ P1) @ P2^T  (2 cuBLAS + Triton)

All classes follow the pattern from write_activations.py:
  • Buffers: register_parameter(..., requires_grad=False)
  • load_*() uses .copy_(), never reassignment
  • Per-token gate: token_indices (max_tokens,) int32 + n_valid_buf (1,) int32
  • Per-row gate:   vec_indices  (max_tokens,) int32 — 0=skip, non-zero=steer
  • BLOCK_H = triton.next_power_of_2(hidden_size), set at __init__
  • .run(x) is the public API
"""

import torch
from vllm.triton_utils import tl, triton


# ─────────────────────────────────────────────────────────────────────────────
# Replace
# ─────────────────────────────────────────────────────────────────────────────

@triton.jit
def replace_selected_kernel(
    x_ptr,            # (n_rows, hidden_size) — modified in-place
    v_ptr,            # (hidden_size,) replacement vector
    indices_ptr,      # (max_tokens,) int32 — selected row indices
    vec_indices_ptr,  # (n_rows,) int32 — 0 = skip, non-zero = replace
    n_valid_ptr,      # scalar int32
    hidden_size,
    BLOCK_H: tl.constexpr,
):
    """
    For each selected token where vec_indices[row] != 0:
        x[row, :] = v[:]
    Grid: (max_tokens,)
    """
    sel_idx = tl.program_id(0)
    n_valid = tl.load(n_valid_ptr)
    if sel_idx >= n_valid:
        return

    row  = tl.load(indices_ptr     + sel_idx)
    flag = tl.load(vec_indices_ptr + row)
    if flag == 0:
        return

    offs  = tl.arange(0, BLOCK_H)
    mask  = offs < hidden_size
    x_raw = tl.load(x_ptr + row * hidden_size + offs, mask=mask, other=0.0)
    v     = tl.load(v_ptr + offs,                     mask=mask, other=0.0)
    tl.store(x_ptr + row * hidden_size + offs, v.to(x_raw.dtype), mask=mask)


class SteeringVectorReplacer(torch.nn.Module):
    """
    Replaces selected token activations with a fixed replacement vector.

    For each token in token_indices[:n_valid_buf] where vec_indices[row] != 0:
        x[row, :] = v

    Usage
    ─────
    v  = ...                              # (hidden_size,) steering vector
    vi = ...                              # (max_tokens,) int32 gate flags
    replacer = SteeringVectorReplacer(v, vi, max_tokens=2048)
    replacer.load_vector(new_v)           # hot-swap vector
    replacer.load_vec_indices(new_vi)     # hot-swap gate flags
    replacer(x)
    """

    def __init__(
        self,
        v:           torch.Tensor,  # (hidden_size,) — must be 1-D
        vec_indices: torch.Tensor,  # (max_tokens,) int32
        max_tokens:  int,
    ):
        super().__init__()
        assert v.ndim == 1, f"v must be 1-D (hidden_size,), got shape {v.shape}"
        assert vec_indices.ndim == 1 and vec_indices.shape[0] == max_tokens, \
            f"vec_indices must be ({max_tokens},), got {vec_indices.shape}"
        assert vec_indices.dtype == torch.int32, "vec_indices must be int32"

        hidden_size = v.shape[0]
        device      = v.device

        self.hidden_size = hidden_size
        self.max_tokens  = max_tokens
        self._block_h    = triton.next_power_of_2(hidden_size)

        self.register_parameter('v',
            torch.nn.Parameter(v, requires_grad=False))

        self.register_parameter('token_indices',
            torch.nn.Parameter(
                torch.zeros(max_tokens, dtype=torch.int32, device=device),
                requires_grad=False))
        self.register_parameter('n_valid_buf',
            torch.nn.Parameter(
                torch.zeros(1, dtype=torch.int32, device=device),
                requires_grad=False))
        self.register_parameter('vec_indices',
            torch.nn.Parameter(vec_indices, requires_grad=False))

    def load_vector(self, v: torch.Tensor) -> None:
        """Hot-swap replacement vector.

        Args:
            v: (hidden_size,) — same shape as the vector passed at construction
        """
        assert v.shape == self.v.shape, \
            f"v shape mismatch: expected {self.v.shape}, got {v.shape}"
        self.v.copy_(v)

    def load_vec_indices(self, vi: torch.Tensor) -> None:
        """Copy per-row gate flags.

        Args:
            vi: (max_tokens,) int32 — 0 = skip, non-zero = replace
        """
        assert vi.shape == self.vec_indices.shape, \
            f"Shape mismatch: expected {self.vec_indices.shape}, got {vi.shape}"
        self.vec_indices.copy_(vi)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        replace_selected_kernel[(self.max_tokens,)](
            x, self.v, self.token_indices, self.vec_indices, self.n_valid_buf,
            self.hidden_size, BLOCK_H=self._block_h,
        )
        return x

    def run(self, x: torch.Tensor) -> torch.Tensor:
        return self(x)


# ─────────────────────────────────────────────────────────────────────────────
# Concept Replace
# ─────────────────────────────────────────────────────────────────────────────

@triton.jit
def concept_replace_kernel(
    x_ptr,            # (n_rows, hidden_size) — modified in-place
    h1_ptr,           # (hidden_size,) unit-norm "from" direction (caller-normalized)
    h2_ptr,           # (hidden_size,) "to" direction
    indices_ptr,      # (max_tokens,) int32 — selected row indices
    vec_indices_ptr,  # (n_rows,) int32 — 0 = skip, non-zero = steer
    n_valid_ptr,      # scalar int32
    hidden_size,
    BLOCK_H: tl.constexpr,
):
    """
    For each selected token where vec_indices[row] != 0:
        direction    = h2 - h1              # computed inline
        lam          = dot(x[row], h1)      # h1 is unit-norm (caller's responsibility)
        x[row, :]   += lam * direction
    Grid: (max_tokens,)
    """
    sel_idx = tl.program_id(0)
    n_valid = tl.load(n_valid_ptr)
    if sel_idx >= n_valid:
        return

    row  = tl.load(indices_ptr     + sel_idx)
    flag = tl.load(vec_indices_ptr + row)
    if flag == 0:
        return

    offs  = tl.arange(0, BLOCK_H)
    mask  = offs < hidden_size

    x_raw = tl.load(x_ptr   + row * hidden_size + offs, mask=mask, other=0.0)
    h1    = tl.load(h1_ptr   + offs,                    mask=mask, other=0.0).to(tl.float32)
    h2    = tl.load(h2_ptr   + offs,                    mask=mask, other=0.0).to(tl.float32)
    x     = x_raw.to(tl.float32)

    direction = h2 - h1
    lam    = tl.sum(x * h1, axis=0)
    result = x + lam * direction
    tl.store(x_ptr + row * hidden_size + offs, result.to(x_raw.dtype), mask=mask)


class SteeringConceptReplace(torch.nn.Module):
    """
    EasySteer ConceptReplace: steer activations from concept h1 toward concept h2.

    For each selected token where vec_indices[row] != 0:
        direction    = h2 - h1                     # computed in kernel
        lam          = dot(x[row], h1)             # h1 must be unit-norm
        x[row, :]   += lam * direction

    h1 MUST be unit-normalized by the caller before passing.

    Usage
    ─────
    h1 = raw_h1 / raw_h1.norm()           # caller normalizes
    steerer = SteeringConceptReplace(h1, h2, vi, max_tokens=2048)
    steerer.load_vectors(new_h1, new_h2)   # hot-swap (new_h1 must be unit-norm)
    steerer.load_vec_indices(new_vi)
    steerer(x)
    """

    def __init__(
        self,
        h1:          torch.Tensor,  # (hidden_size,) — must be unit-normalized by caller
        h2:          torch.Tensor,  # (hidden_size,)
        vec_indices: torch.Tensor,  # (max_tokens,) int32
        max_tokens:  int,
    ):
        super().__init__()
        assert h1.ndim == 1 and h2.ndim == 1 and h1.shape == h2.shape, \
            f"h1 and h2 must be 1-D with the same shape, got {h1.shape} vs {h2.shape}"
        assert vec_indices.ndim == 1 and vec_indices.shape[0] == max_tokens, \
            f"vec_indices must be ({max_tokens},), got {vec_indices.shape}"
        assert vec_indices.dtype == torch.int32, "vec_indices must be int32"

        hidden_size = h1.shape[0]
        device      = h1.device

        self.hidden_size = hidden_size
        self.max_tokens  = max_tokens
        self._block_h    = triton.next_power_of_2(hidden_size)

        self.register_parameter('h1',
            torch.nn.Parameter(h1, requires_grad=False))
        self.register_parameter('h2',
            torch.nn.Parameter(h2, requires_grad=False))

        self.register_parameter('token_indices',
            torch.nn.Parameter(
                torch.zeros(max_tokens, dtype=torch.int32, device=device),
                requires_grad=False))
        self.register_parameter('n_valid_buf',
            torch.nn.Parameter(
                torch.zeros(1, dtype=torch.int32, device=device),
                requires_grad=False))
        self.register_parameter('vec_indices',
            torch.nn.Parameter(vec_indices, requires_grad=False))

    def load_vectors(self, h1: torch.Tensor, h2: torch.Tensor) -> None:
        """Hot-swap concept vectors. h1 must be unit-normalized by the caller.

        Args:
            h1: (hidden_size,) — unit-norm "from" direction
            h2: (hidden_size,) — "to" direction
        """
        assert h1.shape == self.h1.shape, \
            f"h1 shape mismatch: expected {self.h1.shape}, got {h1.shape}"
        assert h2.shape == self.h2.shape, \
            f"h2 shape mismatch: expected {self.h2.shape}, got {h2.shape}"
        self.h1.copy_(h1)
        self.h2.copy_(h2)

    def load_vec_indices(self, vi: torch.Tensor) -> None:
        assert vi.shape == self.vec_indices.shape, \
            f"Shape mismatch: expected {self.vec_indices.shape}, got {vi.shape}"
        self.vec_indices.copy_(vi)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        concept_replace_kernel[(self.max_tokens,)](
            x, self.h1, self.h2,
            self.token_indices, self.vec_indices, self.n_valid_buf,
            self.hidden_size, BLOCK_H=self._block_h,
        )
        return x

    def run(self, x: torch.Tensor) -> torch.Tensor:
        return self(x)


# ─────────────────────────────────────────────────────────────────────────────
# Linear
# ─────────────────────────────────────────────────────────────────────────────

@triton.jit
def linear_select_write_kernel(
    x_ptr,            # (n_rows, hidden_size) — modified in-place
    out_ptr,          # (n_rows, hidden_size) — pre-computed W @ h + b (float32)
    indices_ptr,      # (max_tokens,) int32
    vec_indices_ptr,  # (n_rows,) int32 — 0 = skip, non-zero = write
    n_valid_ptr,      # scalar int32
    hidden_size,
    BLOCK_H: tl.constexpr,
):
    """
    For each selected token where vec_indices[row] != 0:
        x[row, :] = out_buf[row, :]    # scatter GEMM result into x
    Grid: (max_tokens,)
    """
    sel_idx = tl.program_id(0)
    n_valid = tl.load(n_valid_ptr)
    if sel_idx >= n_valid:
        return

    row  = tl.load(indices_ptr     + sel_idx)
    flag = tl.load(vec_indices_ptr + row)
    if flag == 0:
        return

    offs  = tl.arange(0, BLOCK_H)
    mask  = offs < hidden_size
    x_raw = tl.load(x_ptr   + row * hidden_size + offs, mask=mask, other=0.0)
    val   = tl.load(out_ptr  + row * hidden_size + offs, mask=mask, other=0.0)
    tl.store(x_ptr + row * hidden_size + offs, val.to(x_raw.dtype), mask=mask)


class SteeringLinear(torch.nn.Module):
    """
    EasySteer Linear: replace selected token activations with W @ h + b.

    For each selected token where vec_indices[row] != 0:
        x[row, :] = W @ x[row, :] + b

    cuBLAS computes the full GEMM for all tokens into a pre-allocated buffer;
    a Triton kernel scatters the result back only for selected/gated rows.

    Usage
    ─────
    steerer = SteeringLinear(W_tensor, b_tensor, vi, max_tokens=2048)
    steerer.load_weights(new_W, new_b)   # hot-swap weights
    steerer.load_vec_indices(new_vi)     # hot-swap gate flags
    steerer(x)
    """

    def __init__(
        self,
        W:           torch.Tensor,  # (hidden_size, hidden_size) float32
        b:           torch.Tensor,  # (hidden_size,) float32
        vec_indices: torch.Tensor,  # (max_tokens,) int32
        max_tokens:  int,
    ):
        super().__init__()
        assert W.ndim == 2 and W.shape[0] == W.shape[1], \
            f"W must be square (hidden_size, hidden_size), got {W.shape}"
        assert b.shape == (W.shape[0],), \
            f"b shape {b.shape} must match hidden_size {W.shape[0]}"
        assert vec_indices.ndim == 1 and vec_indices.shape[0] == max_tokens, \
            f"vec_indices must be ({max_tokens},), got {vec_indices.shape}"
        assert vec_indices.dtype == torch.int32, "vec_indices must be int32"

        hidden_size = W.shape[0]
        device      = W.device

        self.hidden_size = hidden_size
        self.max_tokens  = max_tokens
        self._block_h    = triton.next_power_of_2(hidden_size)

        self.register_parameter('W',
            torch.nn.Parameter(W, requires_grad=False))
        self.register_parameter('b',
            torch.nn.Parameter(b, requires_grad=False))

        # Pre-allocated GEMM output buffer (float32 — cuBLAS output)
        self.register_parameter('output_buf',
            torch.nn.Parameter(
                torch.zeros(max_tokens, hidden_size, dtype=torch.float32, device=device),
                requires_grad=False))

        self.register_parameter('token_indices',
            torch.nn.Parameter(
                torch.zeros(max_tokens, dtype=torch.int32, device=device),
                requires_grad=False))
        self.register_parameter('n_valid_buf',
            torch.nn.Parameter(
                torch.zeros(1, dtype=torch.int32, device=device),
                requires_grad=False))
        self.register_parameter('vec_indices',
            torch.nn.Parameter(vec_indices, requires_grad=False))

    def load_weights(self, W: torch.Tensor, b: torch.Tensor) -> None:
        """Copy linear transformation weights and bias.

        Args:
            W: (hidden_size, hidden_size) float32
            b: (hidden_size,) float32
        """
        assert W.shape == self.W.shape, \
            f"W shape mismatch: expected {self.W.shape}, got {W.shape}"
        assert b.shape == self.b.shape, \
            f"b shape mismatch: expected {self.b.shape}, got {b.shape}"
        self.W.copy_(W)
        self.b.copy_(b)

    def load_vec_indices(self, vi: torch.Tensor) -> None:
        assert vi.shape == self.vec_indices.shape, \
            f"Shape mismatch: expected {self.vec_indices.shape}, got {vi.shape}"
        self.vec_indices.copy_(vi)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n_tok = x.shape[0]
        out   = self.output_buf[:n_tok]

        # cuBLAS: out = x @ W.T + b  (no allocation when x is already float32)
        torch.mm(x.float(), self.W.T, out=out)
        out += self.b

        # Triton scatter: out[row] → x[row] for selected+gated tokens only
        linear_select_write_kernel[(self.max_tokens,)](
            x, out, self.token_indices, self.vec_indices, self.n_valid_buf,
            self.hidden_size, BLOCK_H=self._block_h,
        )
        return x

    def run(self, x: torch.Tensor) -> torch.Tensor:
        return self(x)


# ─────────────────────────────────────────────────────────────────────────────
# LM-Steer
# ─────────────────────────────────────────────────────────────────────────────

@triton.jit
def lm_steer_residual_kernel(
    x_ptr,            # (n_rows, hidden_size) — modified in-place
    delta_ptr,        # (n_rows, hidden_size) — pre-computed (h @ P1) @ P2^T (float32)
    indices_ptr,      # (max_tokens,) int32
    vec_indices_ptr,  # (n_rows,) int32 — 0 = skip, non-zero = steer
    n_valid_ptr,      # scalar int32
    alpha_ptr,        # scalar float32
    hidden_size,
    BLOCK_H: tl.constexpr,
):
    """
    For each selected token where vec_indices[row] != 0:
        x[row, :] += alpha * delta[row, :]
    Grid: (max_tokens,)
    """
    sel_idx = tl.program_id(0)
    n_valid = tl.load(n_valid_ptr)
    if sel_idx >= n_valid:
        return

    row  = tl.load(indices_ptr     + sel_idx)
    flag = tl.load(vec_indices_ptr + row)
    if flag == 0:
        return

    alpha = tl.load(alpha_ptr).to(tl.float32)

    offs  = tl.arange(0, BLOCK_H)
    mask  = offs < hidden_size
    x_raw = tl.load(x_ptr     + row * hidden_size + offs, mask=mask, other=0.0)
    delta = tl.load(delta_ptr  + row * hidden_size + offs, mask=mask, other=0.0).to(tl.float32)
    x     = x_raw.to(tl.float32)
    tl.store(x_ptr + row * hidden_size + offs, (x + alpha * delta).to(x_raw.dtype), mask=mask)


class SteeringLMSteer(torch.nn.Module):
    """
    EasySteer LM-Steer: low-rank residual steering via two projection matrices.

    For each selected token where vec_indices[row] != 0:
        x[row, :] += alpha * ((x[row, :] @ P1) @ P2^T)

    P1: (hidden_size, rank)   P2: (hidden_size, rank)

    Two cuBLAS GEMMs compute the low-rank delta for all tokens; a Triton kernel
    adds alpha * delta only to selected/gated rows.

    Usage
    ─────
    steerer = SteeringLMSteer(P1_tensor, P2_tensor, alpha=1.0, vi=vi, max_tokens=2048)
    steerer.load_projectors(new_P1, new_P2, alpha=2.0)   # hot-swap
    steerer.load_vec_indices(new_vi)                      # hot-swap gate flags
    steerer(x)
    """

    def __init__(
        self,
        P1:          torch.Tensor,  # (hidden_size, rank) float32
        P2:          torch.Tensor,  # (hidden_size, rank) float32
        alpha:       float,
        vec_indices: torch.Tensor,  # (max_tokens,) int32
        max_tokens:  int,
    ):
        super().__init__()
        assert P1.ndim == 2, f"P1 must be 2-D (hidden_size, rank), got {P1.shape}"
        assert P2.shape == P1.shape, \
            f"P2 shape {P2.shape} must match P1 shape {P1.shape}"
        assert vec_indices.ndim == 1 and vec_indices.shape[0] == max_tokens, \
            f"vec_indices must be ({max_tokens},), got {vec_indices.shape}"
        assert vec_indices.dtype == torch.int32, "vec_indices must be int32"

        hidden_size = P1.shape[0]
        rank_       = P1.shape[1]
        device      = P1.device

        self.hidden_size = hidden_size
        self.rank        = rank_
        self.max_tokens  = max_tokens
        self._block_h    = triton.next_power_of_2(hidden_size)

        self.register_parameter('P1',
            torch.nn.Parameter(P1, requires_grad=False))
        self.register_parameter('P2',
            torch.nn.Parameter(P2, requires_grad=False))
        self.register_parameter('alpha',
            torch.nn.Parameter(
                torch.tensor([alpha], dtype=torch.float32, device=device),
                requires_grad=False))

        # Intermediate buffers for the two-GEMM pipeline
        self.register_parameter('tmp_buf',
            torch.nn.Parameter(
                torch.zeros(max_tokens, rank_, dtype=torch.float32, device=device),
                requires_grad=False))
        self.register_parameter('delta_buf',
            torch.nn.Parameter(
                torch.zeros(max_tokens, hidden_size, dtype=torch.float32, device=device),
                requires_grad=False))

        self.register_parameter('token_indices',
            torch.nn.Parameter(
                torch.zeros(max_tokens, dtype=torch.int32, device=device),
                requires_grad=False))
        self.register_parameter('n_valid_buf',
            torch.nn.Parameter(
                torch.zeros(1, dtype=torch.int32, device=device),
                requires_grad=False))
        self.register_parameter('vec_indices',
            torch.nn.Parameter(vec_indices, requires_grad=False))

    def load_projectors(
        self,
        P1:    torch.Tensor,
        P2:    torch.Tensor,
        alpha: float = 1.0,
    ) -> None:
        """Copy low-rank projection matrices and scale factor.

        Args:
            P1:    (hidden_size, rank) float32
            P2:    (hidden_size, rank) float32
            alpha: scalar — scale applied to the low-rank residual
        """
        assert P1.shape == self.P1.shape, \
            f"P1 shape mismatch: expected {self.P1.shape}, got {P1.shape}"
        assert P2.shape == self.P2.shape, \
            f"P2 shape mismatch: expected {self.P2.shape}, got {P2.shape}"
        self.P1.copy_(P1)
        self.P2.copy_(P2)
        self.alpha.fill_(alpha)

    def load_vec_indices(self, vi: torch.Tensor) -> None:
        assert vi.shape == self.vec_indices.shape, \
            f"Shape mismatch: expected {self.vec_indices.shape}, got {vi.shape}"
        self.vec_indices.copy_(vi)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n_tok = x.shape[0]
        tmp   = self.tmp_buf[:n_tok]
        delta = self.delta_buf[:n_tok]
        x_f   = x.float()

        # Step 1: tmp   = x @ P1     (n_tok, rank)
        torch.mm(x_f, self.P1, out=tmp)
        # Step 2: delta = tmp @ P2^T (n_tok, hidden_size)
        torch.mm(tmp, self.P2.T, out=delta)

        # Step 3: Triton adds alpha * delta to selected+gated rows only
        lm_steer_residual_kernel[(self.max_tokens,)](
            x, delta, self.token_indices, self.vec_indices, self.n_valid_buf,
            self.alpha,
            self.hidden_size, BLOCK_H=self._block_h,
        )
        return x

    def run(self, x: torch.Tensor) -> torch.Tensor:
        return self(x)
