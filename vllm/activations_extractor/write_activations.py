# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import torch
import torch.library
import logging

from packaging import version

from vllm.triton_utils import tl, triton


@torch.library.custom_op("mylib::load_sv", mutates_args=())
def load_sv(sv: torch.Tensor) -> torch.Tensor:
    """Load steering vector into a local tensor, opaque to inductor (prevents constant folding)."""
    return sv.clone()

@load_sv.register_fake
def _(sv: torch.Tensor) -> torch.Tensor:
    return sv.clone()

logger = logging.getLogger(__name__)

# Only print the following warnings when triton version < 3.2.0.
# The issue won't affect performance or accuracy.
if version.parse(triton.__version__) < version.parse("3.2.0"):
    logger.warning(
        "The following error message 'operation scheduled before its operands' "
        "can be ignored."
    )

# Michael Kernel
@triton.jit
def add_kernel(
      x_ptr,        # pointer to input matrix (n_rows, hidden_size), modified in-place
      y_ptr,        # pointer to a single row vector of size hidden_size
      n_rows,
      hidden_size,
      BLOCK_SIZE: tl.constexpr,
  ):
      row = tl.program_id(axis=0)        # which token
      col_block = tl.program_id(axis=1)  # which chunk of the hidden dim
      col_offsets = col_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
      mask = col_offsets < hidden_size

      x = tl.load(x_ptr + row * hidden_size + col_offsets, mask=mask)
      y = tl.load(y_ptr + col_offsets, mask=mask)

      tl.store(x_ptr + row * hidden_size + col_offsets, x + y, mask=mask)

def add_vector_to_activations(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
      hidden_size = x.shape[-1]              # last dim is always hidden size
      n_rows = x.numel() // hidden_size      # flatten all leading dims into rows
      BLOCK_SIZE = 1024                      # 4 blocks per row for hidden_size=4096

      #y = torch.full((hidden_size,), 0.1, device=x.device, dtype=x.dtype)

      x_2d = x.view(n_rows, hidden_size)    # view as 2D (same memory, no copy)
      grid = (n_rows, triton.cdiv(hidden_size, BLOCK_SIZE))  # (n_tokens, 4)
      add_kernel[grid](x_2d, y, n_rows, hidden_size, BLOCK_SIZE=BLOCK_SIZE)

      return x                               # original shape preserved


# kl-then-steer linear_comb steering: x[row, :] += coeff * r[:] for every row.
# Single-layer, single-vector, unconditional — no per-row gate, no token_pos.
@triton.jit
def add_scaled_vector_kernel(
      x_ptr,        # (n_rows, hidden_size) — modified in-place
      r_ptr,        # (hidden_size,) steering vector — same dtype as x
      coeff_ptr,    # (1,) GPU scalar — multiplier (same dtype as x)
      n_rows,
      hidden_size,
      BLOCK_SIZE: tl.constexpr,
):
      row         = tl.program_id(axis=0)
      col_block   = tl.program_id(axis=1)
      col_offsets = col_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
      mask        = col_offsets < hidden_size

      coeff = tl.load(coeff_ptr)  # scalar in x's dtype

      x = tl.load(x_ptr + row * hidden_size + col_offsets, mask=mask)
      r = tl.load(r_ptr + col_offsets,                     mask=mask, other=0.0)

      tl.store(x_ptr + row * hidden_size + col_offsets,
               x + coeff * r, mask=mask)


class SteeringVectorScaledAdder(torch.nn.Module):
    """
    nn.Module wrapper around add_scaled_vector_kernel.

    Single-layer, single-vector linear_comb steering operator:
        x[row, :] += coeff * r[:]   for every row in x
    Unconditional (no per-row gate), in-place, CUDA-graph safe.

    Buffers are registered parameters (stable addresses). r and coeff are
    initialised from the ctor args; load_vector / load_coeff hot-swap them
    in place without invalidating captured CUDA graphs.

    Usage
    -----
    adder = SteeringVectorScaledAdder(r, coeff, max_tokens=2048)
    adder(x)                     # in-place
    adder.load_coeff(new_c)      # optional hot-swap
    """

    _BLOCK_SIZE = 1024

    def __init__(
        self,
        r:           torch.Tensor,   # (hidden_size,) — defines dtype and device
        coeff,                       # python float, 0-d or 1-elem tensor
        max_tokens:  int,            # accepted for signature symmetry; unused
    ):
        super().__init__()
        assert r.ndim == 1, f"r must be 1-D (hidden_size,), got shape {tuple(r.shape)}"

        self.hidden_size = r.shape[0]
        self.max_tokens  = max_tokens

        self.register_parameter('r',
            torch.nn.Parameter(r.detach().clone(), requires_grad=False))

        if not torch.is_tensor(coeff):
            coeff_t = torch.tensor([float(coeff)], dtype=r.dtype, device=r.device)
        else:
            coeff_t = coeff.reshape(1).to(dtype=r.dtype, device=r.device).clone()
        self.register_parameter('coeff',
            torch.nn.Parameter(coeff_t, requires_grad=False))

    def load_vector(self, r: torch.Tensor) -> None:
        assert r.shape == self.r.shape, \
            f"Shape mismatch: expected {self.r.shape}, got {r.shape}"
        self.r.copy_(r)

    def load_coeff(self, c) -> None:
        if not torch.is_tensor(c):
            c = torch.tensor([float(c)], dtype=self.coeff.dtype,
                             device=self.coeff.device)
        else:
            c = c.reshape(1).to(dtype=self.coeff.dtype,
                                device=self.coeff.device)
        self.coeff.copy_(c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden_size = x.shape[-1]
        n_rows      = x.numel() // hidden_size
        x_2d        = x.view(n_rows, hidden_size)
        grid        = (n_rows, triton.cdiv(hidden_size, self._BLOCK_SIZE))
        add_scaled_vector_kernel[grid](
            x_2d, self.r, self.coeff,
            n_rows, hidden_size,
            BLOCK_SIZE=self._BLOCK_SIZE,
        )
        return x

    def run(self, x: torch.Tensor, r: torch.Tensor,
            steered_tokens_num: torch.Tensor) -> torch.Tensor:
        return self(x)


# Michael conditional kernel
@triton.jit
def conditional_add_kernel(
      x_ptr,           # pointer to input matrix (n_rows, hidden_size), modified in-place
      y_ptr,           # pointer to a single row vector of size hidden_size
      condition_ptr,   # pointer to a pre-allocated scalar bool GPU tensor
      n_rows,
      hidden_size,
      BLOCK_SIZE: tl.constexpr,
  ):
      row = tl.program_id(axis=0)        # which token
      col_block = tl.program_id(axis=1)  # which chunk of the hidden dim
      col_offsets = col_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
      mask = col_offsets < hidden_size

      # Read condition from GPU memory — no CPU/GPU sync, CUDA graph safe
      # float16 dtype matches x and y — avoids inductor mixed-dtype input processing bug
      # Value is 1.0 (add) or 0.0 (skip), written by condition_fn

      #Misha Debug check
      cond_scale = tl.load(condition_ptr)  # scalar float16: 1.0 or 0.0

      x = tl.load(x_ptr + row * hidden_size + col_offsets, mask=mask)
      y = tl.load(y_ptr + col_offsets, mask=mask)

      # Scalar cond_scale broadcasts across vector y:
      #   1.0 → add y,  0.0 → add nothing
      
      #Misha Debug check
      result = x + cond_scale * y
      #result = x + y
      tl.store(x_ptr + row * hidden_size + col_offsets, result, mask=mask)


#@torch.library.custom_op("vllm::conditional_add_vector_to_activations",mutates_args={"condition_buf"})
def conditional_add_vector_to_activations(
      x: torch.Tensor,
      y: torch.Tensor,
      condition_fn: callable,
      condition_buf: torch.Tensor,
) -> torch.Tensor:
      """
      Adds y to x in-place if condition_fn(x, condition_buf) evaluates to True.

      Args:
          x:             input activation tensor, any shape, last dim = hidden_size
          y:             vector of shape (hidden_size,) to add
          condition_fn:  callable(x, buf) that writes a scalar bool result into buf
          condition_buf: pre-allocated scalar float16 GPU tensor (torch.float16, device='cuda')
                         Must be float16 (same dtype as x and y) to avoid inductor
                         mixed-dtype input handling bugs. condition_fn writes 1.0 (True)
                         or 0.0 (False) into it.

      CUDA graph compatibility requirements:
          - condition_fn must use only GPU operations (no .item(), no CPU ops)
          - condition_fn must write its result into condition_buf IN-PLACE
          - condition_buf must be allocated ONCE before graph capture and reused

      Example:
          condition_buf = torch.zeros(1, dtype=torch.float16, device='cuda')
          def my_condition(x, buf):
              buf.copy_((x.abs().max() > 0.5).to(torch.float16))
          conditional_add_vector_to_activations(x, y, my_condition, condition_buf)
      """
      condition_fn(x, condition_buf)

      hidden_size = x.shape[-1]
      n_rows = x.numel() // hidden_size
      BLOCK_SIZE = 1024

      # Create a local tensor to avoid inductor constant-folding the caller's buffer.
      # Read-only tensors passed from outside (e.g. registered buffers) get folded
      # out of new_inputs at compile time, causing copy_misaligned_inputs to crash.
      # A locally-created tensor is an internal graph node and is never folded.
      #y_local = torch.full((hidden_size,), 0.1, device=x.device, dtype=x.dtype)
      # load_sv is a custom op opaque to inductor — prevents constant folding of y.
      #y_local = load_sv(y)
      # y (sv_buf) is written to inside forward() before this call → mutable → not constant-folded
      #y_local = y.view(-1)
      #y_local = y
      x_2d = x.view(n_rows, hidden_size) # Flattens matrix to vector
      grid = (n_rows, triton.cdiv(hidden_size, BLOCK_SIZE))
      conditional_add_kernel[grid](
            x_2d, y, condition_buf, n_rows, hidden_size, BLOCK_SIZE=BLOCK_SIZE
            #x_2d, y_local, condition_buf, n_rows, hidden_size, BLOCK_SIZE=BLOCK_SIZE
      )

      return x

def at_least_half_positive(x: torch.Tensor, buf: torch.Tensor) -> None:
      """
      Writes True into buf if at least half of all entries in x are positive.

      All operations are on the GPU — safe to use inside a CUDA graph.
      buf must be a pre-allocated float16 GPU tensor (shape [1], dtype=torch.float16).
      Writes 1.0 (condition met) or 0.0 (condition not met).

      Usage:
          condition_buf = torch.zeros(1, dtype=torch.float16, device='cuda')
          conditional_add_vector_to_activations(x, y, at_least_half_positive, condition_buf)
      """
      total    = x.numel()          # Python int — resolved at trace/capture time, not a GPU op
      n_pos    = (x > 0).sum()      # GPU reduction: count positive elements
      buf.copy_((n_pos >= total // 2).to(torch.float16)) # write 1.0 or 0.0


# Michael refusal direction ablation kernels

@triton.jit
def dot_product_kernel(
      x_ptr,        # (n_rows, hidden_size) activation matrix
      r_ptr,        # (hidden_size,) normalized refusal direction
      dot_ptr,      # (n_rows,) output buffer — accumulates partial dot products (float32)
      n_rows,
      hidden_size,
      vec_indices,
      BLOCK_SIZE: tl.constexpr,
):
      row = tl.program_id(axis=0)

      flag_run = tl.load(vec_indices + row)
      if flag_run == 0:
          return      

      col_block = tl.program_id(axis=1)
      col_offsets = col_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
      #mask = col_offsets < hidden_size
      mask = (col_offsets < hidden_size) & (flag_run != 0)

      x = tl.load(x_ptr + row * hidden_size + col_offsets, mask=mask, other=0.0)
      r = tl.load(r_ptr + col_offsets, mask=mask, other=0.0)

      # Accumulate in float32 to avoid float16 precision loss across 4096 elements
      partial = tl.sum(x.to(tl.float32) * r.to(tl.float32), axis=0)
      tl.atomic_add(dot_ptr + row, partial)


@triton.jit
def subtract_projection_kernel(
      x_ptr,        # (n_rows, hidden_size) activation matrix — modified in-place
      r_ptr,        # (hidden_size,) normalized refusal direction
      dot_ptr,      # (n_rows,) precomputed dot products (float32)
      n_rows,
      hidden_size,
      vec_indices,
      BLOCK_SIZE: tl.constexpr,
):
      row = tl.program_id(axis=0)

      flag_run = tl.load(vec_indices + row)
      if flag_run == 0:
          return      
      
      col_block = tl.program_id(axis=1)
      col_offsets = col_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
      #mask = col_offsets < hidden_size
      mask = (col_offsets < hidden_size) & (flag_run != 0)

      dot = tl.load(dot_ptr + row)  # scalar float32
      x = tl.load(x_ptr + row * hidden_size + col_offsets, mask=mask)
      r = tl.load(r_ptr + col_offsets, mask=mask, other=0.0)

      # x_i -= dot_i * r  (cast dot back to x dtype for the subtraction)
      result = x - dot.to(x.dtype) * r
      tl.store(x_ptr + row * hidden_size + col_offsets, result, mask=mask)



def subtract_refusal_projection(x: torch.Tensor, r: torch.Tensor, vec_indices: torch.Tensor) -> torch.Tensor:
      """
      Removes the component of x along the refusal direction r, in-place.

      Equivalent to:
          r = r / r.norm()
          projection = x @ r          # (n_rows,)
          x -= torch.outer(projection, r)

      Args:
          x: activation tensor, any shape, last dim = hidden_size
          r: refusal direction vector, shape (hidden_size,)

      Returns:
          x modified in-place with refusal component removed
      """
      hidden_size = x.shape[-1]
      n_rows = x.numel() // hidden_size
      BLOCK_SIZE = 1024

      r_norm = r / r.norm()  # normalize on GPU, creates new tensor

      x_2d = x.view(n_rows, hidden_size)
      grid = (n_rows, triton.cdiv(hidden_size, BLOCK_SIZE))

      # float32 accumulation buffer — one scalar per row, zeroed before atomic adds
      dot_buf = torch.zeros(n_rows, device=x.device, dtype=torch.float32)

      dot_product_kernel[grid](x_2d, r_norm, dot_buf, n_rows, hidden_size, vec_indices, BLOCK_SIZE=BLOCK_SIZE)
      subtract_projection_kernel[grid](x_2d, r_norm, dot_buf, n_rows, hidden_size, vec_indices, BLOCK_SIZE=BLOCK_SIZE)

      return x


@triton.jit
def add_scaled_summed_vectors_kernel(
      x_ptr,                # (n_rows, hidden_size) — modified in-place
      r_ptr,                # (max_vecs, hidden_size) steering vector library
      scales_ptr,           # (max_tokens, MAX_VECS) float32 — per-row per-vector scale
      mask_ptr,             # (max_tokens,) int32 — 1 = steer this row, 0 = skip
      vec_indices_ptr,      # (max_tokens, MAX_VECS) int32 — which vectors per row
      n_valid_ptr,          # scalar int32 — number of real rows in x for this pass
      n_vecs_per_token_ptr, # (max_tokens,) int32
      hidden_size,
      MAX_VECS: tl.constexpr,
      BLOCK_V: tl.constexpr,  # power of 2 >= MAX_VECS
      BLOCK_C: tl.constexpr,
):
      """
      Per-row apply: for each row in x where mask[row] != 0 and row < n_valid,
              x[row, c] += Σ_v scales[row, v] * r[vec_indices[row, v], c]
      sels/vec_indices/n_vecs_per_token are all keyed by the row index directly.
      """
      row       = tl.program_id(axis=0)
      col_block = tl.program_id(axis=1)

      n_valid = tl.load(n_valid_ptr)
      if row >= n_valid:
            return

      if tl.load(mask_ptr + row) == 0:
            return

      n_vecs = tl.load(n_vecs_per_token_ptr + row)

      v_idx = tl.arange(0, BLOCK_V)
      c_idx = col_block * BLOCK_C + tl.arange(0, BLOCK_C)

      v_mask  = v_idx < n_vecs
      c_mask  = c_idx < hidden_size
      mask_2d = v_mask[:, None] & c_mask[None, :]

      # Load per-vector scales and indices for this row
      scales   = tl.load(scales_ptr      + row * MAX_VECS + v_idx, mask=v_mask, other=0.0)  # (BLOCK_V,)
      vec_idxs = tl.load(vec_indices_ptr + row * MAX_VECS + v_idx, mask=v_mask, other=0)    # (BLOCK_V,)

      # 2D gather: r[vec_idxs[v], c_idx[c]] for all (v, c) simultaneously
      r = tl.load(
            r_ptr + vec_idxs[:, None] * hidden_size + c_idx[None, :],
            mask=mask_2d, other=0.0,
      )   # (BLOCK_V, BLOCK_C)

      # Scale each vector row, then tree-reduce over vectors → (BLOCK_C,)
      combined = tl.sum(scales[:, None].to(tl.float32) * r.to(tl.float32), axis=0)

      # Add in fp32 to avoid precision loss with large scale values
      x = tl.load(x_ptr + row * hidden_size + c_idx, mask=c_mask)
      tl.store(x_ptr + row * hidden_size + c_idx,
               (x.to(tl.float32) + combined).to(x.dtype), mask=c_mask)


def add_scaled_summed_vectors_per_token(
      x: torch.Tensor,
      r: torch.Tensor,                   # (max_vecs, hidden_size)
      scales: torch.Tensor,              # (max_tokens, max_vecs) float32 — per-token per-vector scale
      token_indices: torch.Tensor,       # (max_tokens,) int32
      vec_indices: torch.Tensor,         # (max_tokens, max_vecs) int32
      n_valid_buf: torch.Tensor,         # scalar int32
      n_vecs_per_token: torch.Tensor,    # (max_tokens,) int32
) -> torch.Tensor:
      """
      For each selected token, scales each assigned vector and adds the sum to x:

          combined[col] = Σ_v  scales[sel_idx, v] * r[vec_idxs[v], col]
          x[row, col]  += combined[col]

      Use negative scales to subtract instead.
      scales shape: (max_tokens, max_vecs) — same layout as vec_indices.
      """
      if token_indices is None:
            return x

      hidden_size = x.shape[-1]
      BLOCK_C     = 512
      max_tokens  = token_indices.shape[0]
      max_vecs    = r.shape[0]
      BLOCK_V     = triton.next_power_of_2(max_vecs)

      add_scaled_summed_vectors_kernel[
            (max_tokens, triton.cdiv(hidden_size, BLOCK_C))
      ](
            x, r, scales, token_indices, vec_indices, n_valid_buf, n_vecs_per_token,
            hidden_size, MAX_VECS=max_vecs, BLOCK_V=BLOCK_V, BLOCK_C=BLOCK_C,
      )

      return x


class SteeringVectorAdder(torch.nn.Module):
    """
    nn.Module wrapper around add_scaled_summed_vectors_kernel.

    All buffers are registered parameters — Dynamo treats them as module state
    so no shape guards are emitted. The per-row mask (`input_map`) is written
    by an upstream condition kernel (e.g. EOLTokenDetector) before forward()
    and read directly by the kernel.

    Usage
    ─────
    adder = SteeringVectorAdder(r, scales, vec_indices, n_vecs_per_token)

    # Wire EOLTokenDetector to populate the mask in-place each forward pass:
    detector.set_output_buffers([adder.input_map])

    # In each forward pass, caller passes the actual token count:
    adder(x, n_valid_buf)   # n_valid_buf: (1,) int32 GPU tensor (stable addr)
    """

    _BLOCK_C = 512

    def __init__(
        self,
        r:                torch.Tensor,   # (max_vecs, hidden_size)
        scales:           torch.Tensor,   # (max_tokens, max_vecs) float32
        vec_indices:      torch.Tensor,   # (max_tokens, max_vecs) int32
        n_vecs_per_token: torch.Tensor,   # (max_tokens,) int32
    ):
        super().__init__()

        assert r.ndim == 2,                "r must be 2-D (max_vecs, hidden_size)"
        assert scales.ndim == 2,           "scales must be 2-D (max_tokens, max_vecs)"
        assert vec_indices.ndim == 2,      "vec_indices must be 2-D (max_tokens, max_vecs)"
        assert n_vecs_per_token.ndim == 1, "n_vecs_per_token must be 1-D (max_tokens,)"

        max_vecs, hidden_size = r.shape
        max_tokens            = scales.shape[0]

        assert scales.shape           == (max_tokens, max_vecs), \
            f"scales shape {scales.shape} must be ({max_tokens}, {max_vecs})"
        assert vec_indices.shape      == (max_tokens, max_vecs), \
            f"vec_indices shape {vec_indices.shape} must be ({max_tokens}, {max_vecs})"
        assert n_vecs_per_token.shape == (max_tokens,), \
            f"n_vecs_per_token shape {n_vecs_per_token.shape} must be ({max_tokens},)"

        self.max_vecs    = max_vecs
        self.hidden_size = hidden_size
        self.max_tokens  = max_tokens
        self._block_v    = triton.next_power_of_2(max_vecs)

        device = r.device

        # Steering vector library + per-token assignments — registered as-is from caller.
        self.register_parameter('r',
            torch.nn.Parameter(r.to(device), requires_grad=False))
        self.register_parameter('scales',
            torch.nn.Parameter(scales.to(device), requires_grad=False))
        self.register_parameter('vec_indices',
            torch.nn.Parameter(vec_indices.to(device), requires_grad=False))
        self.register_parameter('n_vecs_per_token',
            torch.nn.Parameter(n_vecs_per_token.to(device), requires_grad=False))

        # Per-row steering mask — entry r == 1 means "steer row r of x".
        # Written by EOLTokenDetector (or any other condition kernel) before
        # forward(); read directly by add_scaled_summed_vectors_kernel.
        # CUDA-graph safe: registered, address-stable.
        self.register_parameter('input_map',
            torch.nn.Parameter(
                torch.zeros(max_tokens, dtype=torch.int32, device=device),
                requires_grad=False))

    def load_vectors(self, r: torch.Tensor) -> None:
        """Copy steering vectors into the pre-allocated buffer.
        Args:
            r: (max_vecs, hidden_size)
        """
        assert r.shape == self.r.shape, \
            f"Shape mismatch: expected {self.r.shape}, got {r.shape}"
        self.r.copy_(r)

    def load_scales(self, scales: torch.Tensor) -> None:
        """Copy per-token per-vector scales into the pre-allocated buffer.
        Args:
            scales: (max_tokens, max_vecs) float32
        """
        assert scales.shape == self.scales.shape, \
            f"Shape mismatch: expected {self.scales.shape}, got {scales.shape}"
        self.scales.copy_(scales)

    def load_vec_indices(
        self,
        vec_indices:      torch.Tensor,   # (max_tokens, max_vecs) int32
        n_vecs_per_token: torch.Tensor,   # (max_tokens,) int32
    ) -> None:
        """Copy vector index assignments into the pre-allocated buffers.
        Args:
            vec_indices:      (max_tokens, max_vecs) int32
            n_vecs_per_token: (max_tokens,) int32
        """
        assert vec_indices.shape == self.vec_indices.shape, \
            f"Shape mismatch: expected {self.vec_indices.shape}, got {vec_indices.shape}"
        assert n_vecs_per_token.shape == self.n_vecs_per_token.shape, \
            f"Shape mismatch: expected {self.n_vecs_per_token.shape}, got {n_vecs_per_token.shape}"
        self.vec_indices.copy_(vec_indices)
        self.n_vecs_per_token.copy_(n_vecs_per_token)

    def forward(self, x: torch.Tensor, n_valid_buf: torch.Tensor) -> torch.Tensor:
        """
        Apply steering to x. Each row r where input_map[r] != 0 and
        r < n_valid_buf[0] is updated in-place by adding the per-row
        scaled sum of steering vectors.

        Args:
            x:           (n_rows, hidden_size) activation tensor — modified in-place
            n_valid_buf: (1,) int32 GPU tensor — number of real rows in x for
                         this forward pass. Caller-owned and address-stable
                         for CUDA graph capture.
        """
        add_scaled_summed_vectors_kernel[
            (self.max_tokens, triton.cdiv(self.hidden_size, self._BLOCK_C))
        ](
            x, self.r, self.scales, self.input_map, self.vec_indices,
            n_valid_buf, self.n_vecs_per_token,
            self.hidden_size,
            MAX_VECS=self.max_vecs,
            BLOCK_V=self._block_v,
            BLOCK_C=self._BLOCK_C,
        )
        return x

    def run(self, x: torch.Tensor, r: torch.Tensor,steered_tokens_num:torch.Tensor) -> torch.Tensor:
        #return self(x,steered_tokens_num)
        return self(x,steered_tokens_num)

@triton.jit
def check_bigger_kernel(
    input_ptr,      # (n_tokens,) int32 — token ids to check
    output_ptr,     # (n_tokens,) int32 — output map: 1 where condition holds, 0 otherwise
    target_val,     # scalar int32 — value to compare against
    n_tokens,
    BLOCK: tl.constexpr,
):
    """
    For each token: output[i] = 1 if input[i] == target_val else 0.
    Grid: (ceil(n_tokens / BLOCK),)
    """
    idx  = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = idx < n_tokens
    val    = tl.load(input_ptr  + idx, mask=mask, other=0)
    result = (val > target_val).to(tl.int32)
    tl.store(output_ptr + idx, result, mask=mask)

@triton.jit
def check_smaller_kernel(
    input_ptr,      # (n_tokens,) int32 — token ids to check
    output_ptr,     # (n_tokens,) int32 — output map: 1 where condition holds, 0 otherwise
    target_val,     # scalar int32 — value to compare against
    n_tokens,
    BLOCK: tl.constexpr,
):
    """
    For each token: output[i] = 1 if input[i] == target_val else 0.
    Grid: (ceil(n_tokens / BLOCK),)
    """
    idx  = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = idx < n_tokens
    val    = tl.load(input_ptr  + idx, mask=mask, other=0)
    result = (val < target_val).to(tl.int32)
    tl.store(output_ptr + idx, result, mask=mask)


class ConditionEvaluator(torch.nn.Module):
    """
    Evaluates a per-token condition and writes the result (0 or 1) into an
    output buffer.  The output buffer can either be the class's own internal
    buffer or an externally-supplied buffer (e.g. SteeringVectorAdder.input_map)
    so that the result feeds directly into another kernel without an extra copy.

    Optionally stores a CUDA graph conditional-node handle; callers can read
    self.cond_handle to wire the result into a graph conditional node.

    Usage
    ─────
    evaluator = ConditionEvaluator(max_tokens=2048, device="cuda")

    # Option A — use internal buffer, read result via evaluator.output_buf
    evaluator(token_ids, target_val=198)   # 198 = '\n\n'

    # Option B — write directly into another module's buffer
    evaluator.set_output_buffer(steering_adder.input_map)
    evaluator(token_ids, target_val=198)

    # Store a CUDA graph conditional handle for graph-level gating
    evaluator.set_cond_handle(handle)
    """

    _BLOCK = 1024

    def __init__(
        self,
        max_tokens: int,
        case_of_comparison:str,
        device: str = "cuda",
    ):
        super().__init__()
        self.max_tokens = max_tokens

        # Own output buffer — 0/1 per token, int32
        self.register_parameter(
            'output_buf',
            torch.nn.Parameter(
                torch.zeros(max_tokens, dtype=torch.int32, device=device),
                requires_grad=False,
            )
        )

        # External output target: if set, forward() writes here instead of
        # output_buf.  Stored as a plain attribute — not registered — because
        # it is owned and tracked by the other module.
        self._ext_output: torch.Tensor | None = None

        # Optional CUDA graph conditional-node handle (opaque Python object)
        self.cond_handle = None
        assert case_of_comparison == "smaller" or case_of_comparison == "bigger", \
            f"case of comparison mismatch, expected value: bigger or smaller" 
        self.rule = case_of_comparison

    def set_output_buffer(self, buf: torch.Tensor) -> None:
        """
        Point this evaluator at an external buffer (e.g. SteeringVectorAdder.input_map).
        Call outside CUDA graph capture, before the first model forward.
        The buffer must already be a registered parameter/buffer of its owner
        module so its address is stable.
        """
        self._ext_output = buf

    def set_cond_handle(self, handle) -> None:
        """Store a CUDA graph conditional-node handle for graph-level gating."""
        self.cond_handle = handle

    def forward(
        self,
        token_ids:  torch.Tensor,   # (n_tokens,) int32
        target_val: int,            # token id to check for
    ) -> torch.Tensor:
        """
        Writes 1 into the output buffer at every position where
        token_ids[i] == target_val, 0 elsewhere.

        Returns the output buffer that was written (own or external).
        """
        out      = self._ext_output if self._ext_output is not None else self.output_buf
        n_tokens = token_ids.shape[0]
        grid     = (triton.cdiv(n_tokens, self._BLOCK),)
        if self.rule == "smaller":
            check_smaller_kernel[grid](
                token_ids, out, target_val, n_tokens, BLOCK=self._BLOCK,
            )
        if self.rule == "bigger":
            check_bigger_kernel[grid](
                token_ids, out, target_val, n_tokens, BLOCK=self._BLOCK,
            )
        return out

    def run(self, token_ids: torch.Tensor, target_val: int) -> torch.Tensor:
        return self(token_ids, target_val)


@triton.jit
def compact_mask_kernel(
    mask_ptr,    # (n,) bool — True where token matches
    out_ptr,     # (BLOCK,) int32 — output: indices of True entries
    count_ptr,   # scalar int32 — output: number of True entries
    n,
    BLOCK: tl.constexpr,  # power of 2 >= max possible n
):
    """
    GPU-side stream compaction: collects the indices of all True entries in
    mask into out_ptr and writes the count into count_ptr.

    Replaces torch.nonzero() — no CPU/GPU sync required.
    count_ptr is a GPU tensor read by downstream kernels via n_valid_ptr.

    Grid: (1,) — single program handles up to BLOCK tokens.
    """
    idx   = tl.arange(0, BLOCK)
    valid = idx < n
    flags = tl.load(mask_ptr + idx, mask=valid, other=0).to(tl.int1)

    # Prefix sum assigns each True entry a unique sequential write position.
    pos   = tl.cumsum(flags.to(tl.int32), axis=0) - 1
    total = tl.sum(flags.to(tl.int32), axis=0)

    tl.store(count_ptr, total)
    tl.store(out_ptr + pos, idx.to(tl.int32), mask=flags & valid)


def compact_mask_to_indices(
    mask:          torch.Tensor,   # (n,) bool
    out_indices:   torch.Tensor,   # (max_n,) int32 — pre-allocated
    count_buf:     torch.Tensor,   # scalar int32   — pre-allocated
) -> None:
    """
    Fills out_indices with the positions where mask is True and writes the
    count into count_buf, entirely on the GPU — no CPU/GPU sync.

    out_indices and count_buf must be pre-allocated once before CUDA graph
    capture and reused every call (stable addresses).

    Args:
        mask:        boolean tensor, shape (n,)
        out_indices: pre-allocated int32 output buffer, shape (max_n,)
        count_buf:   pre-allocated scalar int32 tensor
    """
    n     = mask.shape[0]
    BLOCK = triton.next_power_of_2(n)
    compact_mask_kernel[(1,)](mask, out_indices, count_buf, n, BLOCK=BLOCK)


@triton.jit
def eol_mask_kernel(
    input_ids_ptr,  # (n_tokens,) int32/int64 — token ids for current forward pass
    lookup_ptr,     # (vocab_size,) bool — True where token is EOL
    out_ptr,        # (max_tokens,) int32 — dense per-token mask, written every call
    n_tokens,       # int — number of real tokens in input_ids
    vocab_size,     # int — bound for the lookup table
    BLOCK: tl.constexpr,  # = next_power_of_2(max_tokens)
):
    """
    Per-token EOL mask. Grid (1,), no CPU/GPU sync, no atomics.

    Writes 1 at out[i] iff i < n_tokens and lookup[input_ids[i]] is True.
    All other positions in [0, BLOCK) are written 0, so padding slots
    [n_tokens, max_tokens) are wiped on every call (no stale data).
    """
    idx        = tl.arange(0, BLOCK)
    in_input   = idx < n_tokens

    token_ids  = tl.load(input_ids_ptr + idx, mask=in_input, other=0).to(tl.int32)
    in_range   = (token_ids >= 0) & (token_ids < vocab_size)

    flag       = tl.load(lookup_ptr + token_ids, mask=in_input & in_range, other=0).to(tl.int1)

    val        = tl.where(in_input & flag, 1, 0).to(tl.int32)
    tl.store(out_ptr + idx, val)


class EOLTokenDetector(torch.nn.Module):
    """
    Per-forward-pass EOL detector. Emits a dense (max_tokens,) int32 mask:
    mask[i] == 1 iff input_ids[i] is an EOL token, else 0. Padding slots
    [n_tokens, max_tokens) are always written 0.

    Supports fan-out to N pre-configured sink buffers (one kernel launch
    each). All buffers are registered or supplied at startup so addresses
    are stable for CUDA graph capture.

    Usage
    ─────
    detector = EOLTokenDetector(eol_token_ids, vocab_size=200000,
                                max_tokens=2048, device="cuda")

    # Optional fan-out — wire into downstream modules' mask buffers.
    # Call ONCE, before the first forward pass / before graph capture.
    detector.set_output_buffers([adder.input_map, projector.input_map])

    # Each forward pass:
    detector(input_ids)
    # → every configured sink now holds the current mask.
    """

    def __init__(
        self,
        eol_token_ids: list[int],
        vocab_size:    int = 200000,
        max_tokens:    int = 2048,
        device:        str = "cuda",
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_tokens = max_tokens
        self._block     = triton.next_power_of_2(max_tokens)

        lookup = torch.zeros(vocab_size, dtype=torch.bool, device=device)
        lookup[torch.tensor(eol_token_ids, dtype=torch.long, device=device)] = True
        self.register_parameter('lookup',
            torch.nn.Parameter(lookup, requires_grad=False))

        # Owned default sink — used until set_output_buffers replaces the list.
        self.register_parameter('mask',
            torch.nn.Parameter(
                torch.zeros(max_tokens, dtype=torch.int32, device=device),
                requires_grad=False))

        self._sinks: list[torch.Tensor] = [self.mask]

    def set_output_buffers(self, buffers: list[torch.Tensor]) -> None:
        """
        Replace the sink list. Each buffer must be (max_tokens,) int32 on
        the detector's device, and must be a registered parameter/buffer of
        its owning module so its address is stable for CUDA graph capture.

        Call ONCE, before the first forward pass / before graph capture.
        """
        for b in buffers:
            assert b.shape == (self.max_tokens,), \
                f"sink shape {tuple(b.shape)} != ({self.max_tokens},)"
            assert b.dtype == torch.int32, f"sink dtype {b.dtype} != int32"
        self._sinks = list(buffers)

    def forward(self, input_ids: torch.Tensor) -> None:
        """
        Populate every configured sink with the per-token EOL mask for the
        current forward pass. Returns nothing — read sinks directly.
        """
        n = input_ids.shape[0]
        for buf in self._sinks:
            eol_mask_kernel[(1,)](
                input_ids, self.lookup, buf,
                n, self.vocab_size,
                BLOCK=self._block,
            )
     
    def run(self, x: torch.Tensor) -> torch.Tensor:
        return self(x)

@triton.jit
def fused_subtract_projection_selected_kernel(
    x_ptr,            # (n_rows, hidden_size) — modified in-place
    r_ptr,            # (hidden_size,) normalized direction
    indices_ptr,      # (max_tokens,) int32 — selected row indices
    n_valid_ptr,      # scalar int32 — number of valid entries in indices_ptr
    vec_indices_ptr,  # (n_rows,) int32 — 0 = skip this row, non-zero = steer
    hidden_size,
    BLOCK_C: tl.constexpr,
):
    """
    For each selected token row where vec_indices[row] != 0:
        dot  = sum( x[row, :] * r[:] )      # float32 accumulation
        x[row, :] -= dot * r[:]             # in-place

    Grid: (max_tokens,) — programs with sel_idx >= n_valid exit immediately.
    Programs whose row has vec_indices[row] == 0 also exit immediately.
    One program owns the full hidden dimension: no atomics, no intermediate buffer.
    Two passes over hidden_size keep register pressure low.
    """
    sel_idx = tl.program_id(0)

    n_valid = tl.load(n_valid_ptr)
    #n_valid = n_valid_ptr
    if sel_idx >= n_valid:
        return

    row  = tl.load(indices_ptr + sel_idx)
    flag = tl.load(vec_indices_ptr + row)
    if flag == 0:
        return

    # ── Pass 1: accumulate dot product ────────────────────────────────────────
    dot = tl.zeros((1,), dtype=tl.float32)
    for c_start in range(0, hidden_size, BLOCK_C):
        c_idx  = c_start + tl.arange(0, BLOCK_C)
        c_mask = c_idx < hidden_size
        x = tl.load(x_ptr + row * hidden_size + c_idx, mask=c_mask, other=0.0)
        r = tl.load(r_ptr + c_idx,                     mask=c_mask, other=0.0)
        dot += tl.sum(x.to(tl.float32) * r.to(tl.float32), axis=0, keep_dims=True)

    # ── Pass 2: subtract projection ───────────────────────────────────────────
    for c_start in range(0, hidden_size, BLOCK_C):
        c_idx  = c_start + tl.arange(0, BLOCK_C)
        c_mask = c_idx < hidden_size
        x = tl.load(x_ptr + row * hidden_size + c_idx, mask=c_mask)
        r = tl.load(r_ptr + c_idx,                     mask=c_mask, other=0.0)
        tl.store(
            x_ptr + row * hidden_size + c_idx,
            (x.to(tl.float32) - dot * r.to(tl.float32)).to(x.dtype),
            mask=c_mask,
        )


def subtract_projection_selected(
    x:           torch.Tensor,   # (n_rows, hidden_size)
    r:           torch.Tensor,   # (hidden_size,) — must be pre-normalised
    indices:     torch.Tensor,   # (max_tokens,) int32
    n_valid_buf: torch.Tensor,   # scalar int32
    vec_indices: torch.Tensor,   # (n_rows,) int32 — 0 = skip, non-zero = steer
    BLOCK_C:     int = 512,
) -> torch.Tensor:
    """
    Removes the component along r for each token listed in indices[:n_valid_buf]
    where vec_indices[row] != 0.
    r must be normalised before calling. No intermediate buffer allocated.
    """
    hidden_size = x.shape[-1]
    max_tokens  = indices.shape[0]
    fused_subtract_projection_selected_kernel[(max_tokens,)](
        x, r, indices, n_valid_buf, vec_indices,
        hidden_size, BLOCK_C=BLOCK_C,
    )
    return x

class SteeringVectorDotSubtractNormalized(torch.nn.Module):
    """
    nn.Module wrapper around fused_subtract_projection_selected_kernel.

    All buffers are registered parameters — Dynamo treats them as module state
    so no shape guards are emitted.  Buffers are written by other kernels
    (e.g. EOLTokenDetector fills token_indices / n_valid_buf) and read here
    at forward time.

    Usage
    ─────
    adder = SteeringVectorAdder(max_vecs=16, hidden_size=4096, max_tokens=2048,
                                dtype=model.dtype, device="cuda")
    adder.load_vectors(r_tensor)          # (max_vecs, hidden_size)
    adder.load_scales(scales_tensor)      # (max_tokens, max_vecs) float32
    adder.load_vec_indices(vi, nvpt)      # (max_tokens, max_vecs) int32, (max_tokens,) int32

    # Inside forward — only x is passed; everything else comes from buffers:
    adder(x)
    """

    _BLOCK_C = 512

    def __init__(
        self,
        hidden_size: int,
        max_tokens:  int,
        steering_vector : torch.Tensor,
        vec_indices : torch.Tensor,
        dtype:       torch.dtype = torch.float16,
        device:      str         = "cuda",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_tokens  = max_tokens

        # Steering vector library
        self.register_parameter('r',
            torch.nn.Parameter(
                #steering_vector/steering_vector.norm(),
                steering_vector,
                requires_grad=False))
        #self.register_parameter('r',
        #    torch.nn.Parameter(
        #        torch.zeros((hidden_size,), dtype=dtype, device=device),
        #        requires_grad=False))

        # Which token rows in x to steer — written by EOLTokenDetector or any
        # other upstream kernel; n_valid_buf is read from GPU by the Triton
        # kernel (no CPU sync needed).
        self.register_parameter('token_indices',
            torch.nn.Parameter(
                torch.zeros(max_tokens, dtype=torch.int32, device=device),
                requires_grad=False))
        #self.register_parameter('n_valid_buf',
        #    torch.nn.Parameter(
        #        torch.zeros(1, dtype=torch.int32, device=device),
        #        requires_grad=False))
        self.register_parameter('n_valid_buf',
            torch.nn.Parameter(
                torch.tensor([1], dtype=torch.int32, device=device),
                requires_grad=False))


        # This buffer is equivalent to EBPF map and enables communication with the kernel.
        # Specifically, we intend it to be used to mark on what tokens to intervene.
        # So each entry correlates to the matching index.
        # Which kernel updates this - is defined by the user according to the condition.
        # For example: m.write(steer_vec).layer(17).cond(if token == \n\n)
        # Means we will use the steer_vec on layer 17 if the token is \n\n, which is checked
        # by another kernel.
        self.register_parameter('input_map',
            torch.nn.Parameter(
                torch.zeros(max_tokens, dtype=torch.int32, device=device),
                requires_grad=False))

        # Per-row steering gate: vec_indices[i] != 0 means row i in x is steered.
        # Indexed directly by the row index in x, not by sel_idx.
        self.register_parameter('vec_indices',
            torch.nn.Parameter(
                #torch.zeros(max_tokens, dtype=torch.int32, device=device),
                #torch.full((max_tokens,),1, dtype=torch.int32, device=device),
                vec_indices,
                requires_grad=False))

    #def load_vec_indices(self, vi: torch.Tensor) -> None:
        """Copy per-row steering flags into the pre-allocated buffer.

        Args:
            vi: (max_tokens,) int32 — 0 = skip that row, non-zero = steer it.
                Entry i corresponds to row i in the activation tensor x.
        """
    #   assert vi.shape == self.vec_indices.shape, \
    #        f"Shape mismatch: expected {self.vec_indices.shape}, got {vi.shape}"
    #    self.vec_indices.copy_(vi)

    def load_vector(self, r: torch.Tensor) -> None:
        """Normalise and copy the steering vector into the pre-allocated buffer.

        The kernel expects a unit-norm direction; normalisation is done here
        once at load time so every forward pass is allocation-free.

        Args:
            r: (1, hidden_size) — raw (unnormalised) steering direction
        """
        assert r.shape == self.r.shape, \
            f"Shape mismatch: expected {self.r.shape}, got {r.shape}"
        #self.r.copy_(r / r.norm())
        self.r.copy_(r)
    def load_n_valid_tokens(self,n_valid_buf_input:torch.Tensor ) -> None:
        
        assert self.n_valid_buf.shape == n_valid_buf_input.shape, \
            f"Shape mismatch: expected {self.n_valid_buf.shape}, got {n_valid_buf_input.shape}"
        self.n_valid_buf.copy_(n_valid_buf_input)


    #def forward(self, x: torch.Tensor,steered_tokens_num : torch.Tensor) -> torch.Tensor:
    def forward(self, x: torch.Tensor, r: torch.Tensor,steered_tokens_num:torch.Tensor) -> torch.Tensor:
        """
        Apply steering to x using all pre-loaded buffers.

        token_indices and n_valid_buf are typically written by an upstream
        kernel (e.g. EOLTokenDetector) before this call.

        Args:
            x: (n_rows, hidden_size) activation tensor — modified in-place
        """
        #hidden_size = x.shape[-1]
        #fused_subtract_projection_selected_kernel[(self.max_tokens,)](
        #    x, self.r, self.token_indices, self.n_valid_buf, self.vec_indices,
        #    #x, self.r, self.token_indices, steered_tokens_num, self.vec_indices,
        #    self.hidden_size, BLOCK_C=self._BLOCK_C,
        #)
        #return x
        return subtract_refusal_projection(x,self.r,self.vec_indices)        

    #def run(self, x: torch.Tensor,steered_tokens_num : torch.Tensor) -> torch.Tensor:
    def run(self, x: torch.Tensor, r: torch.Tensor,steered_tokens_num:torch.Tensor) -> torch.Tensor:
        #return self(x,steered_tokens_num)
        return self(x,r,steered_tokens_num)

# Michael addition end

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

        # Pre-allocated GEMM output buffer — dtype must match W/x so torch.mm(out=...) is valid.
        self.register_buffer('output_buf',
            torch.zeros(max_tokens, hidden_size, dtype=W.dtype, device=device))

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
        torch.mm(x, self.W.T, out=out)
        out += self.b

        # Triton scatter: out[row] → x[row] for selected+gated tokens only
        linear_select_write_kernel[(self.max_tokens,)](
            x, out, self.token_indices, self.vec_indices, self.n_valid_buf,
            self.hidden_size, BLOCK_H=self._block_h,
        )
        return x

    def run(self, x: torch.Tensor, r: torch.Tensor,steered_tokens_num:torch.Tensor) -> torch.Tensor:
        return self(x)

