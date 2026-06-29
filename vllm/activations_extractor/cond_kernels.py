
import torch
from vllm.triton_utils import tl, triton


# ---------------------------------------------------------------------------
# Conditional projection cosine-similarity kernel
#
# The projection step is done by cuBLAS in the calling code before invoking
# the kernel.  The kernel receives the already-projected tensor y and computes
# per-token cosine similarity:
#
#   projected[i] = tanh(condition_projector @ x[i])  — cuBLAS, done outside
#   result[i]    = dot(x[i], projected[i]) / (norm(x[i]) * norm(projected[i]))
# ---------------------------------------------------------------------------

@triton.jit
def _cond_proj_cos_sim_kernel(
    x_ptr,
    y_ptr,       # pre-computed: tanh(x @ cp.T), shape (n_tokens, hidden_size)
    out_ptr,
    hidden_size: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    token_idx = tl.program_id(0)

    offs = tl.arange(0, BLOCK_H)
    mask = offs < hidden_size

    x = tl.load(x_ptr + token_idx * hidden_size + offs, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(y_ptr + token_idx * hidden_size + offs, mask=mask, other=0.0).to(tl.float32)

    dot_xy = tl.sum(x * y, axis=0)
    sq_x   = tl.sum(x * x, axis=0)
    sq_y   = tl.sum(y * y, axis=0)

    cos_sim = dot_xy / (tl.sqrt(sq_x) * tl.sqrt(sq_y) + 1e-8)
    tl.store(out_ptr + token_idx, cos_sim)


@triton.jit
def _cond_proj_cos_sim_threshold_kernel(
    x_ptr,
    y_ptr,            # pre-computed: tanh(x @ cp.T), shape (n_tokens, hidden_size)
    out_ptr,          # (max_tokens,) int32 — writes 1 if cos_sim > threshold, else 0
    threshold_ptr,    # scalar float32
    hidden_size: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    token_idx = tl.program_id(0)

    offs = tl.arange(0, BLOCK_H)
    mask = offs < hidden_size

    x = tl.load(x_ptr + token_idx * hidden_size + offs, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(y_ptr + token_idx * hidden_size + offs, mask=mask, other=0.0).to(tl.float32)

    dot_xy = tl.sum(x * y, axis=0)
    sq_x   = tl.sum(x * x, axis=0)
    sq_y   = tl.sum(y * y, axis=0)

    cos_sim   = dot_xy / (tl.sqrt(sq_x) * tl.sqrt(sq_y) + 1e-8)
    threshold = tl.load(threshold_ptr)
    tl.store(out_ptr + token_idx, (cos_sim > threshold).to(tl.int32))


class CondProjCosSim(torch.nn.Module):
    """
    Per-token conditional projection cosine similarity.

    For each token i:
        projected[i] = tanh(condition_projector @ x[i])   — cuBLAS GEMM
        out[i]       = dot(x[i], projected[i]) / (norm(x[i]) * norm(projected[i]))

    condition_projector is a (hidden_size, hidden_size) learned linear transformation.
    The cosine similarity measures how much each hidden state aligns with its own
    projection through that matrix.

    All buffers are registered parameters — Dynamo treats them as module state
    so no shape guards are emitted.

    Supports fan-out to N pre-configured sink buffers per output. The kernel
    runs once per forward pass, writing into the first (primary) sink; the
    result is then copied to each remaining sink. All sinks must be registered
    parameters/buffers of their owner module so addresses are stable for CUDA
    graph capture.

    Usage
    ─────
    scorer = CondProjCosSim(cp_tensor, threshold=0.5, max_tokens=2048)
    scorer.forward_backup(x)  # populates every sink in _sinks_out      (float32)
    scorer(x)                 # populates every sink in _sinks_out_bool (int32)

    # Optional fan-out — wire into downstream modules' buffers.
    scorer.set_output_buffers(out_bool=[adder.input_map, projector.input_map])
    """

    def __init__(
        self,
        condition_projector: torch.Tensor,       # (hidden_size, hidden_size) float32
        threshold:           float,
        max_tokens:          int,
    ):
        super().__init__()
        hidden_size = condition_projector.shape[0]
        device      = condition_projector.device
        self.hidden_size = hidden_size
        self.max_tokens  = max_tokens
        self._block_h    = triton.next_power_of_2(hidden_size)

        # (hidden_size, hidden_size) projection matrix
        self.register_parameter('condition_projector',
            torch.nn.Parameter(condition_projector.to(device), requires_grad=False))

        # Intermediate buffer for tanh(x @ cp.T) — written in-place by cuBLAS
        self.register_parameter('projected',
            torch.nn.Parameter(
                torch.zeros(max_tokens, hidden_size, dtype=torch.float32, device=device),
                requires_grad=False))

        # Owned default sinks — used until set_output_buffers replaces a list.
        self.register_parameter('out',
            torch.nn.Parameter(
                torch.zeros(max_tokens, dtype=torch.float32, device=device),
                requires_grad=False))

        self.register_parameter('out_bool',
            torch.nn.Parameter(
                torch.zeros(max_tokens, dtype=torch.int32, device=device),
                requires_grad=False))

        # Threshold scalar
        self.register_parameter('threshold',
            torch.nn.Parameter(
                torch.tensor([threshold], dtype=torch.float32, device=device),
                requires_grad=False))

        # Sink lists — defaults point at the owned buffers. Replaced by
        # set_output_buffers(). Not registered because external sinks are
        # owned and tracked by their respective modules.
        self._sinks_out:      list[torch.Tensor] = [self.out]
        self._sinks_out_bool: list[torch.Tensor] = [self.out_bool]

    def load_projector(self, cp: torch.Tensor) -> None:
        """Copy condition_projector into the pre-allocated buffer.

        Args:
            cp: (hidden_size, hidden_size) float32
        """
        assert cp.shape == self.condition_projector.shape, \
            f"Shape mismatch: expected {self.condition_projector.shape}, got {cp.shape}"
        self.condition_projector.copy_(cp)

    def load_threshold(self, value: float) -> None:
        """Set the comparison threshold used by forward().

        Args:
            value: scalar float — cos_sim > value writes 1, else 0
        """
        self.threshold.fill_(value)

    def set_output_buffers(
        self,
        out:      list[torch.Tensor] | None = None,   # list of (max_tokens,) float32 sinks
        out_bool: list[torch.Tensor] | None = None,   # list of (max_tokens,) int32 sinks
    ) -> None:
        """
        Replace one or both sink lists. Each forward pass computes the result
        once into the first sink (primary) and copies it to every remaining
        sink — the kernel runs only once regardless of fan-out width. Each
        sink must be a registered parameter/buffer of its owner module so its
        address is stable for CUDA graph capture.

        Only the kwarg(s) passed in are replaced; the other sink list is left
        untouched.

        Call ONCE, before the first forward pass / before graph capture.

        Example — wire out_bool into two SteeringVectorAdder input_maps:
            scorer.set_output_buffers(out_bool=[adder.input_map, projector.input_map])
        """
        if out is not None:
            for b in out:
                assert b.shape == (self.max_tokens,), \
                    f"out sink shape {tuple(b.shape)} != ({self.max_tokens},)"
                assert b.dtype == torch.float32, \
                    f"out sink dtype {b.dtype} != float32"
            self._sinks_out = list(out)

        if out_bool is not None:
            for b in out_bool:
                assert b.shape == (self.max_tokens,), \
                    f"out_bool sink shape {tuple(b.shape)} != ({self.max_tokens},)"
                assert b.dtype == torch.int32, \
                    f"out_bool sink dtype {b.dtype} != int32"
            self._sinks_out_bool = list(out_bool)

    def _project(self, x: torch.Tensor) -> torch.Tensor:
        """cuBLAS GEMM + in-place tanh, written into self.projected. No allocation."""
        n_tokens = x.shape[0]
        proj = self.projected[:n_tokens]
        torch.mm(x.float(), self.condition_projector.T, out=proj)
        proj.tanh_()
        return proj

    def forward_backup(self, x: torch.Tensor) -> None:
        """Compute per-token cosine similarity scores (float32) once and fan
        out to every sink in self._sinks_out by copying from the primary sink.

        Args:
            x: (n_tokens, hidden_size) — read-only

        Returns nothing — read sinks directly.
        """
        n_tokens = x.shape[0]
        proj     = self._project(x)

        primary = self._sinks_out[0]
        _cond_proj_cos_sim_kernel[(n_tokens,)](
            x, proj, primary,
            hidden_size=self.hidden_size,
            BLOCK_H=self._block_h,
        )
        for buf in self._sinks_out[1:]:
            buf.copy_(primary)

    def forward(self, x: torch.Tensor) -> None:
        """Compute per-token threshold flags (int32) once and fan out to every
        sink in self._sinks_out_bool by copying from the primary sink.

        Args:
            x: (n_tokens, hidden_size) — read-only

        Returns nothing — read sinks directly.
        """
        n_tokens = x.shape[0]
        proj     = self._project(x)

        primary = self._sinks_out_bool[0]
        _cond_proj_cos_sim_threshold_kernel[(n_tokens,)](
            x, proj, primary, self.threshold,
            hidden_size=self.hidden_size,
            BLOCK_H=self._block_h,
        )
        for buf in self._sinks_out_bool[1:]:
            buf.copy_(primary)

    def run(self, x: torch.Tensor) -> None:
        return self(x)


@triton.jit
def set_conditional_kernel(n_valid_ptr, handle_ptr):
    n = tl.load(n_valid_ptr)           # read count from GPU (no CPU sync)
    val = (n > 0).to(tl.int32)         # 1 → run body, 0 → skip
    tl.store(handle_ptr, val)

@triton.jit
def set_cond_from_n_valid(n_valid_ptr, cond_ptr):
    val = tl.load(n_valid_ptr).to(tl.int32)
    result = (val > 0).to(tl.int32)
    tl.store(cond_ptr, result)
