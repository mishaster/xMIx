# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch


# ──────────────────────────────────────────────────────────────────────────────
# MLPProbe class
# ──────────────────────────────────────────────────────────────────────────────

class MLPProbe(torch.nn.Module):
    """
    Two-layer MLP probe using cuBLAS GEMMs (torch.mm) for both layers.

        fc1: Linear(input_dim  → hidden_dim) + ReLU
        fc2: Linear(hidden_dim → output_dim) + Sigmoid

    Shapes and dtype are inferred from the weight tensors passed at construction.
    All buffers are pre-allocated and never reallocated — CUDA graph safe.

    Usage
    ─────
    mlp = MLPProbe(w1, b1, w2, b2, max_tokens=2048)

    # hot-swap weights without reconstructing (CUDA-graph-safe):
    mlp.load_weights(new_w1, new_b1, new_w2, new_b2)

    out = mlp.run(hidden_states)   # (n_tokens, output_dim)
    """

    def __init__(
        self,
        w1:         torch.Tensor,   # (hidden_dim, input_dim)
        b1:         torch.Tensor,   # (hidden_dim,)
        w2:         torch.Tensor,   # (output_dim, hidden_dim)
        b2:         torch.Tensor,   # (output_dim,)
        max_tokens: int,
    ):
        super().__init__()

        assert w1.ndim == 2,                          "w1 must be 2-D (hidden_dim, input_dim)"
        assert b1.shape == (w1.shape[0],),            f"b1 shape {b1.shape} does not match w1 rows {w1.shape[0]}"
        assert w2.ndim == 2,                          "w2 must be 2-D (output_dim, hidden_dim)"
        assert w2.shape[1] == w1.shape[0],            f"w2 cols {w2.shape[1]} must equal hidden_dim {w1.shape[0]}"
        assert b2.shape == (w2.shape[0],),            f"b2 shape {b2.shape} does not match w2 rows {w2.shape[0]}"
        assert w1.dtype == b1.dtype == w2.dtype == b2.dtype, "all weight tensors must share the same dtype"

        self.input_dim  = w1.shape[1]
        self.hidden_dim = w1.shape[0]
        self.output_dim = w2.shape[0]
        self.max_tokens = max_tokens

        dtype  = w1.dtype
        device = w1.device

        self.register_parameter('w1', torch.nn.Parameter(w1.to(device), requires_grad=False))
        self.register_parameter('b1', torch.nn.Parameter(b1.to(device), requires_grad=False))
        self.register_parameter('w2', torch.nn.Parameter(w2.to(device), requires_grad=False))
        self.register_parameter('b2', torch.nn.Parameter(b2.to(device), requires_grad=False))

        self.register_buffer('hidden_buf', torch.zeros(max_tokens, self.hidden_dim, dtype=dtype, device=device))
        self.register_buffer('output_buf', torch.zeros(max_tokens, self.output_dim, dtype=dtype, device=device))

    def load_weights(
        self,
        w1: torch.Tensor,
        b1: torch.Tensor,
        w2: torch.Tensor,
        b2: torch.Tensor,
    ) -> None:
        """Hot-swap weights. Addresses stay stable — safe to call inside a live CUDA graph context."""
        assert w1.shape == self.w1.shape, f"w1 shape mismatch: expected {self.w1.shape}, got {w1.shape}"
        assert b1.shape == self.b1.shape, f"b1 shape mismatch: expected {self.b1.shape}, got {b1.shape}"
        assert w2.shape == self.w2.shape, f"w2 shape mismatch: expected {self.w2.shape}, got {w2.shape}"
        assert b2.shape == self.b2.shape, f"b2 shape mismatch: expected {self.b2.shape}, got {b2.shape}"
        self.w1.copy_(w1)
        self.b1.copy_(b1)
        self.w2.copy_(w2)
        self.b2.copy_(b2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: shape (..., input_dim). Leading dims flattened into tokens.
        Returns:
            Probabilities, shape (n_tokens, output_dim).
        """
        n_tokens = x.numel() // self.input_dim
        x_2d     = x.view(n_tokens, self.input_dim)

        torch.mm(x_2d, self.w1.T, out=self.hidden_buf[:n_tokens])
        self.hidden_buf[:n_tokens].add_(self.b1)
        torch.relu_(self.hidden_buf[:n_tokens])

        torch.mm(self.hidden_buf[:n_tokens], self.w2.T, out=self.output_buf[:n_tokens])
        self.output_buf[:n_tokens].add_(self.b2)
        torch.sigmoid_(self.output_buf[:n_tokens])

        return self.output_buf[:n_tokens]

    def run(self, x: torch.Tensor) -> torch.Tensor:
        return self(x)


# ──────────────────────────────────────────────────────────────────────────────
# LinearProbe class
# ──────────────────────────────────────────────────────────────────────────────

class LinearProbe(torch.nn.Module):
    """
    Logistic regression probe: sigmoid( x @ weights.T + bias )

    Uses torch.mm (cuBLAS) for the GEMM. Shapes and dtype are inferred from
    the weight tensors passed at construction. All buffers pre-allocated for
    CUDA graph safety.

    Usage
    ─────
    probe = LinearProbe(weights, bias, max_tokens=2048)

    # hot-swap weights without reconstructing (CUDA-graph-safe):
    probe.load_weights(new_weights, new_bias)

    probs = probe.run(hidden_states)   # (n_tokens, n_classes)
    """

    def __init__(
        self,
        weights:    torch.Tensor,   # (n_classes, hidden_size)
        bias:       torch.Tensor,   # (n_classes,)
        max_tokens: int,
    ):
        super().__init__()

        if weights.ndim == 1:
            weights = weights.unsqueeze(0)   # (hidden_size,) → (1, hidden_size)
        if bias.ndim == 0:
            bias = bias.unsqueeze(0)

        assert weights.ndim == 2,                        "weights must be 2-D (n_classes, hidden_size)"
        assert bias.shape == (weights.shape[0],),        f"bias shape {bias.shape} must match n_classes {weights.shape[0]}"
        assert weights.dtype == bias.dtype,              "weights and bias must share the same dtype"

        self.hidden_size = weights.shape[1]
        self.n_classes   = weights.shape[0]
        self.max_tokens  = max_tokens

        dtype  = weights.dtype
        device = weights.device

        self.register_parameter('weights', torch.nn.Parameter(weights.to(device), requires_grad=False))
        self.register_parameter('bias',    torch.nn.Parameter(bias.to(device),    requires_grad=False))
        self.register_buffer('output', torch.zeros(max_tokens, self.n_classes, dtype=dtype, device=device))

    def load_weights(self, weights: torch.Tensor, bias: torch.Tensor) -> None:
        """Hot-swap weights. Addresses stay stable — safe to call inside a live CUDA graph context."""
        if weights.ndim == 1:
            weights = weights.unsqueeze(0)
        if bias.ndim == 0:
            bias = bias.unsqueeze(0)
        assert weights.shape == self.weights.shape, f"weights shape mismatch: expected {self.weights.shape}, got {weights.shape}"
        assert bias.shape == self.bias.shape,       f"bias shape mismatch: expected {self.bias.shape}, got {bias.shape}"
        self.weights.copy_(weights)
        self.bias.copy_(bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: shape (..., hidden_size). Leading dims flattened into tokens.
        Returns:
            Probabilities, shape (n_tokens, n_classes).
        """
        n_tokens = x.numel() // self.hidden_size
        x_2d     = x.view(n_tokens, self.hidden_size)

        torch.mm(x_2d, self.weights.T, out=self.output[:n_tokens])
        self.output[:n_tokens].add_(self.bias)
        torch.sigmoid_(self.output[:n_tokens])

        return self.output[:n_tokens]

    def run(self, x: torch.Tensor) -> torch.Tensor:
        return self(x)
