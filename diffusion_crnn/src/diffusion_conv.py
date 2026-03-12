"""
diffusion_conv.py

Implements Diffusion Convolution — the core graph operation in DCRNN.

The key idea: replace every standard linear layer Wx inside a GRU with a
graph-aware operation that aggregates information from K-hop neighbourhoods.

For a graph signal X of shape (batch, N, features), diffusion convolution:
    1. Computes powers of the transition matrices: T_f^0, T_f^1, ..., T_f^K
                                                   T_b^0, T_b^1, ..., T_b^K
    2. Applies each power to X  (spreading signal K hops in each direction)
    3. Concatenates all 2K+1 supports along the feature dimension
    4. Applies a single learned linear projection

This means the layer sees not just each node's own signal, but a weighted
combination of its 1-hop, 2-hop, ..., K-hop neighbourhood — in both the
forward (outgoing traffic) and backward (incoming traffic) directions.

Why bidirectional? Traffic causality is directional — an accident upstream
affects downstream sensors, not the reverse. The two transition matrices
T_f (row-normalised A) and T_b (row-normalised A^T) capture both directions.

Shape conventions used throughout:
    B  = batch size
    N  = number of sensors (nodes)
    F  = input features per node
    H  = output features per node (hidden dim)
    K  = diffusion steps
"""

import torch
import torch.nn as nn


class DiffusionConv(nn.Module):
    """
    Graph diffusion convolution layer.

    Replaces a standard nn.Linear(in_features, out_features) with a
    graph-aware equivalent that aggregates K-hop neighbourhood information
    in both forward and backward directions along the graph.

    Input:  (B, N, F)
    Output: (B, N, H)

    The effective input width is F * (2K + 1):
        K=0 term:  identity (the node's own signal)             → F features
        K=1..K forward:  T_f^k @ X for k=1..K                  → K*F features
        K=1..K backward: T_b^k @ X for k=1..K                  → K*F features
    Total: F * (2K + 1) → projected to H via a single linear layer.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        K: int = 3,
        bias: bool = True,
    ):
        """
        Args:
            in_features:  F — input features per node
            out_features: H — output features per node
            K:            number of diffusion steps (hops)
                          K=2 or K=3 is standard in DCRNN
            bias:         whether to add a learnable bias
        """
        super().__init__()
        self.K            = K
        self.in_features  = in_features
        self.out_features = out_features

        # 2K supports (forward + backward, excluding K=0 identity) + 1 identity
        # Total input width to the linear layer: in_features * (2K + 1)
        self.linear = nn.Linear(
            in_features  * (2 * K + 1),
            out_features,
            bias=bias,
        )

    def forward(
        self,
        x: torch.Tensor,
        T_f: torch.Tensor,
        T_b: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x:   (B, N, F)  — node features
            T_f: (N, N)     — forward  transition matrix (sparse or dense)
            T_b: (N, N)     — backward transition matrix (sparse or dense)

        Returns:
            out: (B, N, H)  — diffusion-convolved features
        """
        B, N, F = x.shape

        # Start with the K=0 term — the identity (node's own signal)
        supports = [x]                          # list of (B, N, F) tensors

        x_f = x    # running forward  diffusion state
        x_b = x    # running backward diffusion state

        for _ in range(self.K):
            x_f = self._graph_mm(T_f, x_f)     # one more hop forward
            x_b = self._graph_mm(T_b, x_b)     # one more hop backward
            supports.append(x_f)
            supports.append(x_b)

        # Concatenate all supports along feature dimension
        # Each support is (B, N, F) → cat → (B, N, F * (2K+1))
        out = torch.cat(supports, dim=-1)

        # Linear projection → (B, N, H)
        return self.linear(out)

    @staticmethod
    def _graph_mm(T: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Multiply transition matrix T (N, N) by batch of node features x (B, N, F).

        Always uses dense einsum — N=150 is too small for sparse to help,
        and sparse ops are unsupported on MPS (Apple Silicon GPU).

        Returns: (B, N, F)
        """
        return torch.einsum("nm,bmf->bnf", T, x)


class DiffusionConvResidual(nn.Module):
    """
    Diffusion convolution with a residual (skip) connection.

    If in_features == out_features, adds x directly.
    Otherwise projects x with a 1x1 linear before adding.

    This is not in the original DCRNN but helps with gradient flow
    when stacking multiple DCGRU layers.
    """

    def __init__(self, in_features: int, out_features: int, K: int = 3):
        super().__init__()
        self.conv = DiffusionConv(in_features, out_features, K)
        self.skip = (
            nn.Identity()
            if in_features == out_features
            else nn.Linear(in_features, out_features, bias=False)
        )
        self.norm = nn.LayerNorm(out_features)

    def forward(self, x, T_f, T_b):
        return self.norm(self.conv(x, T_f, T_b) + self.skip(x))


# ---------------------------------------------------------------------------
# Shape verification utility
# ---------------------------------------------------------------------------

def verify_diffusion_conv(N=150, B=4, F=1, H=64, K=3):
    """
    Quick shape check — call this to verify DiffusionConv works
    before plugging it into the GRU cell.
    """
    import numpy as np

    print(f"[DiffusionConv verify] B={B}, N={N}, F={F}, H={H}, K={K}")

    # Build dummy transition matrices (dense for simplicity here)
    A = np.random.rand(N, N).astype(np.float32)
    A = A / A.sum(axis=1, keepdims=True)           # row-normalise
    T_f = torch.FloatTensor(A)
    T_b = torch.FloatTensor(A.T / A.T.sum(axis=1, keepdims=True))

    layer = DiffusionConv(F, H, K=K)
    x     = torch.randn(B, N, F)

    out   = layer(x, T_f, T_b)
    print(f"  Input:  {tuple(x.shape)}")
    print(f"  Output: {tuple(out.shape)}  (expected ({B}, {N}, {H}))")
    assert out.shape == (B, N, H), f"Shape mismatch: {out.shape}"

    # Test backward pass
    loss = out.sum()
    loss.backward()
    grad = layer.linear.weight.grad
    print(f"  Gradient shape: {tuple(grad.shape)}  ✓")
    assert grad is not None, "No gradient!"

    print(f"  ✓ DiffusionConv verified\n")
    return layer


if __name__ == "__main__":
    verify_diffusion_conv()