r"""
diffusion_conv.py

implements 

$$Z = \sum_{k=0}^{K-1} \left(\theta_{k,f} P_f^k X + \theta_{k,b} P_b^k X \right)$$


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

        # 0 has only one support and after that it is 2 support per step (forward and backward)
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

        # k=0 support is just the input features (B, N, F)
        supports = [x]                          

        x_f = x    # running forward  diffusion state
        x_b = x    # running backward diffusion state

        for _ in range(self.K):
            x_f = self._graph_mm(T_f, x_f)     # one more hop forward
            x_b = self._graph_mm(T_b, x_b)     # one more hop backward
            supports.append(x_f)
            supports.append(x_b)

        # matrix of [Z_0 | Z_f1 | Z_b1 | Z_f2 | Z_b2 | ...] (B, N, F * (2K+1))
        out = torch.cat(supports, dim=-1)

        # this is the weight matrix multiplication across the feature dimension
        return self.linear(out)

    @staticmethod
    def _graph_mm(T: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Multiply transition matrix T (N, N) by batch of node features x (B, N, F).

        Returns: (B, N, F)
        """
        return torch.einsum("nm,bmf->bnf", T, x)



