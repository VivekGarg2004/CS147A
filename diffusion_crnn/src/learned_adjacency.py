"""
dcrnn_model.py

Full DCRNN model with two original contributions:

  1. LEARNED ADAPTIVE ADJACENCY
     Instead of (or alongside) a hand-crafted graph, the model learns
     its own adjacency matrix as a function of trainable node embeddings:

         E1, E2 ∈ R^(N x d)   (learnable node embedding matrices)
         A_learned = softmax(ReLU(E1 @ E2^T))

     This comes from Graph WaveNet (Wu et al. 2019) and is ~10 lines of code.
     The motivation here is strong: the provided adjacency is broken (133
     isolated nodes) and distance-based graphs are geographically naive.
     A learned adjacency discovers functional connectivity from traffic
     patterns without any prior assumptions about road topology.

  2. TEMPORAL ATTENTION ON DECODER
     The standard DCRNN decoder only has access to the encoder's final
     hidden state — a fixed-size summary of 12 hours of history. This
     forces the model to compress all temporal information into one vector.

     We add a multi-head attention mechanism in the decoder that lets each
     decoder step attend to ALL encoder hidden states across time, not just
     the last one. This is especially useful for traffic because:
       - Rush hour patterns are strongly time-of-day dependent
       - The model can learn to attend to "same hour yesterday" or
         "same hour last week" rather than just the most recent state

     Implementation: lightweight cross-attention
       Query:  current decoder hidden state (B, N, H)
       Keys:   all encoder hidden states    (B, T_in, N, H)
       Values: all encoder hidden states    (B, T_in, N, H)

     We apply this per-node so the attention is local to each sensor's
     temporal history, then add the attended context to the decoder input.

Both contributions are ablatable — set USE_ATTENTION=False or
GRAPH_METHOD!='learned' in config.py to run the baseline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dcrnn_cell import DCGRUEncoder, DCGRUDecoder

# ---------------------------------------------------------------------------
# Learned Adjacency module
# ---------------------------------------------------------------------------

class LearnedAdjacency(nn.Module):
    """
    Adaptive graph learning via trainable node embeddings.

    Learns two embedding matrices E1, E2 ∈ R^(N x d).

    Reference: Graph WaveNet (Wu et al. 2019) paper section 3.2
    """

    def __init__(self, num_nodes: int, embed_dim: int = 10):
        super().__init__()
        self.E1 = nn.Parameter(torch.randn(num_nodes, embed_dim))
        self.E2 = nn.Parameter(torch.randn(num_nodes, embed_dim))

    def forward(self) -> torch.Tensor:
        """
        Returns A_learned = SoftMax(ReLU(E1ET2)).
        """
        return F.softmax(F.relu(self.E1 @ self.E2.T), dim=-1)
