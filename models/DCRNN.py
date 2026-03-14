import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from diffusion_crnn.src.dcrnn_cell import DCGRUDecoder, DCGRUEncoder
from diffusion_crnn.src.learned_adjacency import LearnedAdjacency

class DCRNN(nn.Module):
    """
    Full DCRNN model with optional learned adjacency.

    Architecture:
        Encoder: L-layer stacked DCGRU reads input sequence
        Decoder: L-layer stacked DCGRU generates output sequence
        Graph:   fixed transition matrices + optional learned adjacency
    """

    def __init__(
        self,
        num_nodes: int,     # N
        in_features: int,
        hidden_dim: int,
        out_features: int,
        output_seq_len: int,
        num_layers: int      = 2,
        K: int               = 3,
        use_learned_adj: bool = True,
        adj_embed_dim: int   = 10,
        graph_mode: str      = "both",   # 'fixed' | 'learned' | 'both'
    ):
        """
        Args:
            num_nodes:       N — number of sensors
            in_features:     F — input features per node
            hidden_dim:      H — DCGRU hidden state size
            out_features:    output features per node (usually = in_features)
            output_seq_len:  number of future steps to predict
            num_layers:      L — stacked DCGRU layers
            K:               diffusion steps
            use_learned_adj: whether to learn an adaptive adjacency
            adj_embed_dim:   embedding dimension for learned adjacency
            graph_mode:      how to combine fixed and learned adjacency
        """
        super().__init__()

        self.num_nodes      = num_nodes
        self.hidden_dim     = hidden_dim
        self.output_seq_len = output_seq_len
        self.use_learned_adj = use_learned_adj
        self.graph_mode     = graph_mode

        # --- Learned adjacency ---
        if use_learned_adj:
            self.learned_adj = LearnedAdjacency(num_nodes, adj_embed_dim)

        # --- Encoder ---
        self.encoder = DCGRUEncoder(
            in_features = in_features,
            hidden_dim  = hidden_dim,
            num_layers  = num_layers,
            K           = K,
        )

        # --- Decoder ---
        self.decoder = DCGRUDecoder(
            in_features  = in_features,
            hidden_dim   = hidden_dim,
            out_features = out_features,
            num_layers   = num_layers,
            K            = K,
        )

    def _get_transition_matrices(self, T_f, T_b):
        """
        Combine fixed and learned adjacency based on graph_mode.

        Returns (T_f_eff, T_b_eff) to use in diffusion convolution.
        
        """

        # if fixed its always the same
        if not self.use_learned_adj or self.graph_mode == "fixed":
            return T_f, T_b

        # pytoch __call__ cals forward 
        A_learned = self.learned_adj()    # (N, N) — differentiable 

        # if learned/both its symmetric so return transpose as backward
        if self.graph_mode == "learned":
            return A_learned, A_learned.T

        # 'both': average fixed and learned (sparse was causing issues so convert to dense)
        T_f_dense = T_f.to_dense() if T_f.is_sparse else T_f
        T_b_dense = T_b.to_dense() if T_b.is_sparse else T_b

        T_f_eff = 0.5 * (T_f_dense + A_learned)
        T_b_eff = 0.5 * (T_b_dense + A_learned.T)
        return T_f_eff, T_b_eff

    def forward(
        self,
        x: torch.Tensor,
        T_f: torch.Tensor,
        T_b: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        teacher_forcing_prob: float = 0.0,
    ) -> torch.Tensor:
        """
        Args:
            x:                   (B, T_in, N, F)  — input sequence
            T_f:                 (N, N)            — forward  transition matrix
            T_b:                 (N, N)            — backward transition matrix
            targets:             (B, T_out, N, F)  — ground truth for teacher forcing
            teacher_forcing_prob: probability of using ground truth in decoder

        Returns:
            predictions: (B, T_out, N, F)
        """
        B, T_in, N, F = x.shape

        # Get effective transition matrices (fixed + learned)
        T_f_eff, T_b_eff = self._get_transition_matrices(T_f, T_b)

        # --- Encode then decode ---
        enc_hidden = self.encoder(x, T_f_eff, T_b_eff)
        predictions = self.decoder(
            enc_hidden, T_f_eff, T_b_eff,
            self.output_seq_len, targets, teacher_forcing_prob,
        )

        return predictions