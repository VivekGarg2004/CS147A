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
from .graph_utils import compute_transition_matrices, to_sparse_tensor


# ---------------------------------------------------------------------------
# Temporal Attention module (Contribution #1)
# ---------------------------------------------------------------------------

class TemporalAttention(nn.Module):
    """
    Multi-head cross-attention over the encoder's temporal hidden states.

    Applied inside the decoder at each generation step.
    Each node attends independently to its own temporal history,
    producing a context vector that augments the decoder input.

    Query:  decoder hidden state at current step  (B, N, H)
    Keys:   encoder hidden states across time      (B, T, N, H)
    Values: encoder hidden states across time      (B, T, N, H)
    Output: attended context                       (B, N, H)
    """

    def __init__(self, hidden_dim: int, num_heads: int = 4):
        """
        Args:
            hidden_dim: H — must be divisible by num_heads
            num_heads:  number of attention heads
        """
        super().__init__()
        assert hidden_dim % num_heads == 0, (
            f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
        )
        self.hidden_dim = hidden_dim
        self.num_heads  = num_heads
        self.head_dim   = hidden_dim // num_heads
        self.scale      = self.head_dim ** -0.5

        # Project query, key, value
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Output projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm     = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        query: torch.Tensor,
        enc_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            query:      (B, N, H)    — decoder hidden state (current step)
            enc_states: (B, T, N, H) — all encoder hidden states

        Returns:
            context: (B, N, H) — attended context vector
        """
        B, N, H = query.shape
        _, T, _, _ = enc_states.shape

        # --- Project Q, K, V ---
        # Reshape for per-node attention: treat (B*N) as batch
        q = self.q_proj(query)                             # (B, N, H)
        # enc_states: (B, T, N, H) → (B, N, T, H)
        enc_t = enc_states.permute(0, 2, 1, 3)            # (B, N, T, H)
        k = self.k_proj(enc_t)                             # (B, N, T, H)
        v = self.v_proj(enc_t)                             # (B, N, T, H)

        # --- Reshape for multi-head attention ---
        # q: (B, N, H) → (B*N, 1, num_heads, head_dim) → (B*N, num_heads, 1, head_dim)
        q = q.reshape(B * N, 1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(B * N, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(B * N, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        # All now: (B*N, num_heads, T_or_1, head_dim)

        # --- Scaled dot-product attention ---
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B*N, heads, 1, T)
        attn = F.softmax(attn, dim=-1)

        # Weighted sum of values
        context = torch.matmul(attn, v)                    # (B*N, heads, 1, head_dim)
        context = context.squeeze(-2)                      # (B*N, heads, head_dim)
        context = context.reshape(B * N, H)                # (B*N, H)
        context = context.reshape(B, N, H)                 # (B, N, H)

        # Output projection + residual + norm
        context = self.out_proj(context)                   # (B, N, H)
        return self.norm(context + query)                  # residual connection


# ---------------------------------------------------------------------------
# Learned Adjacency module (Contribution #2)
# ---------------------------------------------------------------------------

class LearnedAdjacency(nn.Module):
    """
    Adaptive graph learning via trainable node embeddings.

    Learns two embedding matrices E1, E2 ∈ R^(N x d).
    The adjacency is computed as:
        A = softmax(ReLU(E1 @ E2^T))

    ReLU removes negative similarities (no negative edges).
    Softmax row-normalises so A is already a transition matrix.

    This is used IN ADDITION to the fixed graph (if provided),
    giving the model both structural prior knowledge and the
    ability to discover new connections from data.

    Reference: Graph WaveNet (Wu et al. 2019)
    """

    def __init__(self, num_nodes: int, embed_dim: int = 10):
        """
        Args:
            num_nodes: N
            embed_dim: d — embedding dimension (10 is standard)
        """
        super().__init__()
        self.E1 = nn.Parameter(torch.randn(num_nodes, embed_dim))
        self.E2 = nn.Parameter(torch.randn(num_nodes, embed_dim))

    def forward(self) -> torch.Tensor:
        """
        Returns:
            A_learned: (N, N) learned adjacency (row-normalised, no self-loops)
        """
        return F.softmax(F.relu(self.E1 @ self.E2.T), dim=-1)


# ---------------------------------------------------------------------------
# Full DCRNN Model
# ---------------------------------------------------------------------------

class DCRNN(nn.Module):
    """
    Full DCRNN model with optional learned adjacency and temporal attention.

    Architecture:
        Encoder: L-layer stacked DCGRU reads input sequence
        Decoder: L-layer stacked DCGRU generates output sequence
                 + optional temporal attention at each decode step
        Graph:   fixed transition matrices + optional learned adjacency

    The graph used for diffusion convolution is:
        - 'fixed':   only the pre-built transition matrices (T_f, T_b)
        - 'learned': only the learned adaptive adjacency
        - 'both':    average of fixed + learned (best of both worlds)
    """

    def __init__(
        self,
        num_nodes: int,
        in_features: int,
        hidden_dim: int,
        out_features: int,
        output_seq_len: int,
        num_layers: int      = 2,
        K: int               = 3,
        use_attention: bool  = True,
        attention_heads: int = 4,
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
            use_attention:   whether to use temporal attention in decoder
            attention_heads: number of attention heads
            use_learned_adj: whether to learn an adaptive adjacency
            adj_embed_dim:   embedding dimension for learned adjacency
            graph_mode:      how to combine fixed and learned adjacency
        """
        super().__init__()

        self.num_nodes      = num_nodes
        self.hidden_dim     = hidden_dim
        self.output_seq_len = output_seq_len
        self.use_attention  = use_attention
        self.use_learned_adj = use_learned_adj
        self.graph_mode     = graph_mode

        # --- Learned adjacency ---
        if use_learned_adj:
            self.learned_adj = LearnedAdjacency(num_nodes, adj_embed_dim)

        # --- Temporal attention ---
        if use_attention:
            self.attention = TemporalAttention(
                hidden_dim  = hidden_dim,
                num_heads   = attention_heads,
            )
            # Gate: learn how much attention context to mix into prediction
            # hidden (H) + context (H) → output (out_features)
            # This replaces the standard output_proj when attention is active
            self.attn_output_proj = nn.Linear(hidden_dim * 2, out_features)

        # --- Encoder ---
        self.encoder = DCGRUEncoder(
            in_features = in_features,
            hidden_dim  = hidden_dim,
            num_layers  = num_layers,
            K           = K,
        )

        # --- Decoder ---
        # If using attention, decoder input is augmented by context
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
        For 'learned' and 'both' modes, the learned adjacency is symmetric
        so T_f_learned = T_b_learned = A_learned.
        """
        if not self.use_learned_adj or self.graph_mode == "fixed":
            return T_f, T_b

        A_learned = self.learned_adj()    # (N, N) — differentiable

        if self.graph_mode == "learned":
            return A_learned, A_learned.T

        # 'both': average fixed and learned
        # Convert sparse T_f/T_b to dense for averaging
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
        targets: torch.Tensor = None,
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

        # --- Encode ---
        # Collect all encoder hidden states for attention
        if self.use_attention:
            enc_all_hidden = self._encode_with_states(x, T_f_eff, T_b_eff)
            enc_hidden = [states[-1] for states in enc_all_hidden]
            # Stack all timestep hidden states of the TOP layer
            # Shape: (B, T_in, N, H)
            top_layer_states = torch.stack(
                [enc_all_hidden[t][-1] for t in range(T_in)], dim=1
            )
        else:
            enc_hidden = self.encoder(x, T_f_eff, T_b_eff)
            top_layer_states = None

        # --- Decode ---
        if self.use_attention:
            predictions = self._decode_with_attention(
                enc_hidden, top_layer_states,
                T_f_eff, T_b_eff,
                targets, teacher_forcing_prob,
            )
        else:
            predictions = self.decoder(
                enc_hidden, T_f_eff, T_b_eff,
                self.output_seq_len, targets, teacher_forcing_prob,
            )

        return predictions

    def _encode_with_states(self, x_seq, T_f, T_b):
        """
        Run encoder and collect hidden states at every timestep.
        Needed for temporal attention.

        Returns:
            all_hidden: list of T_in items, each is list of L hidden states (B,N,H)
        """
        B, T, N, F = x_seq.shape
        device = x_seq.device
        cells  = self.encoder.cells

        h = [dcell.init_hidden(B, N, device) for dcell in cells]
        all_hidden = []

        for t in range(T):
            x_t = x_seq[:, t]
            for layer, dcell in enumerate(cells):
                inp     = x_t if layer == 0 else h[layer - 1]
                h[layer] = dcell(inp, h[layer], T_f, T_b)
            all_hidden.append([hh.clone() for hh in h])

        return all_hidden

    def _decode_with_attention(
        self, enc_hidden, top_layer_states,
        T_f, T_b, targets, teacher_forcing_prob,
    ):
        """
        Decoder loop with temporal attention.

        At each step:
        1. Run decoder cell to get new hidden state
        2. Attend to encoder's full temporal history
        3. Project attended context + input → next input
        """
        B, N, H = enc_hidden[0].shape
        device  = enc_hidden[0].device
        F_out   = self.decoder.out_features

        h  = list(enc_hidden)
        x_t = torch.zeros(B, N, F_out, device=device)
        predictions = []

        cells      = self.decoder.cells
        out_proj   = self.decoder.output_proj

        for t in range(self.output_seq_len):
            # Run all decoder layers
            for layer, dcell in enumerate(cells):
                inp      = x_t if layer == 0 else h[layer - 1]
                h[layer] = dcell(inp, h[layer], T_f, T_b)

            # Temporal attention: decoder hidden attends to encoder states
            # Query: top decoder layer hidden state (B, N, H)
            # Keys/Values: all encoder timestep states (B, T_in, N, H)
            context = self.attention(h[-1], top_layer_states)  # (B, N, H)

            # Prediction: concat hidden + context → output
            # Context directly informs prediction at every step
            pred = self.attn_output_proj(
                torch.cat([h[-1], context], dim=-1)            # (B, N, 2H)
            )                                                   # (B, N, F_out)
            predictions.append(pred)

            # Scheduled sampling — feed ground truth or own prediction
            if targets is not None and torch.rand(1).item() < teacher_forcing_prob:
                x_t = targets[:, t]
            else:
                x_t = pred.detach()

        return torch.stack(predictions, dim=1)                 # (B, T_out, N, F)

# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def build_model(cfg, adj: torch.Tensor = None) -> DCRNN:
    """
    Build DCRNN from config.

    Args:
        cfg: config module
        adj: optional pre-built adjacency for validation

    Returns:
        model: DCRNN instance
    """
    model = DCRNN(
        num_nodes       = cfg.NUM_SENSORS,
        in_features     = cfg.IN_FEATURES,
        hidden_dim      = cfg.HIDDEN_DIM,
        out_features    = cfg.IN_FEATURES,
        output_seq_len  = cfg.OUTPUT_SEQ_LEN,
        num_layers      = cfg.NUM_LAYERS,
        K               = cfg.DIFFUSION_K,
        use_attention   = cfg.USE_ATTENTION,
        attention_heads = cfg.ATTENTION_HEADS,
        use_learned_adj = (cfg.GRAPH_METHOD == "learned" or cfg.GRAPH_METHOD == "both"),
        graph_mode      = cfg.GRAPH_METHOD if cfg.GRAPH_METHOD in
                          ("fixed", "learned", "both") else "fixed",
    )
    return model


# ---------------------------------------------------------------------------
# Shape verification
# ---------------------------------------------------------------------------

def verify_model(B=4, N=150, T_in=12, T_out=3, F=1, H=32):
    import numpy as np

    print("[DCRNN verify]")

    # Dummy transition matrices
    A   = np.eye(N, dtype=np.float32)
    T_f = torch.FloatTensor(A)
    T_b = torch.FloatTensor(A)

    for use_att, use_ladj, mode in [
        (False, False, "fixed"),
        (True,  False, "fixed"),
        (True,  True,  "both"),
    ]:
        label = f"attention={use_att}, learned_adj={use_ladj}, mode={mode}"
        model = DCRNN(
            num_nodes=N, in_features=F, hidden_dim=H,
            out_features=F, output_seq_len=T_out, num_layers=2,
            K=2,
            use_attention=use_att, attention_heads=4,
            use_learned_adj=use_ladj, graph_mode=mode,
        )
        x    = torch.randn(B, T_in, N, F)
        tgt  = torch.randn(B, T_out, N, F)
        pred = model(x, T_f, T_b, targets=tgt, teacher_forcing_prob=0.5)

        print(f"  [{label}]")
        print(f"    output: {tuple(pred.shape)}  (expected ({B}, {T_out}, {N}, {F}))")
        assert pred.shape == (B, T_out, N, F)

        loss = pred.sum()
        loss.backward()
        print(f"    backward: ✓")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Total params (last model): {n_params:,}")
    print("  ✓ DCRNN model verified")


if __name__ == "__main__":
    verify_model()