"""
dcrnn_cell.py

Implements the Diffusion Convolutional GRU (DCGRU) cell.

A standard GRU cell has three operations:
    reset gate:   r = sigmoid(W_r x  +  U_r h  +  b_r)
    update gate:  u = sigmoid(W_u x  +  U_u h  +  b_u)
    candidate:    c = tanh   (W_c x  +  U_c (r*h)  +  b_c)
    new hidden:   h = (1-u)*h + u*c

In a DCGRU, every matrix-vector multiply "W x" and "U h" is replaced by
diffusion convolution. So instead of:
    W_r x  →  DiffusionConv(x,  T_f, T_b)
    U_r h  →  DiffusionConv(h,  T_f, T_b)

Each gate combines the diffused input and diffused hidden state.
This is implemented by concatenating [x, h] along the feature dimension
and running a single DiffusionConv over the combined signal — equivalent
to the separate W and U multiplications but more efficient.

Additionally we add LayerNorm on the output for training stability.

Reference: Li et al. "Diffusion Convolutional Recurrent Neural Network:
           Data-Driven Traffic Forecasting" ICLR 2018
"""

import torch
import torch.nn as nn
from .diffusion_conv import DiffusionConv


class DCGRUCell(nn.Module):
    """
    Single DCGRU cell — one timestep of the encoder or decoder.

    Replaces all linear projections inside a GRU with DiffusionConv,
    making the gating mechanism graph-aware.

    Hidden state shape: (B, N, hidden_dim)
    """

    def __init__(
        self,
        in_features: int,
        hidden_dim: int,
        K: int = 3,
    ):
        """
        Args:
            in_features: F — input features per node per timestep
            hidden_dim:  H — hidden state size per node
            K:           diffusion steps
        """
        super().__init__()
        self.hidden_dim = hidden_dim

        # Each gate takes concatenated [x, h] as input
        # x: (B, N, F)  h: (B, N, H)  → cat → (B, N, F+H)
        combined_dim = in_features + hidden_dim

        # Reset and update gates combined into one conv for efficiency
        # Output dim = 2*H (first H = reset, second H = update)
        self.gate_conv = DiffusionConv(
            in_features  = combined_dim,
            out_features = 2 * hidden_dim,
            K            = K,
        )

        # Candidate hidden state
        self.cand_conv = DiffusionConv(
            in_features  = combined_dim,
            out_features = hidden_dim,
            K            = K,
        )

        # Layer norm for stability (applied to output hidden state)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        h: torch.Tensor,
        T_f: torch.Tensor,
        T_b: torch.Tensor,
    ) -> torch.Tensor:
        """
        One step of the DCGRU.

        Args:
            x:   (B, N, F)  — input at current timestep
            h:   (B, N, H)  — hidden state from previous timestep
            T_f: (N, N)     — forward  transition matrix
            T_b: (N, N)     — backward transition matrix

        Returns:
            h_new: (B, N, H) — updated hidden state
        """
        # Concatenate input and hidden state → (B, N, F+H)
        xh = torch.cat([x, h], dim=-1)

        # Compute reset and update gates jointly
        gates = torch.sigmoid(self.gate_conv(xh, T_f, T_b))   # (B, N, 2H)
        r, u  = gates.chunk(2, dim=-1)                         # each (B, N, H)

        # Candidate: use reset gate to filter previous hidden state
        xrh      = torch.cat([x, r * h], dim=-1)               # (B, N, F+H)
        c        = torch.tanh(self.cand_conv(xrh, T_f, T_b))   # (B, N, H)

        # GRU update rule
        h_new    = (1.0 - u) * h + u * c                       # (B, N, H)

        # Layer norm
        h_new    = self.norm(h_new)

        return h_new

    def init_hidden(self, batch_size: int, num_nodes: int, device: torch.device):
        """
        Initialise hidden state to zeros.

        Args:
            batch_size: B
            num_nodes:  N
            device:     torch device

        Returns:
            (B, N, H) zero tensor
        """
        return torch.zeros(batch_size, num_nodes, self.hidden_dim, device=device)


class DCGRUEncoder(nn.Module):
    """
    Stacked DCGRU encoder.

    Reads an input sequence of length T and produces a sequence of
    hidden states — one per layer — that summarise the history.
    These hidden states are passed to the decoder as its initial state.

    Input:  (B, T, N, F)
    Output: list of L hidden states, each (B, N, H)
    """

    def __init__(
        self,
        in_features: int,
        hidden_dim: int,
        num_layers: int,
        K: int = 3,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        # Build stacked cells — first cell takes raw input,
        # subsequent cells take the output of the previous cell
        self.cells = nn.ModuleList()
        for layer in range(num_layers):
            in_dim = in_features if layer == 0 else hidden_dim
            self.cells.append(DCGRUCell(in_dim, hidden_dim, K))

    def forward(
        self,
        x_seq: torch.Tensor,
        T_f: torch.Tensor,
        T_b: torch.Tensor,
    ):
        """
        Args:
            x_seq: (B, T, N, F)  — input sequence
            T_f:   (N, N)        — forward  transition matrix
            T_b:   (N, N)        — backward transition matrix

        Returns:
            hidden_states: list of L tensors, each (B, N, H)
                           the final hidden state of each layer
        """
        B, T, N, F = x_seq.shape
        device = x_seq.device

        # Initialise hidden states for all layers
        h = [dcell.init_hidden(B, N, device) for dcell in self.cells]

        # Step through the input sequence
        for t in range(T):
            x_t = x_seq[:, t]              # (B, N, F)

            for layer, dcell in enumerate(self.cells):
                # Each layer's input is previous layer's output
                inp  = x_t if layer == 0 else h[layer - 1]
                h[layer] = dcell(inp, h[layer], T_f, T_b)

        return h   # list of L tensors (B, N, H)


class DCGRUDecoder(nn.Module):
    """
    Stacked DCGRU decoder with scheduled sampling.

    Generates output_seq_len predictions autoregressively.
    At each step it predicts the next timestep, then feeds that prediction
    (or the ground truth, depending on scheduled sampling probability)
    as input to the next step.

    Scheduled sampling curriculum:
        - Early training:  always feed ground truth (teacher forcing)
        - Later training:  increasingly feed own predictions
        - Inference:       always feed own predictions

    This prevents the model from becoming too dependent on ground truth
    at training time, improving generalisation.

    Input:  initial hidden states from encoder + optional target sequence
    Output: (B, output_seq_len, N, F)
    """

    def __init__(
        self,
        in_features: int,
        hidden_dim: int,
        out_features: int,
        num_layers: int,
        K: int = 3,
    ):
        """
        Args:
            in_features:  F — features fed as input each step (same as encoder)
            hidden_dim:   H — hidden state size
            out_features: output features per node (usually same as in_features)
            num_layers:   L — number of stacked DCGRU layers
            K:            diffusion steps
        """
        super().__init__()
        self.num_layers  = num_layers
        self.hidden_dim  = hidden_dim
        self.out_features = out_features

        self.cells = nn.ModuleList()
        for layer in range(num_layers):
            in_dim = in_features if layer == 0 else hidden_dim
            self.cells.append(DCGRUCell(in_dim, hidden_dim, K))

        # Output projection: hidden state → prediction
        # Applied at every timestep to produce the forecast
        self.output_proj = nn.Linear(hidden_dim, out_features)

    def forward(
        self,
        encoder_hidden: list,
        T_f: torch.Tensor,
        T_b: torch.Tensor,
        output_seq_len: int,
        targets: torch.Tensor = None,
        teacher_forcing_prob: float = 0.0,
    ) -> torch.Tensor:
        """
        Args:
            encoder_hidden:      list of L tensors (B, N, H) from encoder
            T_f:                 (N, N) forward  transition matrix
            T_b:                 (N, N) backward transition matrix
            output_seq_len:      number of steps to generate
            targets:             (B, output_seq_len, N, F) ground truth
                                 required during training for teacher forcing
            teacher_forcing_prob: probability [0,1] of using ground truth
                                  0.0 = always use own predictions (inference)
                                  1.0 = always use ground truth (teacher forcing)

        Returns:
            predictions: (B, output_seq_len, N, out_features)
        """
        B, N, H = encoder_hidden[0].shape
        device  = encoder_hidden[0].device

        # Decoder starts with encoder's final hidden states
        h = list(encoder_hidden)

        # First decoder input: zeros (the model has to predict from scratch)
        x_t = torch.zeros(B, N, self.out_features, device=device)

        predictions = []

        for t in range(output_seq_len):
            # Run through all decoder layers
            for layer, cell in enumerate(self.cells):
                inp     = x_t if layer == 0 else h[layer - 1]
                h[layer] = cell(inp, h[layer], T_f, T_b)

            # Project top hidden state to prediction
            pred = self.output_proj(h[-1])              # (B, N, out_features)
            predictions.append(pred)

            # Scheduled sampling: decide input for next step
            if targets is not None and torch.rand(1).item() < teacher_forcing_prob:
                # Use ground truth
                x_t = targets[:, t]                     # (B, N, F)
            else:
                # Use own prediction
                x_t = pred.detach()

        # Stack along time dimension → (B, output_seq_len, N, out_features)
        return torch.stack(predictions, dim=1)


# ---------------------------------------------------------------------------
# Shape verification
# ---------------------------------------------------------------------------

def verify_dcrnn_cell(B=4, N=150, F=1, H=64, K=3, T_in=12, T_out=3):
    import numpy as np

    print(f"[DCGRUCell verify] B={B}, N={N}, F={F}, H={H}, K={K}")

    # Dense transition matrices for simplicity
    A    = np.eye(N, dtype=np.float32) + np.random.rand(N, N).astype(np.float32) * 0.1
    A   /= A.sum(axis=1, keepdims=True)
    T_f  = torch.FloatTensor(A)
    T_b  = torch.FloatTensor(A.T / A.T.sum(axis=1, keepdims=True))

    # --- Single cell ---
    cell  = DCGRUCell(F, H, K=K)
    x     = torch.randn(B, N, F)
    h     = cell.init_hidden(B, N, torch.device("mps"))
    h_new = cell(x, h, T_f, T_b)
    print(f"  Cell output: {tuple(h_new.shape)}  (expected ({B}, {N}, {H}))")
    assert h_new.shape == (B, N, H)

    # --- Encoder ---
    encoder  = DCGRUEncoder(F, H, num_layers=2, K=K)
    x_seq    = torch.randn(B, T_in, N, F)
    enc_h    = encoder(x_seq, T_f, T_b)
    print(f"  Encoder hidden: {len(enc_h)} layers, "
          f"each {tuple(enc_h[0].shape)}  (expected ({B}, {N}, {H}))")
    assert len(enc_h) == 2
    assert enc_h[0].shape == (B, N, H)

    # --- Decoder (inference mode — no teacher forcing) ---
    decoder  = DCGRUDecoder(F, H, out_features=F, num_layers=2, K=K)
    preds    = decoder(enc_h, T_f, T_b, output_seq_len=T_out)
    print(f"  Decoder output: {tuple(preds.shape)}  "
          f"(expected ({B}, {T_out}, {N}, {F}))")
    assert preds.shape == (B, T_out, N, F)

    # --- Decoder (training mode — with teacher forcing) ---
    targets  = torch.randn(B, T_out, N, F)
    preds_tf = decoder(enc_h, T_f, T_b, T_out,
                       targets=targets, teacher_forcing_prob=0.5)
    assert preds_tf.shape == (B, T_out, N, F)

    # --- Backward pass ---
    loss = preds.sum()
    loss.backward()
    print(f"  Backward pass: ✓")

    print(f"  ✓ DCGRU encoder/decoder verified\n")


if __name__ == "__main__":
    from diffusion_conv import verify_diffusion_conv
    verify_diffusion_conv()
    verify_dcrnn_cell()