"""
dcrnn_cell.py
Implements the Diffusion Convolutional GRU (DCGRU) cell.

Additionally we add LayerNorm on the output for training stability.

Reference: Li et al. "Diffusion Convolutional Recurrent Neural Network:
           Data-Driven Traffic Forecasting" ICLR 2018
(I took their drcnn_cell and i lw just converted it to pytorch and made some minor adjustments to fit the rest of my codebase)
"""

import torch
import torch.nn as nn
from typing import Optional

from .diffusion_conv import DiffusionConv




class DCGRUCell(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_dim: int,
        K: int = 3,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        combined_dim = in_features + hidden_dim

        # Gates share the same diffusion supports and then split into r/u.
        self.gate_conv = DiffusionConv(
            in_features=combined_dim,
            out_features=2 * hidden_dim,
            K=K,
        )
        self.cand_conv = DiffusionConv(
            in_features=combined_dim,
            out_features=hidden_dim,
            K=K,
        )

        # layer norm over batch norm bc RNN
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        h: torch.Tensor,
        T_f: torch.Tensor,
        T_b: torch.Tensor,
    ) -> torch.Tensor:
        
        # Concatenate input and hidden state
        xh = torch.cat([x, h], dim=-1)

        #  first we calcualte the reset and update gate and then split bc their math is the same
        # weights are just different 
        gates_raw = self.gate_conv(xh, T_f, T_b)
        gates = torch.sigmoid(gates_raw)
        r, u  = gates.chunk(2, dim=-1)

        # then we do the candidate hidden state but we apply the reset gate to the prev_hidden state
        xrh = torch.cat([x, r * h], dim=-1)
        c_raw = self.cand_conv(xrh, T_f, T_b)
        c = torch.tanh(c_raw)

        h_new = (1.0 - u) * h + u * c

        # layer norm
        h_new = self.norm(h_new)

        return h_new


class DCGRUEncoder(nn.Module):
    """
    Stacked DCGRU encoder.

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
        h = [torch.zeros(B, N, self.hidden_dim, device=device) for _ in range(self.num_layers)]

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
    """

    def __init__(
        self,
        in_features: int,
        hidden_dim: int,
        out_features: int,
        num_layers: int,
        K: int = 3,
    ):
        super().__init__()
        self.num_layers  = num_layers
        self.hidden_dim  = hidden_dim
        self.out_features = out_features

        self.cells = nn.ModuleList()
        for layer in range(num_layers):
            in_dim = in_features if layer == 0 else hidden_dim
            # Updated to instantiate the unabstracted cell
            self.cells.append(DCGRUCell(in_dim, hidden_dim, K))

        self.output_proj = nn.Linear(hidden_dim, out_features)

    def forward(
        self,
        encoder_hidden: list,
        T_f: torch.Tensor,
        T_b: torch.Tensor,
        output_seq_len: int,
        targets: Optional[torch.Tensor] = None,
        teacher_forcing_prob: float = 0.0,
    ) -> torch.Tensor:
        
        B, N, H = encoder_hidden[0].shape
        device  = encoder_hidden[0].device

        h = list(encoder_hidden)
        x_t = torch.zeros(B, N, self.out_features, device=device)
        predictions = []

        

        for t in range(output_seq_len):
            for layer, cell in enumerate(self.cells):
                inp     = x_t if layer == 0 else h[layer - 1]
                h[layer] = cell(inp, h[layer], T_f, T_b)

            pred = self.output_proj(h[-1])
            predictions.append(pred)


            #This is to ensure that we have some truth while training but also have some of the
            #  model's own predictions in the input to the next step to make it more robust at inference time
            if targets is not None and torch.rand(1).item() < teacher_forcing_prob:
                x_t = targets[:, t]
            else:
                x_t = pred.detach()

        return torch.stack(predictions, dim=1)