import torch
import torch.nn as nn
import torch.nn.functional as F

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
        super().__init__()
        self.K            = K
        self.in_features  = in_features
        self.out_features = out_features

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
        B, N, F = x.shape
        supports = [x]

        x_f = x
        x_b = x

        for _ in range(self.K):
            x_f = self._graph_mm(T_f, x_f)
            x_b = self._graph_mm(T_b, x_b)
            supports.append(x_f)
            supports.append(x_b)

        out = torch.cat(supports, dim=-1)
        return self.linear(out)

    @staticmethod
    def _graph_mm(T: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum("nm,bmf->bnf", T, x)


class DCGRUCell(nn.Module):
    """
    Single DCGRU cell — one timestep of the encoder or decoder.
    """

    def __init__(
        self,
        in_features: int,
        hidden_dim: int,
        K: int = 3,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        combined_dim = in_features + hidden_dim

        self.gate_conv = DiffusionConv(
            in_features  = combined_dim,
            out_features = 2 * hidden_dim,
            K            = K,
        )

        self.cand_conv = DiffusionConv(
            in_features  = combined_dim,
            out_features = hidden_dim,
            K            = K,
        )

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        h: torch.Tensor,
        T_f: torch.Tensor,
        T_b: torch.Tensor,
    ) -> torch.Tensor:
        xh = torch.cat([x, h], dim=-1)

        gates = torch.sigmoid(self.gate_conv(xh, T_f, T_b))
        r, u  = gates.chunk(2, dim=-1)

        xrh      = torch.cat([x, r * h], dim=-1)
        c        = torch.tanh(self.cand_conv(xrh, T_f, T_b))

        h_new    = (1.0 - u) * h + u * c

        h_new    = self.norm(h_new)

        return h_new

    def init_hidden(self, batch_size: int, num_nodes: int, device: torch.device):
        return torch.zeros(batch_size, num_nodes, self.hidden_dim, device=device)


class DCGRUEncoder(nn.Module):
    """
    Stacked DCGRU encoder.
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
        B, T, N, F = x_seq.shape
        device = x_seq.device

        h = [dcell.init_hidden(B, N, device) for dcell in self.cells]

        for t in range(T):
            x_t = x_seq[:, t]
            for layer, dcell in enumerate(self.cells):
                inp  = x_t if layer == 0 else h[layer - 1]
                h[layer] = dcell(inp, h[layer], T_f, T_b)

        return h


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
            self.cells.append(DCGRUCell(in_dim, hidden_dim, K))

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

            if targets is not None and torch.rand(1).item() < teacher_forcing_prob:
                x_t = targets[:, t]
            else:
                x_t = pred.detach()

        return torch.stack(predictions, dim=1)


class TemporalAttention(nn.Module):
    """
    Multi-head cross-attention over the encoder's temporal hidden states.
    """

    def __init__(self, hidden_dim: int, num_heads: int = 4):
        super().__init__()
        assert hidden_dim % num_heads == 0, (
            f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
        )
        self.hidden_dim = hidden_dim
        self.num_heads  = num_heads
        self.head_dim   = hidden_dim // num_heads
        self.scale      = self.head_dim ** -0.5

        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm     = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        query: torch.Tensor,
        enc_states: torch.Tensor,
    ) -> torch.Tensor:
        B, N, H = query.shape
        _, T, _, _ = enc_states.shape

        q = self.q_proj(query)
        enc_t = enc_states.permute(0, 2, 1, 3)
        k = self.k_proj(enc_t)
        v = self.v_proj(enc_t)

        q = q.reshape(B * N, 1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(B * N, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(B * N, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)

        context = torch.matmul(attn, v)
        context = context.squeeze(-2)
        context = context.reshape(B * N, H)
        context = context.reshape(B, N, H)

        context = self.out_proj(context)
        return self.norm(context + query)


class LearnedAdjacency(nn.Module):
    """
    Adaptive graph learning via trainable node embeddings.
    """

    def __init__(self, num_nodes: int, embed_dim: int = 10):
        super().__init__()
        self.E1 = nn.Parameter(torch.randn(num_nodes, embed_dim))
        self.E2 = nn.Parameter(torch.randn(num_nodes, embed_dim))

    def forward(self) -> torch.Tensor:
        return F.softmax(F.relu(self.E1 @ self.E2.T), dim=-1)


class DCRNNModel(nn.Module):
    """
    Full DCRNN model with optional learned adjacency and temporal attention.
    """

    def __init__(
        self,
        num_nodes: int,
        in_features: int,
        hidden_dim: int,
        out_features: int,
        output_seq_len: int,
        num_layers: int      = 2,
        K: int               = 2,
        use_attention: bool  = False,
        attention_heads: int = 4,
        use_learned_adj: bool = True,
        adj_embed_dim: int   = 10,
        graph_mode: str      = "both",
    ):
        super().__init__()

        self.num_nodes      = num_nodes
        self.hidden_dim     = hidden_dim
        self.output_seq_len = output_seq_len
        self.use_attention  = use_attention
        self.use_learned_adj = use_learned_adj
        self.graph_mode     = graph_mode

        if use_learned_adj:
            self.learned_adj = LearnedAdjacency(num_nodes, adj_embed_dim)

        if use_attention:
            self.attention = TemporalAttention(
                hidden_dim  = hidden_dim,
                num_heads   = attention_heads,
            )
            self.attn_output_proj = nn.Linear(hidden_dim * 2, out_features)

        self.encoder = DCGRUEncoder(
            in_features = in_features,
            hidden_dim  = hidden_dim,
            num_layers  = num_layers,
            K           = K,
        )

        self.decoder = DCGRUDecoder(
            in_features  = in_features,
            hidden_dim   = hidden_dim,
            out_features = out_features,
            num_layers   = num_layers,
            K            = K,
        )

    def _get_transition_matrices(self, T_f, T_b):
        if not self.use_learned_adj or self.graph_mode == "fixed":
            return T_f, T_b

        A_learned = self.learned_adj()

        if self.graph_mode == "learned":
            return A_learned, A_learned.T

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
        B, T_in, N, F = x.shape

        T_f_eff, T_b_eff = self._get_transition_matrices(T_f, T_b)

        if self.use_attention:
            enc_all_hidden = self._encode_with_states(x, T_f_eff, T_b_eff)
            enc_hidden = [states[-1] for states in enc_all_hidden]
            top_layer_states = torch.stack(
                [enc_all_hidden[t][-1] for t in range(T_in)], dim=1
            )
        else:
            enc_hidden = self.encoder(x, T_f_eff, T_b_eff)
            top_layer_states = None

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
        B, N, H = enc_hidden[0].shape
        device  = enc_hidden[0].device
        F_out   = self.decoder.out_features

        h  = list(enc_hidden)
        x_t = torch.zeros(B, N, F_out, device=device)
        predictions = []

        cells      = self.decoder.cells
        out_proj   = self.decoder.output_proj

        for t in range(self.output_seq_len):
            for layer, dcell in enumerate(cells):
                inp      = x_t if layer == 0 else h[layer - 1]
                h[layer] = dcell(inp, h[layer], T_f, T_b)

            context = self.attention(h[-1], top_layer_states)

            pred = self.attn_output_proj(
                torch.cat([h[-1], context], dim=-1)
            )
            predictions.append(pred)

            if targets is not None and torch.rand(1).item() < teacher_forcing_prob:
                x_t = targets[:, t]
            else:
                x_t = pred.detach()

        return torch.stack(predictions, dim=1)