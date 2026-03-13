import torch
import torch.nn as nn

class GraphConv(nn.Module):
    """GCN layer: X' = relu(A_hat @ X @ W + b)"""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
                                
    def forward(self, x, A_hat):
        return torch.relu(self.linear(A_hat @ x))


class GCNLSTM(nn.Module):
    def __init__(self, num_nodes, in_features, gcn_hidden, lstm_hidden, output_len):
        super().__init__()
        self.gcn1 = GraphConv(in_features, gcn_hidden)
        self.gcn2 = GraphConv(gcn_hidden, gcn_hidden)
        self.dropout = nn.Dropout(0.3)

        self.lstm = nn.LSTM(gcn_hidden, lstm_hidden, batch_first=True,
                            num_layers=2, dropout=0.2)

        self.output = nn.Linear(lstm_hidden, output_len)
        self.output_len = output_len
        self.num_nodes = num_nodes

    def forward(self, x, A_hat):
        batch, T, N = x.shape

        gcn_out = []
        for t in range(T):
            xt = x[:, t, :].unsqueeze(-1)    
            ht = self.gcn1(xt, A_hat)          
            ht = self.dropout(ht)
            ht = self.gcn2(ht, A_hat)           
            gcn_out.append(ht)

        gcn_seq = torch.stack(gcn_out, dim=1)

        gcn_seq = gcn_seq.permute(0, 2, 1, 3)          
        gcn_seq = gcn_seq.reshape(batch * N, T, -1)        

        lstm_out, _ = self.lstm(gcn_seq)                   
        last = lstm_out[:, -1, :]                          

        out = self.output(last)                        
        out = out.reshape(batch, N, self.output_len)       
        return out.permute(0, 2, 1)                        