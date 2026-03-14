import torch
import torch.nn as nn
import torch.nn.functional as F

class TimeBlock(nn.Module):
    """
    Temporal Gated-Conv Layer (The GLU Mechanism).
    Replaces standard ReLUs with a gated linear unit to handle long-term sequences.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(TimeBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, 2 * out_channels, kernel_size=(1, kernel_size))

    def forward(self, x):
        # x shape: (Batch, in_channels, Nodes, Time)
        x = self.conv(x)
        
        P = x[:, :x.size(1)//2, :, :]
        Q = x[:, x.size(1)//2:, :, :]
        
        # GLU Operation: P * sigmoid(Q)
        return P * torch.sigmoid(Q)

class SpatialBlock(nn.Module):
    """
    Spatial Graph Convolution Layer.
    """
    def __init__(self, in_channels, out_channels):
        super(SpatialBlock, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_channels, out_channels))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, adj):
        # x shape: (Batch, Channels, Nodes, Time)
        b, c, n, t = x.shape
        
        x = x.permute(0, 3, 2, 1).contiguous().view(b * t, n, c)
        
        support = torch.matmul(x, self.weight) 
        
        output = torch.matmul(adj, support)
        
        output = output.view(b, t, n, -1).permute(0, 3, 2, 1).contiguous()
        return F.relu(output)

class STConvBlock(nn.Module):
    """
    Spatial-Temporal Convolution Block.
    The "sandwich" from the middle panel of the diagram.
    """
    def __init__(self, in_channels, spatial_channels, out_channels, num_nodes):
        super(STConvBlock, self).__init__()
        
        # Temporal Conv 1
        self.temporal1 = TimeBlock(in_channels=in_channels, out_channels=spatial_channels)
        
        # Spatial Graph Conv
        self.spatial = SpatialBlock(in_channels=spatial_channels, out_channels=spatial_channels)
        
        # Temporal Conv 2
        self.temporal2 = TimeBlock(in_channels=spatial_channels, out_channels=out_channels)
        
        # Norm
        self.batch_norm = nn.BatchNorm2d(num_nodes)

    def forward(self, x, adj):
        # Temporal Gated-Conv
        x = self.temporal1(x)
        
        # Spatial Graph-Conv
        x = self.spatial(x, adj)
        
        # Temporal Gated-Conv
        x = self.temporal2(x)
        
        # Normalization 
        x = x.permute(0, 2, 1, 3).contiguous() 
        x = self.batch_norm(x)
        x = x.permute(0, 2, 1, 3).contiguous() 
        
        return x

class STGCN_Model(nn.Module):
    """
    The main STGCN Model wrapper (Left panel of the diagram).
    """
    def __init__(self, num_nodes, in_features, historical_steps, pred_steps):
        super(STGCN_Model, self).__init__()
        
        # Block 1
        self.block1 = STConvBlock(in_channels=in_features, spatial_channels=16, out_channels=64, num_nodes=num_nodes)
        
        # Block 2
        self.block2 = STConvBlock(in_channels=64, spatial_channels=16, out_channels=64, num_nodes=num_nodes)
        
        # steps update
        remaining_time_steps = historical_steps - 8 
        
        # Final output
        self.final_fc = nn.Linear(64 * remaining_time_steps, pred_steps)

    def forward(self, x, adj):
        # x shape: (Batch, 1_feature, 150_nodes, 12_Historical_steps)
        
        x = self.block1(x, adj)
        x = self.block2(x, adj)
        
        b, c, n, t = x.shape
        x = x.permute(0, 2, 1, 3).contiguous().view(b, n, c * t) 
        
        out = self.final_fc(x) 
        return out