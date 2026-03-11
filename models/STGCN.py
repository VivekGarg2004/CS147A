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
        # We output DOUBLE the out_channels because the GLU splits the tensor in half
        self.conv = nn.Conv2d(in_channels, 2 * out_channels, kernel_size=(1, kernel_size))

    def forward(self, x):
        # x shape: (Batch, in_channels, Nodes, Time)
        x = self.conv(x)
        
        # Split the tensor along the channel dimension (dim=1) into two halves: P and Q
        # P is the linear activation, Q will become the gate
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
        
        # Reshape to (Batch * Time, Nodes, Channels) for matrix multiplication
        x = x.permute(0, 3, 2, 1).contiguous().view(b * t, n, c)
        
        # 1. Feature transformation: X * W
        support = torch.matmul(x, self.weight) 
        
        # 2. Spatial aggregation: A * (X * W)
        output = torch.matmul(adj, support)
        
        # Reshape back to (Batch, out_channels, Nodes, Time)
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
        
        # Normalization (Standardizing across the Node dimension)
        self.batch_norm = nn.BatchNorm2d(num_nodes)

    def forward(self, x, adj):
        # 1. Temporal Gated-Conv
        x = self.temporal1(x)
        
        # 2. Spatial Graph-Conv
        x = self.spatial(x, adj)
        
        # 3. Temporal Gated-Conv
        x = self.temporal2(x)
        
        # 4. Normalization 
        # BatchNorm2d expects the structure (Batch, Channels, Height, Width). 
        # We permute Nodes to be the "Channel" equivalent so it normalizes per intersection.
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
        
        # --- The Time Shrinking Math ---
        # Every TimeBlock uses a kernel size of 3 and NO padding. This shrinks time by 2 steps.
        # Block 1 has two TimeBlocks -> shrinks by 4 (e.g., 12 -> 8)
        # Block 2 has two TimeBlocks -> shrinks by 4 (e.g., 8 -> 4)
        remaining_time_steps = historical_steps - 8 
        
        # Final Output Layer (Green block in diagram)
        self.final_fc = nn.Linear(64 * remaining_time_steps, pred_steps)

    def forward(self, x, adj):
        # x shape: (Batch, 1_feature, 150_nodes, 12_Historical_steps)
        
        # Pass through the stacked blocks
        x = self.block1(x, adj)
        x = self.block2(x, adj)
        
        # Flatten the remaining time and feature dimensions for the final linear layer
        b, c, n, t = x.shape
        x = x.permute(0, 2, 1, 3).contiguous().view(b, n, c * t) 
        
        # Output shape: (Batch, 150_Nodes, Predicted_steps)
        out = self.final_fc(x) 
        return out