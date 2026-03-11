import torch
import torch.nn as nn

class NaiveBaseline(nn.Module):
    """
    A Persistence Baseline Model.
    It takes the very last observed time step in the historical window
    and simply repeats it for all future prediction steps.
    """
    def __init__(self, pred_steps):
        super(NaiveBaseline, self).__init__()
        self.pred_steps = pred_steps

    def forward(self, x, adj=None):
        # x shape: (Batch, Channels, Nodes, Time)
        # e.g., (32, 1, 150, 12)
        
        # 1. Extract the very last observed time step
        # We take channel 0, all nodes, and the last time index (-1)
        last_step = x[:, 0, :, -1] # Shape: (Batch, Nodes)
        
        # 2. Repeat this last step for however many steps we are predicting
        # Adds a dimension at the end and repeats it
        # Shape: (Batch, Nodes, 1) -> (Batch, Nodes, pred_steps)
        out = last_step.unsqueeze(-1).repeat(1, 1, self.pred_steps)
        
        return out