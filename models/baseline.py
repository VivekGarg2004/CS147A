import torch
import torch.nn as nn
import numpy as np

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


class HistoricalAverage:
    """
    Baseline 2: Historical Average (HA)
    For each sensor, compute the average volume at each position
    within a weekly cycle from the training data, then look up
    the matching position for each prediction sample.
    """
    def __init__(self, raw_data, train_ratio=0.7, val_ratio=0.1, seq_len=12, pre_len=3):
        n_total = len(raw_data)
        train_end = int(n_total * train_ratio)
        self.test_idx = int(n_total * (train_ratio + val_ratio))
        
        self.seq_len = seq_len
        self.pre_len = pre_len
        
        # 1 hour interval -> 24 per day -> 168 per week
        self.INTERVAL_PER_WEEK = 24 * 7
        
        train_volume = raw_data[:train_end]
        
        self.ha_table = np.zeros((self.INTERVAL_PER_WEEK, raw_data.shape[1]))
        n_train_steps = len(train_volume)
        week_positions = np.arange(n_train_steps) % self.INTERVAL_PER_WEEK
        
        for pos in range(self.INTERVAL_PER_WEEK):
            mask_pos = week_positions == pos
            if mask_pos.sum() > 0:
                self.ha_table[pos] = train_volume[mask_pos].mean(axis=0)

    def predict(self, test_index, num_samples):
        """
        Predicts using the weekly historical average for a batch.
        test_index: the start index of the batch in the test dataset.
        returns: predictions of shape (Batch, Nodes, Time)
        """
        ha_preds = []
        for i in range(num_samples):
            # The first target timestep index relative to original full dataset
            global_idx = self.test_idx + self.seq_len + test_index + i
            sample_preds = []
            for h in range(self.pre_len):
                pos = (global_idx + h) % self.INTERVAL_PER_WEEK
                sample_preds.append(self.ha_table[pos])
            
            # sample_preds is currently (Time, Nodes)
            # transpose to (Nodes, Time) to match STGCN format
            ha_preds.append(np.array(sample_preds).T)
            
        return np.array(ha_preds) # (Batch, Nodes, Time)
