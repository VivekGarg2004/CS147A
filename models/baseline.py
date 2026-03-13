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
    
# ============================================================
# Baseline 2: Historical Average (HA)
# For each sensor, compute the average volume at each position
# within a weekly cycle from the training data, then look up
# the matching position for each test sample.
# ============================================================

# 1 hour interval → 24 per day → 168 per week
INTERVAL_PER_DAY  = 24
INTERVAL_PER_WEEK = INTERVAL_PER_DAY * 7

# Denormalize training data
train_raw = X_train.reshape(-1, 150)  # we only need a flat time series
# But it's easier to use the original volume_df directly
train_volume = volume_df.values[:train_end + INPUT_LEN]  # all raw training timesteps

# Compute weekly periodic average: for each position in the week, average across all weeks
n_train_steps = len(train_volume)
week_positions = np.arange(n_train_steps) % INTERVAL_PER_WEEK
ha_table = np.zeros((INTERVAL_PER_WEEK, 150))
ha_count = np.zeros((INTERVAL_PER_WEEK, 1))

for pos in range(INTERVAL_PER_WEEK):
    mask_pos = week_positions == pos
    if mask_pos.sum() > 0:
        ha_table[pos] = train_volume[mask_pos].mean(axis=0)
        ha_count[pos] = mask_pos.sum()

# For each test sample, find the week-position of the target timesteps
# The first target timestep index = val_end + i + INPUT_LEN
ha_preds = []
for i in range(len(y_test)):
    global_idx = val_end + i + INPUT_LEN  # index of first prediction target
    sample_preds = []
    for h in range(OUTPUT_LEN):
        pos = (global_idx + h) % INTERVAL_PER_WEEK
        sample_preds.append(ha_table[pos])
    ha_preds.append(sample_preds)

ha_preds = np.array(ha_preds)  # (n_test, output_len, 150)
ha_true  = y_test * std_np + mean_np

ha_mae  = np.mean(np.abs(ha_preds - ha_true))
ha_rmse = np.sqrt(np.mean((ha_preds - ha_true)**2))
mask = ha_true > 10.0
ha_mape = np.mean(np.abs((ha_preds[mask] - ha_true[mask]) / ha_true[mask])) * 100

print("=== Baseline 2: Historical Average (Weekly) ===")
print(f"MAE:  {ha_mae:.2f}")
print(f"RMSE: {ha_rmse:.2f}")
print(f"MAPE: {ha_mape:.2f}%")