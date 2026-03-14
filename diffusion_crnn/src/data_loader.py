"""
data_loader.py

Handles everything data-related:
  - Loading and normalizing the volume CSV
  - Sliding window dataset (input seq → target seq)
  - Train / val / test splits
  - PyTorch DataLoaders

The key object is TrafficDataset which returns:
    x: (input_seq_len,  N, 1)   — historical volumes (normalized)
    y: (output_seq_len, N, 1)   — future volumes     (normalized)

Targets are kept normalized so the model trains in normalized space.
At evaluation time use Scaler.inverse_transform() to get real units.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

class ZScoreScaler:
    """
    Per-sensor Z-score normalization.

    Fit on training data only, then apply to val/test to prevent leakage.
    Stores mean and std per sensor so we can invert predictions back to
    real traffic volume units (vehicles/hour).
    """

    def __init__(self):
        self.mean = None   # (1, N, 1)
        self.std  = None   # (1, N, 1)

    def fit(self, x: np.ndarray):
        """
        Args:
            x: (T, N) raw volume array — training slice only
        """
        self.mean = x.mean(axis=0, keepdims=True)          # (1, N)
        self.std  = x.std(axis=0,  keepdims=True)          # (1, N)
        # Avoid division by zero for sensors with constant signal
        self.std  = np.where(self.std < 1e-8, 1.0, self.std)

    def transform(self, x: np.ndarray) -> np.ndarray:
        """
        Args:
            x: (T, N) volume array
        Returns:
            (T, N) normalized array
        """
        assert self.mean is not None, "Call fit() before transform()"
        return (x - self.mean) / self.std

    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        """
        Invert normalization on model predictions.

        Args:
            x: (..., N, 1) tensor of normalized predictions
        Returns:
            (..., N, 1) tensor in original volume units
        """
        mean = torch.FloatTensor(self.mean).to(x.device)   # (1, N)
        std  = torch.FloatTensor(self.std).to(x.device)    # (1, N)
        mean = mean.unsqueeze(0).unsqueeze(-1)             # (1, N) → (1, 1, N, 1)
        std  = std.unsqueeze(0).unsqueeze(-1)              # (1, N) → (1, 1, N, 1)
        # Broadcast over all leading dimensions
        return x * std.unsqueeze(-1) + mean.unsqueeze(-1)

    def transform_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize a tensor (used for scheduled sampling targets)."""
        mean = torch.FloatTensor(self.mean).to(x.device)
        std  = torch.FloatTensor(self.std).to(x.device)
        mean = mean.unsqueeze(0).unsqueeze(-1)
        std  = std.unsqueeze(0).unsqueeze(-1)
        return (x - mean) / std


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TrafficDataset(Dataset):
    """
    Sliding window dataset over a (T, N) normalized volume array.

    Each sample is a pair:
        x: (input_seq_len,  N, 1)   normalized historical volumes
        y: (output_seq_len, N, 1)   normalized future volumes

    The window slides one step at a time, so consecutive samples overlap.
    Total samples = T - input_seq_len - output_seq_len + 1
    """

    def __init__(
        self,
        volume: np.ndarray,
        input_seq_len: int,
        output_seq_len: int,
    ):
        """
        Args:
            volume:         (T, N) normalized float32 array
            input_seq_len:  number of historical timesteps (e.g. 12)
            output_seq_len: number of future timesteps to predict (e.g. 12)
        """
        super().__init__()
        self.volume         = volume.astype(np.float32)
        self.input_seq_len  = input_seq_len
        self.output_seq_len = output_seq_len
        self.total_len      = input_seq_len + output_seq_len

        T = volume.shape[0]
        self.num_samples = T - self.total_len + 1

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Window: [idx, idx + total_len)
        window = self.volume[idx : idx + self.total_len]    # (total_len, N)

        x = window[: self.input_seq_len]                    # (in_len,  N)
        y = window[self.input_seq_len :]                    # (out_len, N)

        x = x[:, :, np.newaxis]
        y = y[:, :, np.newaxis]

        return torch.FloatTensor(x), torch.FloatTensor(y)


# ---------------------------------------------------------------------------
# Main loader function
# ---------------------------------------------------------------------------

def load_dataset(
    volume_path: str,
    input_seq_len: int,
    output_seq_len: int,
    train_ratio: float  = 0.7,
    val_ratio: float    = 0.1,
    batch_size: int     = 32,
    num_workers: int    = 0,
    num_sensors: int    = 150,
):
    """
    Full pipeline: CSV → normalized splits → DataLoaders + scaler.

    Splits are done on the time axis (no shuffling of time order):
        [0          : train_end]          → train
        [train_end  : val_end]            → val
        [val_end    : T]                  → test

    Args:
        volume_path:    path to sensor_volume_150.csv
        input_seq_len:  historical window length
        output_seq_len: forecast horizon
        train_ratio:    fraction of timesteps for training
        val_ratio:      fraction of timesteps for validation
        batch_size:     DataLoader batch size
        num_workers:    DataLoader workers (0 = main process)
        num_sensors:    expected number of sensors (for shape validation)

    Returns:
        train_loader, val_loader, test_loader: PyTorch DataLoaders
        scaler: fitted ZScoreScaler (needed to invert predictions)
        volume_raw: (T, N) unnormalized array (useful for analysis)
    """

    # -----------------------------------------------------------------------
    # Load CSV
    # -----------------------------------------------------------------------
    df = pd.read_csv(volume_path, header=None)
    volume = df.values.astype(np.float32)

    # Ensure shape is (T, N) the issue was that rows were volume/time and columns were sensors
    if volume.shape[1] != num_sensors:
        volume = volume.T
    assert volume.shape[1] == num_sensors, (
        f"Expected {num_sensors} sensors, got shape {volume.shape}"
    )

    T, N = volume.shape
    volume_raw = volume.copy()

    print(f"[data] Loaded volume: T={T}, N={N}")
    print(f"[data] Volume stats — min={volume.min():.1f}, "
          f"max={volume.max():.1f}, mean={volume.mean():.1f}")

    # -----------------------------------------------------------------------
    # Time-based split indices
    # -----------------------------------------------------------------------
    train_end = int(T * train_ratio)
    val_end   = int(T * (train_ratio + val_ratio))

    print(f"[data] Split: train=[0:{train_end}], "
          f"val=[{train_end}:{val_end}], test=[{val_end}:{T}]")
    print(f"[data]        train={train_end}h, val={val_end-train_end}h, "
          f"test={T-val_end}h")

    # -----------------------------------------------------------------------
    # Fit scaler on training data only
    # -----------------------------------------------------------------------
    scaler = ZScoreScaler()
    scaler.fit(volume[:train_end])

    print(f"[data] Scaler — mean range: [{scaler.mean.min():.1f}, "
          f"{scaler.mean.max():.1f}], "
          f"std range: [{scaler.std.min():.1f}, {scaler.std.max():.1f}]")

    # -----------------------------------------------------------------------
    # Normalize full array
    # -----------------------------------------------------------------------
    volume_norm = scaler.transform(volume)                  # (T, N)

    # -----------------------------------------------------------------------
    # Build datasets
    # -----------------------------------------------------------------------
    train_data = volume_norm[:train_end]
    val_data   = volume_norm[train_end : val_end]
    test_data  = volume_norm[val_end :]

    train_dataset = TrafficDataset(train_data, input_seq_len, output_seq_len)
    val_dataset   = TrafficDataset(val_data,   input_seq_len, output_seq_len)
    test_dataset  = TrafficDataset(test_data,  input_seq_len, output_seq_len)

    print(f"[data] Samples — train={len(train_dataset)}, "
          f"val={len(val_dataset)}, test={len(test_dataset)}")

    # -----------------------------------------------------------------------
    # DataLoaders
    # -----------------------------------------------------------------------
    # Shuffle training only — val/test must stay in time order
    train_loader = DataLoader(
        train_dataset,
        batch_size  = batch_size,
        shuffle     = True,
        num_workers = num_workers,
        pin_memory  = False,
        drop_last   = True,    # avoid partial batches with variable graph ops
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = False,
    )

    print(f"[data] Batches — train={len(train_loader)}, "
          f"val={len(val_loader)}, test={len(test_loader)}")

    return train_loader, val_loader, test_loader, scaler, volume_raw


# ---------------------------------------------------------------------------
# Metrics (live here since they need the scaler)
# ---------------------------------------------------------------------------

def masked_mae(pred: torch.Tensor, target: torch.Tensor,
               null_val: float = 0.0) -> torch.Tensor:
    """
    MAE ignoring positions where target == null_val.
    Traffic datasets often have zero-volume entries (sensor outages, midnight)
    that would otherwise dominate the loss. Masking them out is standard practice
    from the original DCRNN paper.

    Args:
        pred:     (...) predictions in original or normalized units
        target:   (...) ground truth, same units
        null_val: value to mask out (default 0.0)
    """
    mask = (target != null_val).float()
    mask /= mask.mean().clamp(min=1e-8)     # normalize so loss scale is stable
    loss = torch.abs(pred - target) * mask
    return loss.mean()


def masked_rmse(pred: torch.Tensor, target: torch.Tensor,
                null_val: float = 0.0) -> torch.Tensor:
    mask = (target != null_val).float()
    mask /= mask.mean().clamp(min=1e-8)
    loss = ((pred - target) ** 2) * mask
    return torch.sqrt(loss.mean())


def masked_mape(pred: torch.Tensor, target: torch.Tensor,
                null_val: float = 0.0) -> torch.Tensor:
    """
    MAPE is unstable when target is near zero (division explodes).
    We mask out targets below a small threshold in addition to null_val.
    """
    mask = ((target != null_val) & (target.abs() > 1e-3)).float()
    mask /= mask.mean().clamp(min=1e-8)
    loss = torch.abs((pred - target) / target.clamp(min=1e-3)) * mask
    return loss.mean() * 100.0     # return as percentage


def compute_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    scaler: ZScoreScaler,
    horizons: list = [3, 6, 12],
):
    """
    Compute MAE, RMSE, MAPE at specific forecast horizons in original units.

    Args:
        pred:     (batch, output_seq_len, N, 1) normalized predictions
        target:   (batch, output_seq_len, N, 1) normalized targets
        scaler:   fitted ZScoreScaler
        horizons: list of 1-indexed timestep positions to evaluate at

    Returns:
        dict of {horizon: {mae, rmse, mape}}
    """
    # Invert normalization — evaluate in real volume units
    pred_real   = scaler.inverse_transform(pred)
    target_real = scaler.inverse_transform(target)

    results = {}
    for h in horizons:
        h_idx = h - 1    # convert to 0-indexed
        if h_idx >= pred_real.shape[1]:
            continue
        p = pred_real[:, h_idx]     # (batch, N, 1)
        t = target_real[:, h_idx]   # (batch, N, 1)
        results[h] = {
            "mae":  masked_mae(p, t).item(),
            "rmse": masked_rmse(p, t).item(),
            "mape": masked_mape(p, t).item(),
        }
    return results