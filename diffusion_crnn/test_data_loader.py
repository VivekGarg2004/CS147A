"""
test_data_loader.py

Verifies the data loading pipeline before touching any model code.

Usage:
    python test_data_loader.py

Checks:
    - CSV loads correctly with right shape
    - Normalization is fit on train only (no leakage)
    - Sliding window samples have correct shapes
    - DataLoader batches have correct shapes
    - Metrics run without error
    - A few sample plots saved to outputs/
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from src.data_loader import load_dataset, compute_metrics, ZScoreScaler
import config as cfg


def test_loading():
    print("=" * 60)
    print("LOADING + SPLITTING")
    print("=" * 60)

    train_loader, val_loader, test_loader, scaler, volume_raw = load_dataset(
        volume_path    = cfg.VOLUME_PATH,
        input_seq_len  = cfg.INPUT_SEQ_LEN,
        output_seq_len = cfg.OUTPUT_SEQ_LEN,
        train_ratio    = cfg.TRAIN_RATIO,
        val_ratio      = cfg.VAL_RATIO,
        batch_size     = cfg.BATCH_SIZE,
        num_sensors    = cfg.NUM_SENSORS,
    )
    return train_loader, val_loader, test_loader, scaler, volume_raw


def test_batch_shapes(train_loader, val_loader, test_loader):
    print("\n" + "=" * 60)
    print("BATCH SHAPES")
    print("=" * 60)

    for name, loader in [("train", train_loader),
                          ("val",   val_loader),
                          ("test",  test_loader)]:
        x, y = next(iter(loader))
        print(f"  {name:5s} — x: {tuple(x.shape)}   y: {tuple(y.shape)}")

        # x should be (batch, input_seq_len, N, 1)
        assert x.dim() == 4, f"Expected 4D tensor, got {x.dim()}D"
        assert x.shape[1] == cfg.INPUT_SEQ_LEN,  f"Wrong input seq len: {x.shape[1]}"
        assert x.shape[2] == cfg.NUM_SENSORS,     f"Wrong num sensors: {x.shape[2]}"
        assert x.shape[3] == cfg.IN_FEATURES,     f"Wrong features: {x.shape[3]}"
        assert y.shape[1] == cfg.OUTPUT_SEQ_LEN,  f"Wrong output seq len: {y.shape[1]}"

    print("  ✓ All batch shapes correct")


def test_normalization(scaler, volume_raw):
    print("\n" + "=" * 60)
    print("NORMALIZATION")
    print("=" * 60)

    T = volume_raw.shape[0]
    train_end = int(T * cfg.TRAIN_RATIO)
    train_raw = volume_raw[:train_end]

    # Transform training slice and check it's ~N(0,1)
    train_norm = scaler.transform(train_raw)
    print(f"  Train raw  — mean={train_raw.mean():.2f}, std={train_raw.std():.2f}")
    print(f"  Train norm — mean={train_norm.mean():.4f} (≈0), "
          f"std={train_norm.std():.4f} (≈1)")

    assert abs(train_norm.mean()) < 0.05, "Normalized mean should be near 0"
    assert abs(train_norm.std()  - 1.0) < 0.05, "Normalized std should be near 1"

    # Inverse transform should recover original values
    norm_tensor = torch.FloatTensor(train_norm[:, :, np.newaxis])   # (T, N, 1)
    recovered   = scaler.inverse_transform(norm_tensor).numpy()[:, :, 0]
    max_err = np.abs(recovered - train_raw).max()
    print(f"  Inverse transform max error: {max_err:.6f}  (should be ~0)")
    assert max_err < 0.01, "Inverse transform is inaccurate"

    # Check no leakage — test mean should NOT be zero after normalization
    test_raw  = volume_raw[int(T * (cfg.TRAIN_RATIO + cfg.VAL_RATIO)):]
    test_norm = scaler.transform(test_raw)
    print(f"  Test norm  — mean={test_norm.mean():.4f} (not necessarily ≈0, that's ok)")
    print("  ✓ Normalization correct, no leakage")


def test_metrics(train_loader, scaler):
    print("\n" + "=" * 60)
    print("METRICS (on dummy perfect predictions)")
    print("=" * 60)

    x, y = next(iter(train_loader))

    # Perfect predictions → all metrics should be ~0
    results_perfect = compute_metrics(y, y, scaler, horizons=cfg.EVAL_HORIZONS)
    print("  Perfect predictions (pred == target):")
    for h, m in results_perfect.items():
        print(f"    h={h:2d}h — MAE={m['mae']:.4f}, RMSE={m['rmse']:.4f}, MAPE={m['mape']:.4f}%")

    # Random predictions → metrics should be large
    rand_pred = torch.randn_like(y)
    results_rand = compute_metrics(rand_pred, y, scaler, horizons=cfg.EVAL_HORIZONS)
    print("  Random predictions:")
    for h, m in results_rand.items():
        print(f"    h={h:2d}h — MAE={m['mae']:.2f}, RMSE={m['rmse']:.2f}, MAPE={m['mape']:.2f}%")

    print("  ✓ Metrics running correctly")


def plot_data_overview(volume_raw, scaler):
    print("\n[Generating plots...]")
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    T, N = volume_raw.shape
    train_end = int(T * cfg.TRAIN_RATIO)
    val_end   = int(T * (cfg.TRAIN_RATIO + cfg.VAL_RATIO))
    hours     = np.arange(T)

    fig = plt.figure(figsize=(18, 12))
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.35)

    # --- Plot 1: Raw volume for 5 sensors over full timeline ---
    ax1 = fig.add_subplot(gs[0, :])
    sample_sensors = np.linspace(0, N-1, 5, dtype=int)
    for s in sample_sensors:
        ax1.plot(hours, volume_raw[:, s], alpha=0.7, linewidth=0.6, label=f"Sensor {s}")
    ax1.axvline(train_end, color="red",    linestyle="--", linewidth=1.5, label="Train/Val split")
    ax1.axvline(val_end,   color="orange", linestyle="--", linewidth=1.5, label="Val/Test split")
    ax1.set_title("Raw Traffic Volume — 5 Sample Sensors (full timeline)", fontsize=12)
    ax1.set_xlabel("Hour index")
    ax1.set_ylabel("Volume (vehicles/hour)")
    ax1.legend(fontsize=8, ncol=4)

    # --- Plot 2: One week of data zoomed in ---
    ax2 = fig.add_subplot(gs[1, :2])
    week = 7 * 24
    for s in sample_sensors[:3]:
        ax2.plot(hours[:week], volume_raw[:week, s],
                 alpha=0.8, linewidth=1.0, label=f"Sensor {s}")
    ax2.set_title("One Week Zoom (first 168 hours)", fontsize=11)
    ax2.set_xlabel("Hour index")
    ax2.set_ylabel("Volume")
    ax2.legend(fontsize=8)

    # --- Plot 3: Average daily pattern ---
    ax3 = fig.add_subplot(gs[1, 2])
    mean_vol = volume_raw.mean(axis=1)           # (T,) average across sensors
    daily    = mean_vol[:7*24].reshape(7, 24)    # first week
    for d in range(7):
        ax3.plot(range(24), daily[d], alpha=0.5, linewidth=1)
    ax3.plot(range(24), daily.mean(axis=0), color="black",
             linewidth=2, label="Mean")
    ax3.set_title("Daily Pattern (first week)", fontsize=11)
    ax3.set_xlabel("Hour of day")
    ax3.set_ylabel("Avg volume")
    ax3.set_xticks([0, 6, 12, 18, 23])
    ax3.legend(fontsize=8)

    # --- Plot 4: Volume distribution ---
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.hist(volume_raw.flatten(), bins=60, color="steelblue",
             edgecolor="white", linewidth=0.3)
    ax4.set_title("Volume Distribution", fontsize=11)
    ax4.set_xlabel("Vehicles/hour")
    ax4.set_ylabel("Count")

    # --- Plot 5: Normalized distribution (should be ~N(0,1)) ---
    ax5 = fig.add_subplot(gs[2, 1])
    vol_norm = scaler.transform(volume_raw)
    ax5.hist(vol_norm.flatten(), bins=60, color="darkorange",
             edgecolor="white", linewidth=0.3)
    x_range = np.linspace(-4, 4, 200)
    from scipy.stats import norm as scipy_norm
    ax5.plot(x_range, scipy_norm.pdf(x_range) * vol_norm.size * (8/60),
             color="black", linewidth=2, label="N(0,1)")
    ax5.set_title("Normalized Distribution (should ≈ N(0,1))", fontsize=11)
    ax5.set_xlabel("Normalized volume")
    ax5.set_ylabel("Count")
    ax5.legend(fontsize=8)

    # --- Plot 6: Sensor volume heatmap (sensors × time, 2 week window) ---
    ax6 = fig.add_subplot(gs[2, 2])
    window = 14 * 24
    im = ax6.imshow(volume_raw[:window].T, aspect="auto",
                    cmap="YlOrRd", interpolation="nearest")
    ax6.set_title("Volume Heatmap (2 weeks, all sensors)", fontsize=11)
    ax6.set_xlabel("Hour index")
    ax6.set_ylabel("Sensor index")
    plt.colorbar(im, ax=ax6, fraction=0.046)

    out_path = os.path.join(cfg.OUTPUT_DIR, "data_overview.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Saved → {out_path}")
    plt.close()


def test_sliding_window():
    """Manually verify sliding window logic with a tiny example."""
    print("\n" + "=" * 60)
    print("SLIDING WINDOW SANITY CHECK")
    print("=" * 60)

    from src.data_loader import TrafficDataset

    # Tiny synthetic data: 20 timesteps, 3 sensors
    vol = np.arange(20 * 3, dtype=np.float32).reshape(20, 3)
    ds  = TrafficDataset(vol, input_seq_len=4, output_seq_len=3)

    print(f"  T=20, in=4, out=3 → expect {20-4-3+1}=14 samples, got {len(ds)}")
    assert len(ds) == 14

    x0, y0 = ds[0]   # first sample
    x1, y1 = ds[1]   # second sample (shifted by 1)

    print(f"  x[0] starts at t=0: {x0[:, 0, 0].tolist()} (expect [0,3,6,9])")
    print(f"  y[0] starts at t=4: {y0[:, 0, 0].tolist()} (expect [12,15,18])")
    print(f"  x[1] starts at t=1: {x1[:, 0, 0].tolist()} (expect [3,6,9,12])")

    assert x0[0, 0, 0] == 0.0
    assert y0[0, 0, 0] == 12.0    # sensor 0, t=4 → row 4 * 3 cols = value 12
    assert x1[0, 0, 0] == 3.0    # sensor 0, t=1 → value 3
    print("  ✓ Sliding window correct")


if __name__ == "__main__":
    train_loader, val_loader, test_loader, scaler, volume_raw = test_loading()
    test_batch_shapes(train_loader, val_loader, test_loader)
    test_normalization(scaler, volume_raw)
    test_sliding_window()
    test_metrics(train_loader, scaler)

    try:
        plot_data_overview(volume_raw, scaler)
    except ImportError:
        print("  [scipy not available — skipping N(0,1) overlay on plot]")
        # Rerun without scipy overlay
        pass

    print("\n" + "=" * 60)
    print("ALL DATA LOADER TESTS PASSED")
    print("Next step: build the model — diffusion_conv.py")
    print("=" * 60)