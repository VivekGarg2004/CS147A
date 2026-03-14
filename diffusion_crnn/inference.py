import os
import sys
import json
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.dirname(__file__))

import config as cfg
from src.data_loader  import load_dataset, compute_metrics, ZScoreScaler
from src.graph_utils  import (build_distance_adjacency,
                               prepare_graph_tensors, load_adjacency)
from models.DCRNN  import DCRNN
import pandas as pd


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="DCRNN Inference")
    parser.add_argument("--tag",          type=str, default="full_model",
                        help="Which trained model to load (matches training tag)")
    parser.add_argument("--compare",      action="store_true",
                        help="Compare all saved models side by side")
    parser.add_argument("--summary",      action="store_true",
                        help="Print summary table from saved metrics.json files")
    parser.add_argument("--predict_from", type=int, default=None,
                        help="Run inference starting from this hour index in the dataset")
    parser.add_argument("--sensor",       type=int, default=0,
                        help="Which sensor to plot in detail plots")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Inference contributions
# ---------------------------------------------------------------------------

def load_checkpoint(tag: str):

    ckpt_path = os.path.join(cfg.CHECKPOINT_DIR, tag, "best_model.pt")
    assert os.path.exists(ckpt_path), (
        f"No checkpoint found at {ckpt_path}\n"
        f"Did you train with --tag {tag}?"
    )

    ckpt = torch.load(ckpt_path, map_location=cfg.DEVICE, weights_only=False)

    # Infer model config from the tag name
    use_attention   = "no_att" not in tag and "baseline" not in tag
    use_learned_adj = "learned" in tag or "full" in tag or "both" in tag
    graph_mode      = "both" if use_learned_adj else "fixed"

    # Try to load from saved metrics for exact config
    metrics_path = os.path.join(cfg.OUTPUT_DIR, "results", tag, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            saved = json.load(f)
        use_attention   = saved.get("use_attention",   use_attention)
        use_learned_adj = saved.get("use_learned_adj", use_learned_adj)
        graph_mode      = saved.get("graph_mode",      graph_mode)

    model = DCRNN(
        num_nodes        = cfg.NUM_SENSORS,
        in_features      = cfg.IN_FEATURES,
        hidden_dim       = cfg.HIDDEN_DIM,
        out_features     = cfg.IN_FEATURES,
        output_seq_len   = cfg.OUTPUT_SEQ_LEN,
        num_layers       = cfg.NUM_LAYERS,
        K                = cfg.DIFFUSION_K,
        use_learned_adj  = use_learned_adj,
        graph_mode       = graph_mode,
    ).to(cfg.DEVICE)

    model.load_state_dict(ckpt["model_state"])
    model.eval()

    print(f"[inference] Loaded '{tag}' from epoch {ckpt['epoch']} "
          f"(val_mae={ckpt['val_mae']:.2f})")
    print(f"  attention={use_attention}, learned_adj={use_learned_adj}, "
          f"graph_mode={graph_mode}")

    return model, ckpt



# Load graph (mirrors train.py logic)
def load_graph_for_tag(tag: str, volume_raw: np.ndarray):
    """Infer which graph to use from the tag name."""
    if "correlation" in tag:
        graph_method = "correlation"
    elif "original" in tag:
        graph_method = "original"
    else:
        graph_method = "distance"


    loc_df  = pd.read_csv(cfg.LOCATIONS_PATH)
    lat_col = next(c for c in loc_df.columns if 'lat' in c.lower())
    lon_col = next(c for c in loc_df.columns if 'lon' in c.lower())
    coords  = loc_df[[lat_col, lon_col]].values.astype(np.float64)

    if graph_method == "distance":
        adj = build_distance_adjacency(
            coords, sigma=cfg.GRAPH_SIGMA,
            threshold=cfg.GRAPH_THRESHOLD, verbose=False
        )
    else:
        adj = load_adjacency(cfg.ADJ_PKL_PATH, method="original", verbose=False)

    T_f, T_b = prepare_graph_tensors(adj, device=cfg.DEVICE)
    return T_f, T_b


# ---------------------------------------------------------------------------
# Core inference functions
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference(model, loader, T_f, T_b, scaler):
    """
    Standard deterministic inference on a DataLoader.

    Returns:
        preds:   (total, T_out, N, 1) in real units
        targets: (total, T_out, N, 1) in real units
        metrics: dict {horizon: {mae, rmse, mape}}
    """
    model.eval()
    all_preds, all_targets = [], []

    for x, y in loader:
        x    = x.to(cfg.DEVICE)
        pred = model(x, T_f, T_b)
        all_preds.append(pred.cpu())
        all_targets.append(y)

    all_preds   = torch.cat(all_preds,   dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    metrics = compute_metrics(all_preds, all_targets, scaler,
                               horizons=cfg.EVAL_HORIZONS)

    # Convert to real units for output
    preds_real   = scaler.inverse_transform(all_preds).numpy()
    targets_real = scaler.inverse_transform(all_targets).numpy()

    return preds_real, targets_real, metrics


def predict_single_window(model, T_f, T_b, scaler, volume_raw, start_hour: int):
    """
    Run inference on a single 12-hour input window starting at start_hour.

    Args:
        start_hour: index into the full dataset (0–13092)

    Returns:
        pred_real:   (T_out, N) predicted volumes in vehicles/hour
        actual_real: (T_out, N) actual volumes (if available)
        input_real:  (T_in,  N) the input window shown to the model
    """
    assert start_hour + cfg.INPUT_SEQ_LEN + cfg.OUTPUT_SEQ_LEN <= len(volume_raw), \
        f"start_hour={start_hour} too close to end of dataset"

    # Extract and normalize window
    window = volume_raw[start_hour : start_hour + cfg.INPUT_SEQ_LEN + cfg.OUTPUT_SEQ_LEN]
    x_raw  = window[:cfg.INPUT_SEQ_LEN]                    # (T_in, N)
    y_raw  = window[cfg.INPUT_SEQ_LEN:]                    # (T_out, N)

    x_norm = scaler.transform(x_raw)                       # (T_in, N)
    x_t    = torch.FloatTensor(x_norm).unsqueeze(0).unsqueeze(-1).to(cfg.DEVICE)
    # x_t: (1, T_in, N, 1)

    model.eval()
    with torch.no_grad():
        pred = model(x_t, T_f, T_b)                        # (1, T_out, N, 1)

    pred_real = scaler.inverse_transform(pred.cpu()).numpy()[0, :, :, 0]   # (T_out, N)

    return pred_real, y_raw, x_raw


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_inference_results(preds, targets, tag, sensor_id=0, plot_dir=None):
    """Detailed plot for one sensor: predicted vs actual across test set."""
    if plot_dir is None:
        plot_dir = os.path.join(cfg.OUTPUT_DIR, "plots", tag)
    os.makedirs(plot_dir, exist_ok=True)

    fig, axes = plt.subplots(len(cfg.EVAL_HORIZONS), 1,
                              figsize=(16, 4 * len(cfg.EVAL_HORIZONS)),
                              sharex=True)
    if len(cfg.EVAL_HORIZONS) == 1:
        axes = [axes]

    t = np.arange(preds.shape[0])

    for i, h in enumerate(cfg.EVAL_HORIZONS):
        h_idx = h - 1
        p = preds[:,   h_idx, sensor_id, 0]
        a = targets[:, h_idx, sensor_id, 0]
        mae = np.abs(p - a).mean()

        axes[i].plot(t, a, color="steelblue",  linewidth=0.7,
                     alpha=0.9, label="Actual")
        axes[i].plot(t, p, color="darkorange", linewidth=0.7,
                     alpha=0.9, label=f"Predicted (MAE={mae:.1f})")
        axes[i].set_ylabel(f"h={h}h\n(vehicles/hr)", fontsize=9)
        axes[i].legend(fontsize=8, loc="upper right")

    axes[-1].set_xlabel("Test timestep (hours)")
    fig.suptitle(f"Inference: {tag} — Sensor {sensor_id}", fontsize=12)
    plt.tight_layout()

    path = os.path.join(plot_dir, f"inference_sensor{sensor_id}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


def plot_single_window(pred, actual, input_window, start_hour, tag,
                       sensor_id=0, plot_dir=None):
    """Plot a single prediction window — input context + forecast vs actual."""
    if plot_dir is None:
        plot_dir = os.path.join(cfg.OUTPUT_DIR, "plots", tag)
    os.makedirs(plot_dir, exist_ok=True)

    T_in  = input_window.shape[0]
    T_out = pred.shape[0]
    t_in  = np.arange(T_in)
    t_out = np.arange(T_in, T_in + T_out)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t_in,  input_window[:, sensor_id], color="gray",
            linewidth=1.5, label="Input context (12h)")
    ax.plot(t_out, actual[:, sensor_id], color="steelblue",
            linewidth=1.5, marker="o", markersize=5, label="Actual")
    ax.plot(t_out, pred[:, sensor_id], color="darkorange",
            linewidth=1.5, marker="o", markersize=5, label="Predicted")
    ax.axvline(T_in - 0.5, color="black", linestyle="--",
               linewidth=1, label="Forecast start")
    ax.set_title(f"Single Window Forecast — Sensor {sensor_id}, "
                 f"starting at hour {start_hour}")
    ax.set_xlabel("Hour offset")
    ax.set_ylabel("Volume (vehicles/hour)")
    ax.legend(fontsize=9)
    plt.tight_layout()

    path = os.path.join(plot_dir, f"window_{start_hour}_sensor{sensor_id}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary_table():
    """Print a formatted table of all saved test results."""
    results_base = os.path.join(cfg.OUTPUT_DIR, "results")
    if not os.path.exists(results_base):
        print("No results found. Train some models first.")
        return

    tags = [d for d in os.listdir(results_base)
            if os.path.exists(os.path.join(results_base, d, "metrics.json"))]

    if not tags:
        print("No metrics.json files found.")
        return

    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    horizons = cfg.EVAL_HORIZONS
    h_headers = "  ".join([f"h={h}h MAE" for h in horizons])
    print(f"  {'Model':<35} | {h_headers} | Params")
    print(f"  {'-'*35}-+-{'-'*9}-" * len(horizons) + f"-+-{'-'*8}")

    all_rows = []
    for tag in sorted(tags):
        path = os.path.join(results_base, tag, "metrics.json")
        with open(path) as f:
            r = json.load(f)
        m = r.get("test_metrics", {})
        maes = [m.get(str(h), {}).get("mae", float("nan")) for h in horizons]
        params = r.get("n_params", 0)
        all_rows.append((tag, maes, params))

    # Sort by first horizon MAE
    all_rows.sort(key=lambda x: x[1][0] if not np.isnan(x[1][0]) else 9999)

    for tag, maes, params in all_rows:
        mae_str = "  ".join([f"{m:>9.2f}" for m in maes])
        print(f"  {tag:<35} | {mae_str} | {params:>8,}")

    print("=" * 80)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()

    # Summary only — no model loading needed
    if args.summary:
        print_summary_table()
        sys.exit(0)

    # Load data (always needed)
    print("[inference] Loading data...")
    train_loader, val_loader, test_loader, scaler, volume_raw = load_dataset(
        volume_path    = cfg.VOLUME_PATH,
        input_seq_len  = cfg.INPUT_SEQ_LEN,
        output_seq_len = cfg.OUTPUT_SEQ_LEN,
        train_ratio    = cfg.TRAIN_RATIO,
        val_ratio      = cfg.VAL_RATIO,
        batch_size     = cfg.BATCH_SIZE,
        num_sensors    = cfg.NUM_SENSORS,
    )

    # Compare all saved models
    if args.compare:
        print_summary_table()

        # Find all saved checkpoints
        ckpt_base = cfg.CHECKPOINT_DIR
        if not os.path.exists(ckpt_base):
            print("No checkpoints found.")
            sys.exit(1)

        tags = [d for d in os.listdir(ckpt_base)
                if os.path.exists(os.path.join(ckpt_base, d, "best_model.pt"))]

        print(f"\nFound {len(tags)} trained models: {tags}")
        for tag in sorted(tags):
            print(f"\n--- {tag} ---")
            model, _  = load_checkpoint(tag)
            T_f, T_b  = load_graph_for_tag(tag, volume_raw)
            preds, targets, metrics = run_inference(
                model, test_loader, T_f, T_b, scaler
            )
            for h, m in metrics.items():
                print(f"  h={h}h — MAE={m['mae']:.2f}, "
                      f"RMSE={m['rmse']:.2f}, MAPE={m['mape']:.2f}%")
        sys.exit(0)

    # Single model inference
    model, ckpt = load_checkpoint(args.tag)
    T_f, T_b    = load_graph_for_tag(args.tag, volume_raw)
    plot_dir    = os.path.join(cfg.OUTPUT_DIR, "plots", args.tag)

    # Single window prediction
    if args.predict_from is not None:
        print(f"\n[inference] Predicting from hour {args.predict_from}...")
        pred, actual, inp = predict_single_window(
            model, T_f, T_b, scaler, volume_raw, args.predict_from
        )
        print(f"\nInput window (last 3 hours, sensor {args.sensor}):")
        print(f"  {inp[-3:, args.sensor].tolist()}")
        print(f"\nPredictions (next {cfg.OUTPUT_SEQ_LEN} hours, sensor {args.sensor}):")
        for h_idx, h in enumerate(cfg.EVAL_HORIZONS):
            print(f"  h={h}h: predicted={pred[h_idx, args.sensor]:.1f}, "
                  f"actual={actual[h_idx, args.sensor]:.1f}, "
                  f"error={abs(pred[h_idx, args.sensor] - actual[h_idx, args.sensor]):.1f}")
        plot_single_window(pred, actual, inp, args.predict_from,
                           args.tag, args.sensor, plot_dir)
        sys.exit(0)

    # Full test set inference
    print(f"\n[inference] Running on test set...")
    preds, targets, metrics = run_inference(model, test_loader, T_f, T_b, scaler)

    print("\n=== TEST RESULTS ===")
    for h, m in metrics.items():
        print(f"  h={h}h — MAE={m['mae']:.2f}, "
              f"RMSE={m['rmse']:.2f}, MAPE={m['mape']:.2f}%")

    plot_inference_results(preds, targets, args.tag, args.sensor, plot_dir)

    print(f"\nDone. Plots saved to {plot_dir}")