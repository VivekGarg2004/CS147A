"""
train.py

Full training pipeline for DCRNN with temporal attention.

Usage:
    # Train with distance-based graph (baseline)
    python train.py --graph distance --tag baseline

    # Train with learned adjacency + attention (your contribution)
    python train.py --graph both --tag full_model

    # Train all three configurations for ablation study
    python train.py --run_ablation

Each run saves:
    outputs/checkpoints/<tag>/best_model.pt   — best val MAE checkpoint
    outputs/results/<tag>/metrics.json        — test metrics per horizon
    outputs/results/<tag>/predictions.pt      — test predictions + targets
    outputs/plots/<tag>/                      — training curves + prediction plots
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker

sys.path.insert(0, os.path.dirname(__file__))

import config as cfg
from src.data_loader  import load_dataset, compute_metrics, masked_mae, masked_rmse
from src.graph_utils  import (load_adjacency, prepare_graph_tensors,
                               build_distance_adjacency, build_correlation_adjacency)
from src.dcrnn_model  import DCRNN
import pandas as pd


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Train DCRNN")
    parser.add_argument("--graph", type=str, default="distance",
                        choices=["original", "distance", "correlation"],
                        help="Which pre-built graph to use as fixed adjacency")
    parser.add_argument("--graph_mode", type=str, default="both",
                        choices=["fixed", "learned", "both"],
                        help="How to combine fixed and learned adjacency")
    parser.add_argument("--no_attention", action="store_true",
                        help="Disable temporal attention (run baseline DCRNN)")
    parser.add_argument("--no_learned_adj", action="store_true",
                        help="Disable learned adjacency")
    parser.add_argument("--tag", type=str, default=None,
                        help="Run tag for saving results (auto-generated if not set)")
    parser.add_argument("--epochs", type=int, default=cfg.NUM_EPOCHS,
                        help="Number of training epochs")
    parser.add_argument("--run_ablation", action="store_true",
                        help="Run all ablation configurations sequentially")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Graph loading
# ---------------------------------------------------------------------------

def load_graph(graph_method: str, volume_raw: np.ndarray):
    """Load and prepare graph transition matrices."""
    loc_df  = pd.read_csv(cfg.LOCATIONS_PATH)
    lat_col = next(c for c in loc_df.columns if 'lat' in c.lower())
    lon_col = next(c for c in loc_df.columns if 'lon' in c.lower())
    coords  = loc_df[[lat_col, lon_col]].values.astype(np.float64)

    if graph_method == "original":
        adj = load_adjacency(cfg.ADJ_PKL_PATH, method="original", verbose=True)
    elif graph_method == "distance":
        adj = build_distance_adjacency(
            coords,
            sigma     = cfg.GRAPH_SIGMA,
            threshold = cfg.GRAPH_THRESHOLD,
            verbose   = True,
        )
    elif graph_method == "correlation":
        adj = build_correlation_adjacency(
            volume_raw,
            threshold = cfg.CORR_THRESHOLD,
            verbose   = True,
        )
    else:
        raise ValueError(f"Unknown graph method: {graph_method}")

    T_f, T_b = prepare_graph_tensors(adj, device=cfg.DEVICE)
    return T_f, T_b, adj


# ---------------------------------------------------------------------------
# Scheduled sampling schedule
# ---------------------------------------------------------------------------

def get_teacher_forcing_prob(step: int) -> float:
    """
    Linear decay from SCHEDULED_SAMPLING_START to SCHEDULED_SAMPLING_MAX
    over SCHEDULED_SAMPLING_STEPS steps.

    Early training: low probability (mostly teacher forcing / ground truth)
    Later training: higher probability (model uses its own predictions more)

    This is the opposite of the usual convention — here the probability
    is the chance of using the MODEL'S OWN prediction, not ground truth.
    """
    progress = min(step / cfg.SCHEDULED_SAMPLING_STEPS, 1.0)
    return (cfg.SCHEDULED_SAMPLING_START
            + progress * (cfg.SCHEDULED_SAMPLING_MAX - cfg.SCHEDULED_SAMPLING_START))


# ---------------------------------------------------------------------------
# Training epoch
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, T_f, T_b, step, scaler):
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch_idx, (x, y) in enumerate(loader):
        x = x.to(cfg.DEVICE)    # (B, T_in, N, 1)
        y = y.to(cfg.DEVICE)    # (B, T_out, N, 1)

        tf_prob = get_teacher_forcing_prob(step)

        # Forward pass
        pred = model(x, T_f, T_b, targets=y, teacher_forcing_prob=tf_prob)

        # Masked MAE loss — ignores zero-volume entries
        loss = masked_mae(pred, y)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.CLIP_GRAD_NORM)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1
        step += 1

        if batch_idx % cfg.LOG_INTERVAL == 0:
            print(f"    batch {batch_idx:4d}/{len(loader)} — "
                  f"loss(norm)={loss.item():.4f}, tf_prob={tf_prob:.3f}")

    # Return mean normalized loss — val/test metrics are in real units
    # so train loss is not directly comparable (train is Z-score scale ~0-1,
    # val MAE is in vehicles/hour ~200-700)
    return total_loss / num_batches, step


# ---------------------------------------------------------------------------
# Validation / test epoch
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader, T_f, T_b, scaler, split="val"):
    model.eval()
    all_preds   = []
    all_targets = []

    for x, y in loader:
        x    = x.to(cfg.DEVICE)
        y    = y.to(cfg.DEVICE)
        pred = model(x, T_f, T_b)      # no teacher forcing at eval
        all_preds.append(pred.cpu())
        all_targets.append(y.cpu())

    all_preds   = torch.cat(all_preds,   dim=0)   # (total, T_out, N, 1)
    all_targets = torch.cat(all_targets, dim=0)

    # Metrics at each horizon in real units
    metrics = compute_metrics(all_preds, all_targets, scaler,
                               horizons=cfg.EVAL_HORIZONS)

    # Summary MAE across all horizons (used for early stopping)
    mean_mae = np.mean([m["mae"] for m in metrics.values()])

    return metrics, mean_mae, all_preds, all_targets


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_training_curves(history: dict, plot_dir: str):
    os.makedirs(plot_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    train_loss = np.asarray(history.get("train_loss", []), dtype=np.float64)
    val_loss = np.asarray(history.get("val_loss", []), dtype=np.float64)

    # Plot normalized train loss on its own axis so it never appears flattened.
    axes[0].plot(train_loss, label="Train loss (normalized)",
                 color="steelblue", linewidth=1.6, marker="o", markersize=3)
    axes[0].set_title("Train Loss (Normalized)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss (z-score space)")
    axes[0].yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))
    if train_loss.size > 0 and np.isfinite(train_loss).any():
        y_min = np.nanmin(train_loss)
        y_max = np.nanmax(train_loss)
        pad = max((y_max - y_min) * 0.15, 1e-3)
        axes[0].set_ylim(y_min - pad, y_max + pad)
    axes[0].legend(loc="upper right")

    # Horizon metrics can have int or string keys depending on source/serialization.
    def _epoch_mae_for_h(epoch_metrics, h):
        if h in epoch_metrics:
            return epoch_metrics[h].get("mae", np.nan)
        h_str = str(h)
        if h_str in epoch_metrics:
            return epoch_metrics[h_str].get("mae", np.nan)
        return np.nan

    plotted_any = False
    if val_loss.size > 0:
        axes[1].plot(val_loss, color="black", linestyle="--", linewidth=1.3,
                     label="Mean val MAE")
        plotted_any = True
    for h in cfg.EVAL_HORIZONS:
        val_maes = np.asarray([
            _epoch_mae_for_h(e, h) for e in history.get("val_metrics", [])
        ], dtype=np.float64)
        if val_maes.size == 0 or not np.isfinite(val_maes).any():
            continue
        axes[1].plot(val_maes, linewidth=1.5, marker="o", markersize=3,
                     label=f"h={h}h")
        plotted_any = True

    axes[1].set_title("Val MAE by Horizon")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MAE (vehicles/hour)")
    if plotted_any:
        axes[1].legend()
    else:
        axes[1].text(0.5, 0.5, "No valid horizon MAE values found",
                     transform=axes[1].transAxes, ha="center", va="center")

    axes[2].plot(history["lr"], color="green")
    axes[2].set_title("Learning Rate")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("LR")
    axes[2].set_yscale("log")

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "training_curves.png"), dpi=150,
                bbox_inches="tight")
    plt.close()
    print(f"  Saved training curves → {plot_dir}/training_curves.png")


def plot_predictions(preds, targets, scaler, plot_dir: str, n_sensors=4):
    """Plot predicted vs actual for a few sensors across the test set."""
    os.makedirs(plot_dir, exist_ok=True)

    # Invert normalization
    pred_real   = scaler.inverse_transform(preds).numpy()      # (total, T_out, N, 1)
    target_real = scaler.inverse_transform(targets).numpy()

    # Use first output step (h=1) for simplicity
    pred_h1   = pred_real[:,   0, :, 0]    # (total, N)
    target_h1 = target_real[:, 0, :, 0]

    sensor_ids = np.linspace(0, pred_h1.shape[1] - 1, n_sensors, dtype=int)

    fig, axes = plt.subplots(n_sensors, 1, figsize=(16, 3 * n_sensors), sharex=True)
    if n_sensors == 1:
        axes = [axes]

    t = np.arange(len(pred_h1))
    for i, s in enumerate(sensor_ids):
        axes[i].plot(t, target_h1[:, s], label="Actual",
                     color="steelblue", linewidth=0.8, alpha=0.9)
        axes[i].plot(t, pred_h1[:, s],   label="Predicted",
                     color="darkorange", linewidth=0.8, alpha=0.9)
        axes[i].set_ylabel(f"Sensor {s}\n(vehicles/hr)", fontsize=9)
        axes[i].legend(fontsize=8, loc="upper right")

    axes[-1].set_xlabel("Test timestep")
    fig.suptitle("Predicted vs Actual Volume (h=1h ahead)", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "predictions.png"), dpi=150,
                bbox_inches="tight")
    plt.close()
    print(f"  Saved prediction plot → {plot_dir}/predictions.png")


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(
    graph_method: str  = "distance",
    graph_mode: str    = "both",
    use_attention: bool = True,
    use_learned_adj: bool = True,
    num_epochs: int    = None,
    tag: str           = None,
):
    if num_epochs is None:
        num_epochs = cfg.NUM_EPOCHS
    if tag is None:
        tag = f"{graph_method}_{graph_mode}_att{int(use_attention)}_ladj{int(use_learned_adj)}"

    print("\n" + "=" * 70)
    print(f"RUN: {tag}")
    print(f"  graph={graph_method}, mode={graph_mode}, "
          f"attention={use_attention}, learned_adj={use_learned_adj}")
    print("=" * 70)

    # --- Directories ---
    ckpt_dir  = os.path.join(cfg.CHECKPOINT_DIR, tag)
    plot_dir  = os.path.join(cfg.OUTPUT_DIR, "plots",   tag)
    res_dir   = os.path.join(cfg.OUTPUT_DIR, "results", tag)
    for d in [ckpt_dir, plot_dir, res_dir]:
        os.makedirs(d, exist_ok=True)

    # --- Seed ---
    torch.manual_seed(cfg.SEED)
    np.random.seed(cfg.SEED)

    # --- Data ---
    print("\n[1/5] Loading data...")
    train_loader, val_loader, test_loader, scaler, volume_raw = load_dataset(
        volume_path    = cfg.VOLUME_PATH,
        input_seq_len  = cfg.INPUT_SEQ_LEN,
        output_seq_len = cfg.OUTPUT_SEQ_LEN,
        train_ratio    = cfg.TRAIN_RATIO,
        val_ratio      = cfg.VAL_RATIO,
        batch_size     = cfg.BATCH_SIZE,
        num_sensors    = cfg.NUM_SENSORS,
    )

    # --- Graph ---
    print("\n[2/5] Building graph...")
    T_f, T_b, adj = load_graph(graph_method, volume_raw)

    # --- Model ---
    print("\n[3/5] Building model...")
    model = DCRNN(
        num_nodes        = cfg.NUM_SENSORS,
        in_features      = cfg.IN_FEATURES,
        hidden_dim       = cfg.HIDDEN_DIM,
        out_features     = cfg.IN_FEATURES,
        output_seq_len   = cfg.OUTPUT_SEQ_LEN,
        num_layers       = cfg.NUM_LAYERS,
        K                = cfg.DIFFUSION_K,
        use_attention    = use_attention,
        attention_heads  = cfg.ATTENTION_HEADS,
        use_learned_adj  = use_learned_adj,
        graph_mode       = graph_mode if use_learned_adj else "fixed",
    ).to(cfg.DEVICE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr           = cfg.LEARNING_RATE,
        weight_decay = cfg.WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode     = "min",
        patience = cfg.LR_PATIENCE,
        factor   = cfg.LR_FACTOR,
        min_lr   = cfg.LR_MIN,
    )

    # --- Training loop ---
    print(f"\n[4/5] Training for {num_epochs} epochs...")
    history = {
        "train_loss":  [],
        "val_loss":    [],
        "val_metrics": [],
        "lr":          [],
    }

    best_val_mae  = float("inf")
    best_ckpt     = os.path.join(ckpt_dir, "best_model.pt")
    step          = 0
    start_time    = time.time()

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()

        # Train
        train_loss, step = train_epoch(
            model, train_loader, optimizer, T_f, T_b, step, scaler
        )

        # Validate
        val_metrics, val_mae, _, _ = evaluate(
            model, val_loader, T_f, T_b, scaler, split="val"
        )

        # Scheduler step
        scheduler.step(val_mae)
        current_lr = optimizer.param_groups[0]["lr"]

        # Save history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_mae)
        history["val_metrics"].append(val_metrics)
        history["lr"].append(current_lr)

        # Checkpoint best model
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "opt_state":   optimizer.state_dict(),
                "val_mae":     val_mae,
                "val_metrics": val_metrics,
            }, best_ckpt)

        epoch_time = time.time() - epoch_start
        print(f"\nEpoch {epoch:3d}/{num_epochs} ({epoch_time:.1f}s) — "
              f"train_mae={train_loss:.2f}, val_mae={val_mae:.2f}, "
              f"best={best_val_mae:.2f}, lr={current_lr:.2e}")
        print("  Val metrics by horizon:")
        for h, m in val_metrics.items():
            print(f"    h={h}h — MAE={m['mae']:.2f}, "
                  f"RMSE={m['rmse']:.2f}, MAPE={m['mape']:.2f}%")

    total_time = time.time() - start_time
    print(f"\nTraining complete in {total_time/60:.1f} minutes")

    # --- Test evaluation ---
    print("\n[5/5] Evaluating on test set...")

    # Load best checkpoint
    ckpt = torch.load(best_ckpt, map_location=cfg.DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    print(f"  Loaded best checkpoint from epoch {ckpt['epoch']} "
          f"(val_mae={ckpt['val_mae']:.4f})")

    test_metrics, test_mae, test_preds, test_targets = evaluate(
        model, test_loader, T_f, T_b, scaler, split="test"
    )

    print("\n  === TEST RESULTS ===")
    for h, m in test_metrics.items():
        print(f"  h={h}h — MAE={m['mae']:.2f}, "
              f"RMSE={m['rmse']:.2f}, MAPE={m['mape']:.2f}%")

    # Save metrics
    results = {
        "tag":          tag,
        "graph_method": graph_method,
        "graph_mode":   graph_mode,
        "use_attention":   use_attention,
        "use_learned_adj": use_learned_adj,
        "n_params":     n_params,
        "best_epoch":   ckpt["epoch"],
        "best_val_mae": best_val_mae,
        "test_metrics": {str(k): v for k, v in test_metrics.items()},
        "training_time_min": total_time / 60,
    }
    with open(os.path.join(res_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    torch.save({"preds": test_preds, "targets": test_targets},
               os.path.join(res_dir, "predictions.pt"))

    # --- Plots ---
    print("\n  Generating plots...")
    plot_training_curves(history, plot_dir)
    plot_predictions(test_preds, test_targets, scaler, plot_dir)

    print(f"\n  Results saved to {res_dir}")
    print(f"  Plots saved to   {plot_dir}")

    return results


# ---------------------------------------------------------------------------
# Ablation study
# ---------------------------------------------------------------------------

ABLATION_CONFIGS = [
    # (graph_method, graph_mode, use_attention, use_learned_adj, tag)
    ("distance",    "fixed",   False, False, "baseline_dcrnn_distance"),
    ("correlation", "fixed",   False, False, "baseline_dcrnn_correlation"),
    ("original",    "fixed",   False, False, "baseline_dcrnn_original"),
    ("distance",    "fixed",   True,  False, "dcrnn_attention"),
    ("distance",    "learned", False, True,  "dcrnn_learned_adj"),
    ("distance",    "both",    True,  True,  "full_model"),
]


def run_ablation(num_epochs):
    print("\n" + "=" * 70)
    print("ABLATION STUDY — running all configurations")
    print("=" * 70)

    all_results = []
    for graph, mode, att, ladj, tag in ABLATION_CONFIGS:
        results = train(
            graph_method    = graph,
            graph_mode      = mode,
            use_attention   = att,
            use_learned_adj = ladj,
            num_epochs      = num_epochs,
            tag             = tag,
        )
        all_results.append(results)

    # Print summary table
    print("\n" + "=" * 70)
    print("ABLATION RESULTS SUMMARY")
    print("=" * 70)
    print(f"  {'Tag':<35} | {'h=1 MAE':>8} | {'h=2 MAE':>8} | {'h=3 MAE':>8}")
    print(f"  {'-'*35}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
    for r in all_results:
        m = r["test_metrics"]
        maes = [m.get(str(h), {}).get("mae", float("nan"))
                for h in cfg.EVAL_HORIZONS]
        print(f"  {r['tag']:<35} | "
              f"{maes[0]:>8.2f} | {maes[1]:>8.2f} | {maes[2]:>8.2f}")

    # Save summary
    summary_path = os.path.join(cfg.OUTPUT_DIR, "ablation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Full summary saved → {summary_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()

    if args.run_ablation:
        run_ablation(num_epochs=args.epochs)
    else:
        train(
            graph_method    = args.graph,
            graph_mode      = args.graph_mode,
            use_attention   = not args.no_attention,
            use_learned_adj = not args.no_learned_adj,
            num_epochs      = args.epochs,
            tag             = args.tag,
        )