"""
test_graph.py

Run this first to verify your adjacency rebuild is working correctly
before touching any model code.

Usage:
    python test_graph.py

Expected output:
    - Original adjacency stats (broken — 133 isolated nodes)
    - Rebuilt distance adjacency stats (should show 0 isolated nodes)
    - Rebuilt correlation adjacency stats
    - Transition matrix sanity checks
    - A saved heatmap plot in outputs/
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from src.graph_utils import (
    diagnose_original_adjacency,
    build_distance_adjacency,
    build_correlation_adjacency,
    compute_transition_matrices,
    prepare_graph_tensors,
    haversine_matrix,
)
import config as cfg


def test_original():
    print("=" * 60)
    print("ORIGINAL ADJACENCY (from pkl)")
    print("=" * 60)
    diagnose_original_adjacency(cfg.ADJ_PKL_PATH)


def load_data():
    # Volume: shape (T, N)
    df = pd.read_csv(cfg.VOLUME_PATH, header=None)
    volume = df.values.astype(np.float32)
    if volume.shape[1] != cfg.NUM_SENSORS:
        volume = volume.T  # ensure (T, N)
    print(f"\nVolume shape: {volume.shape}  (T={volume.shape[0]}, N={volume.shape[1]})")

    # Coordinates: shape (N, 2)
    loc_df = pd.read_csv(cfg.LOCATIONS_PATH)
    print(f"Location columns: {loc_df.columns.tolist()}")

    # Auto-detect lat/lon columns
    lat_col = next(c for c in loc_df.columns if 'lat' in c.lower())
    lon_col = next(c for c in loc_df.columns if 'lon' in c.lower())
    coords = loc_df[[lat_col, lon_col]].values.astype(np.float64)
    print(f"Coords shape: {coords.shape}")
    print(f"Lat range: {coords[:,0].min():.4f} – {coords[:,0].max():.4f}")
    print(f"Lon range: {coords[:,1].min():.4f} – {coords[:,1].max():.4f}")

    return volume, coords


def test_distance_adjacency(coords):
    print("\n" + "=" * 60)
    print("DISTANCE-BASED ADJACENCY")
    print("=" * 60)

    # First look at the raw distance distribution to inform threshold choice
    dist = haversine_matrix(coords)
    np.fill_diagonal(dist, np.inf)
    flat = dist[dist != np.inf]
    print(f"\nPairwise distance stats (km):")
    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        print(f"  {p:3d}% → {np.percentile(flat, p):.2f} km")

    # Try a range of thresholds so you can see the tradeoff
    print("\nThreshold sweep (lower threshold = more edges):")
    print(f"  {'threshold':>10} | {'avg_neighbors':>13} | {'isolated':>8} | {'edges':>6}")
    print(f"  {'-'*10}-+-{'-'*13}-+-{'-'*8}-+-{'-'*6}")
    for t in [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.93, 0.95, 0.99]:
        adj = build_distance_adjacency(coords, sigma=cfg.GRAPH_SIGMA,  threshold=t, verbose=False)
        adj_nd = adj.copy()
        np.fill_diagonal(adj_nd, 0)
        nb = (adj_nd > 0).sum(axis=1)
        edges = (adj_nd > 0).sum() // 2
        print(f"  {t:>10.2f} | {nb.mean():>13.2f} | {(nb==0).sum():>8} | {edges:>6}")

    # Build with the config threshold
    print(f"\nBuilding with threshold={cfg.GRAPH_THRESHOLD}, sigma={cfg.GRAPH_SIGMA}")
    adj = build_distance_adjacency(
        coords,
        sigma=cfg.GRAPH_SIGMA,
        threshold=cfg.GRAPH_THRESHOLD,
        verbose=True,
    )
    return adj


def test_correlation_adjacency(volume):
    print("\n" + "=" * 60)
    print("CORRELATION-BASED ADJACENCY")
    print("=" * 60)

    print("\nCorrelation threshold sweep:")
    print(f"  {'threshold':>10} | {'avg_neighbors':>13} | {'isolated':>8} | {'edges':>6}")
    print(f"  {'-'*10}-+-{'-'*13}-+-{'-'*8}-+-{'-'*6}")
    for t in [0.5, 0.6, 0.7, 0.8, 0.9, 0.93, 0.95, 0.99]:
        adj = build_correlation_adjacency(volume, threshold=t, verbose=False)
        adj_nd = adj.copy()
        np.fill_diagonal(adj_nd, 0)
        nb = (adj_nd > 0).sum(axis=1)
        edges = (adj_nd > 0).sum() // 2
        print(f"  {t:>10.2f} | {nb.mean():>13.2f} | {(nb==0).sum():>8} | {edges:>6}")

    adj = build_correlation_adjacency(
        volume, threshold=cfg.CORR_THRESHOLD, verbose=True
    )
    return adj


def test_transition_matrices(adj, label):
    print(f"\n[Transition matrix check — {label}]")
    T_f, T_b = compute_transition_matrices(adj)

    # Row sums of transition matrices should all be exactly 1.0
    row_sums_f = T_f.sum(axis=1)
    row_sums_b = T_b.sum(axis=1)
    print(f"  T_f row sum: min={row_sums_f.min():.6f}, max={row_sums_f.max():.6f}  (should be 1.0)")
    print(f"  T_b row sum: min={row_sums_b.min():.6f}, max={row_sums_b.max():.6f}  (should be 1.0)")
    assert np.allclose(row_sums_f, 1.0, atol=1e-5), "T_f rows don't sum to 1!"
    assert np.allclose(row_sums_b, 1.0, atol=1e-5), "T_b rows don't sum to 1!"
    print("  ✓ Transition matrices are valid")
    return T_f, T_b


def plot_adjacencies(adj_dist, adj_corr, coords):
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # --- Heatmaps ---
    for ax, adj, title in zip(
        axes[:2],
        [adj_dist, adj_corr],
        ["Distance-based adjacency", "Correlation-based adjacency"],
    ):
        im = ax.imshow(adj, cmap="YlOrRd", aspect="auto",
                       norm=mcolors.Normalize(vmin=0, vmax=1))
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Sensor index")
        ax.set_ylabel("Sensor index")
        plt.colorbar(im, ax=ax, fraction=0.046)

    # --- Sensor map coloured by degree ---
    adj_nd = adj_dist.copy()
    np.fill_diagonal(adj_nd, 0)
    degree = (adj_nd > 0).sum(axis=1)

    sc = axes[2].scatter(
        coords[:, 1], coords[:, 0],
        c=degree, cmap="plasma", s=40, edgecolors="k", linewidths=0.3
    )
    plt.colorbar(sc, ax=axes[2], label="Degree (# neighbors)")
    axes[2].set_title("Sensor locations — coloured by degree", fontsize=12)
    axes[2].set_xlabel("Longitude")
    axes[2].set_ylabel("Latitude")

    plt.tight_layout()
    out_path = os.path.join(cfg.OUTPUT_DIR, "adjacency_comparison.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved → {out_path}")
    plt.close()


def test_torch_tensors(adj):
    import torch
    T_f_t, T_b_t = prepare_graph_tensors(adj, device=cfg.DEVICE)
    print(f"\n[Torch tensor check]")
    print(f"  Device : {cfg.DEVICE}")
    print(f"  T_f    : {T_f_t.shape}, sparse={T_f_t.is_sparse}")
    print(f"  T_b    : {T_b_t.shape}, sparse={T_b_t.is_sparse}")

    # Quick matmul test — multiply a random signal through T_f
    x = torch.randn(cfg.NUM_SENSORS, 8).to(cfg.DEVICE)   # (N, features)
    out = torch.sparse.mm(T_f_t, x)
    print(f"  T_f @ x: {x.shape} → {out.shape}  ✓")


if __name__ == "__main__":
    test_original()

    volume, coords = load_data()

    adj_dist = test_distance_adjacency(coords)
    adj_corr = test_correlation_adjacency(volume)

    test_transition_matrices(adj_dist, "distance")
    test_transition_matrices(adj_corr, "correlation")

    plot_adjacencies(adj_dist, adj_corr, coords)
    test_torch_tensors(adj_dist)

    print("\n" + "=" * 60)
    print("ALL GRAPH TESTS PASSED")
    print("Next step: adjust GRAPH_THRESHOLD in config.py until")
    print("avg_neighbors is 3-8 with 0 isolated nodes, then run")
    print("the data_loader tests.")
    print("=" * 60)