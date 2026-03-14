"""
graph_utils.py

Handles everything graph-related:
  - Rebuilding the adjacency matrix from lat/lon coordinates
  - Computing diffusion transition matrices (T_f, T_b)
  - Converting to torch sparse tensors
  - Diagnostic utilities
"""

import numpy as np
import pandas as pd
import pickle
import torch


# Haversine distance bc earth
def haversine_matrix(coords: np.ndarray) -> np.ndarray:
    lat = np.radians(coords[:, 0])
    lon = np.radians(coords[:, 1])

    dlat = lat[:, None] - lat[None, :]
    dlon = lon[:, None] - lon[None, :]

    a = (np.sin(dlat / 2) ** 2
         + np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin(dlon / 2) ** 2)

    return 6371.0 * 2.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))  # km


# ---------------------------------------------------------------------------
# Distance-based adjacency
# ---------------------------------------------------------------------------

def build_distance_adjacency(
    coords: np.ndarray,
    sigma: float = None,
    threshold: float = 0.1,
    verbose: bool = True,
) -> np.ndarray:
    """
    Build a Gaussian-kernel weighted adjacency matrix from sensor coordinates.

    Self-loops are added after thresholding (diagonal = 1).

    Args:
        coords:    (N, 2) lat/lon array
        sigma:     Gaussian bandwidth in km. Defaults to std of all pairwise
                   distances — a common heuristic that adapts to the network.
        threshold: Edges with weight below this are zeroed out.
                   Lower = denser graph. Start at 0.1 and tune.
        verbose:   Print graph statistics

    Returns:
        adj: (N, N) weighted adjacency matrix with self-loops
    """
    N = len(coords)
    dist = haversine_matrix(coords)          # (N, N) in km

    # Sigma defaults to std of upper triangle (ignore diagonal)
    mask = np.ones((N, N), dtype=bool)
    np.fill_diagonal(mask, False)
    if sigma is None:
        sigma = dist[mask].std()
        if verbose:
            print(f"[graph] Auto sigma = {sigma:.2f} km  (set GRAPH_SIGMA in config to override)")
    else:
        if verbose:
            print(f"[graph] Using sigma = {sigma:.2f} km")

    adj = np.exp(-(dist ** 2) / (sigma ** 2))
    adj[adj < threshold] = 0.0

    np.fill_diagonal(adj, 1.0)

    # Guarantee no isolated nodes — connect each to its nearest neighbour
    dist_no_self = dist.copy()
    np.fill_diagonal(dist_no_self, np.inf)
    adj_no_diag = adj.copy()
    np.fill_diagonal(adj_no_diag, 0)
    isolated = np.where(adj_no_diag.sum(axis=1) == 0)[0]
    if len(isolated) > 0:
        if verbose:
            print(f"[graph] Connecting {len(isolated)} isolated nodes to nearest neighbour")
        for i in isolated:
            j = int(np.argmin(dist_no_self[i]))
            w = float(np.exp(-(dist_no_self[i, j] ** 2) / (sigma ** 2)))
            w = max(w, 1e-4)
            adj[i, j] = w
            adj[j, i] = w

    if verbose:
        _print_graph_stats(adj, label="Distance-based")

    return adj.astype(np.float32)


# ---------------------------------------------------------------------------
# Transition matrices for diffusion convolution
# ---------------------------------------------------------------------------

def compute_transition_matrices(adj: np.ndarray):
    """
    Compute forward and backward transition matrices for diffusion convolution.

        T_f = D_out^{-1} A      (row-normalised  — outgoing traffic)
        T_b = D_in^{-1}  A^T    (col-normalised  — incoming traffic)
    """
    # Out-degree 
    d_out = adj.sum(axis=1)
    # Guard against zero-degree nodes (isolated after thresholding)
    d_out = np.where(d_out == 0, 1.0, d_out)
    T_f = adj / d_out[:, None]              

    # In-degree 
    d_in = adj.sum(axis=0)
    d_in = np.where(d_in == 0, 1.0, d_in)
    T_b = adj.T / d_in[:, None]            

    return T_f.astype(np.float32), T_b.astype(np.float32)


#utils bc issues with sparse tensors on MPS and small graph size (N=150) where dense is fine

def prepare_graph_tensors(adj: np.ndarray, device: torch.device = None):
    """
    Full pipeline: adjacency → transition matrices → dense tensors.

    N=150 is too small to benefit from sparse format, and sparse ops
    are not supported on MPS (Apple Silicon). Dense tensors run on
    MPS/CUDA/CPU equally and are faster at this scale.

    Args:
        adj:    (N, N) adjacency matrix with self-loops
        device: torch device to move tensors to

    Returns:
        T_f_t, T_b_t: dense float32 torch tensors
    """
    if device is None:
        device = torch.device("cpu")

    T_f, T_b = compute_transition_matrices(adj)
    T_f_t = torch.FloatTensor(T_f).to(device)
    T_b_t = torch.FloatTensor(T_b).to(device)
    return T_f_t, T_b_t

def _print_graph_stats(adj: np.ndarray, label: str = "Graph"):
    N = adj.shape[0]
    adj_no_diag = adj.copy()
    np.fill_diagonal(adj_no_diag, 0)

    neighbors = (adj_no_diag > 0).sum(axis=1)
    n_edges   = (adj_no_diag > 0).sum() // 2      # undirected
    sparsity  = (adj_no_diag == 0).mean()
    isolated  = (neighbors == 0).sum()

    print(f"\n[{label} stats]")
    print(f"  Nodes         : {N}")
    print(f"  Edges         : {n_edges}")
    print(f"  Avg neighbors : {neighbors.mean():.2f}")
    print(f"  Max neighbors : {neighbors.max()}")
    print(f"  Isolated nodes: {isolated}")
    print(f"  Sparsity      : {sparsity:.4f}")

def diagnose_original_adjacency(pkl_path: str):
    with open(pkl_path, "rb") as f:
        adj = pickle.load(f)

    print("[Original adjacency]")
    print(f"  Shape     : {adj.shape}")
    print(f"  Dtype     : {adj.dtype}")
    print(f"  Min / Max : {adj.min():.4f} / {adj.max():.4f}")
    print(f"  Symmetric : {np.allclose(adj, adj.T)}")

    adj_no_diag = adj.copy()
    np.fill_diagonal(adj_no_diag, 0)
    nonzero = adj_no_diag[adj_no_diag > 0]
    neighbors = (adj_no_diag > 0).sum(axis=1)

    print(f"  Non-zero edges : {len(nonzero)}")
    print(f"  Isolated nodes : {(neighbors == 0).sum()}")
    if len(nonzero):
        print(f"  Value percentiles (non-zero):")
        for p in [0, 25, 50, 75, 100]:
            print(f"    {p:3d}% → {np.percentile(nonzero, p):.4f}")


def load_adjacency(
    pkl_path: str,
    coords: np.ndarray = None,
    volume: np.ndarray = None,
    method: str = "distance",
    sigma: float = 0,
    threshold: float = 0.1,
    verbose: bool = True,
) -> np.ndarray:
    if method == "original":
        with open(pkl_path, "rb") as f:
            adj = pickle.load(f)[2].astype(np.float32)
        np.fill_diagonal(adj, 1.0)
        if verbose:
            _print_graph_stats(adj, label="Original (as-is)")
        return adj

    elif method == "distance":
        assert coords is not None, "coords required for distance method"
        return build_distance_adjacency(coords, sigma=sigma,
                                        threshold=threshold, verbose=verbose)

    else:
        raise ValueError(f"Unknown method '{method}'. "
                         "Choose 'distance', 'correlation', or 'original'.")