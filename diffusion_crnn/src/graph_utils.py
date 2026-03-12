"""
graph_utils.py

Handles everything graph-related:
  - Rebuilding the adjacency matrix from lat/lon coordinates
  - Computing diffusion transition matrices (T_f, T_b)
  - Converting to torch sparse tensors
  - Diagnostic utilities

Two approaches are supported:
  1. Distance-based adjacency  (Gaussian kernel on haversine distance)
  2. Correlation-based adjacency (Pearson correlation on time series)
"""

import numpy as np
import pandas as pd
import pickle
import torch


# ---------------------------------------------------------------------------
# Haversine distance
# ---------------------------------------------------------------------------

def haversine_matrix(coords: np.ndarray) -> np.ndarray:
    """
    Compute pairwise haversine distances between all sensors.

    Args:
        coords: (N, 2) array of [latitude, longitude] in decimal degrees

    Returns:
        dist: (N, N) matrix of distances in kilometres
    """
    lat = np.radians(coords[:, 0])
    lon = np.radians(coords[:, 1])

    dlat = lat[:, None] - lat[None, :]
    dlon = lon[:, None] - lon[None, :]

    a = (np.sin(dlat / 2) ** 2
         + np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin(dlon / 2) ** 2)

    return 6371.0 * 2.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))  # km


# ---------------------------------------------------------------------------
# Approach 1 — Distance-based adjacency
# ---------------------------------------------------------------------------

def build_distance_adjacency(
    coords: np.ndarray,
    sigma: float = None,
    threshold: float = 0.1,
    verbose: bool = True,
) -> np.ndarray:
    """
    Build a Gaussian-kernel weighted adjacency matrix from sensor coordinates.

    The weight between sensors i and j is:
        w_ij = exp(-dist(i,j)^2 / sigma^2)   if w_ij >= threshold
        w_ij = 0                               otherwise

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

    # Gaussian kernel
    adj = np.exp(-(dist ** 2) / (sigma ** 2))

    # Zero out sub-threshold edges
    adj[adj < threshold] = 0.0

    # Force diagonal to exactly 1 (self-loops)
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
# Approach 2 — Correlation-based adjacency
# ---------------------------------------------------------------------------

def build_correlation_adjacency(
    volume: np.ndarray,
    threshold: float = 0.7,
    verbose: bool = True,
) -> np.ndarray:
    """
    Build an adjacency matrix from Pearson correlations between sensor
    time series. Sensors with highly correlated volume patterns get an edge.

    This captures *functional* similarity — sensors that behave alike —
    rather than physical proximity.  Useful when the road network geometry
    is unavailable or unreliable.

    Args:
        volume:    (T, N) array of traffic volumes
        threshold: Correlations below this are zeroed out.
                   Start at 0.7; lower = denser graph.
        verbose:   Print graph statistics

    Returns:
        adj: (N, N) correlation-weighted adjacency matrix with self-loops
    """
    # Ensure volume is (T, N) — sensors are columns
    if volume.shape[0] < volume.shape[1]:
        # Looks like (N, T) — transpose to (T, N)
        volume = volume.T
    T, N = volume.shape
    assert N <= 500, (
        f"Got N={N} which looks like timesteps not sensors. "
        f"volume should be (T, N), got {volume.shape}"
    )

    # Remove global trend before correlating — raw traffic volumes are all
    # correlated with each other (rush hour affects every sensor) which makes
    # raw Pearson correlation useless as a graph signal. Correlating residuals
    # captures which sensors deviate from the citywide average *together*,
    # which is a much more meaningful local relationship.
    global_mean = volume.mean(axis=1, keepdims=True)   # (T, 1) mean across sensors
    residuals   = volume - global_mean                  # (T, N) local deviations

    # np.corrcoef expects (N, T) — each row is one sensor's time series
    corr = np.corrcoef(residuals.T)                     # (N, N)
    assert corr.shape == (N, N), f"Expected ({N},{N}), got {corr.shape}"

    # Only keep positive correlations above threshold
    adj = np.where(corr >= threshold, corr, 0.0)

    # Self-loops
    np.fill_diagonal(adj, 1.0)

    if verbose:
        _print_graph_stats(adj, label="Correlation-based")

    return adj.astype(np.float32)


# ---------------------------------------------------------------------------
# Transition matrices for diffusion convolution
# ---------------------------------------------------------------------------

def compute_transition_matrices(adj: np.ndarray):
    """
    Compute the forward and backward transition matrices used in DCRNN's
    diffusion convolution.

        T_f = D_out^{-1} A      (row-normalised  — outgoing traffic)
        T_b = D_in^{-1}  A^T    (col-normalised  — incoming traffic)

    For an undirected graph T_b = T_f^T, but we compute both explicitly
    so the code works for directed graphs too.

    Args:
        adj: (N, N) adjacency matrix WITH self-loops

    Returns:
        T_f: (N, N) forward  transition matrix
        T_b: (N, N) backward transition matrix
    """
    # Out-degree (row sum) for forward
    d_out = adj.sum(axis=1)
    # Guard against zero-degree nodes (isolated after thresholding)
    d_out = np.where(d_out == 0, 1.0, d_out)
    T_f = adj / d_out[:, None]              # row normalise

    # In-degree (col sum) for backward
    d_in = adj.sum(axis=0)
    d_in = np.where(d_in == 0, 1.0, d_in)
    T_b = adj.T / d_in[:, None]            # row normalise of transpose

    return T_f.astype(np.float32), T_b.astype(np.float32)


# ---------------------------------------------------------------------------
# Torch conversion
# ---------------------------------------------------------------------------

def to_sparse_tensor(matrix: np.ndarray) -> torch.Tensor:
    """
    Convert a dense numpy adjacency / transition matrix to a torch sparse
    COO tensor.  Sparse format matters for large graphs; at N=150 dense
    is fine too, but we keep sparse for consistency with larger datasets.

    Args:
        matrix: (N, N) numpy array

    Returns:
        sparse_tensor: torch.sparse_coo_tensor on CPU
    """
    matrix_t = torch.FloatTensor(matrix)
    indices = matrix_t.nonzero(as_tuple=False).T          # (2, nnz)
    values = matrix_t[indices[0], indices[1]]              # (nnz,)
    return torch.sparse_coo_tensor(indices, values, matrix_t.shape)


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


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

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
    """
    Quick diagnostic on the raw pkl adjacency to understand what's in it
    before deciding whether to use or rebuild it.
    """
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


# ---------------------------------------------------------------------------
# Convenience loader
# ---------------------------------------------------------------------------

def load_adjacency(
    pkl_path: str,
    coords: np.ndarray = None,
    volume: np.ndarray = None,
    method: str = "distance",
    sigma: float = None,
    threshold: float = 0.1,
    verbose: bool = True,
) -> np.ndarray:
    """
    Master loader. Chooses which adjacency to build based on `method`.

    Args:
        pkl_path:  Path to original pkl (used for diagnostics)
        coords:    (N, 2) lat/lon — required if method='distance'
        volume:    (T, N) volumes  — required if method='correlation'
        method:    'distance' | 'correlation' | 'original'
        sigma:     Gaussian bandwidth (distance method only)
        threshold: Edge threshold
        verbose:   Print stats

    Returns:
        adj: (N, N) float32 adjacency with self-loops
    """
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

    elif method == "correlation":
        assert volume is not None, "volume array required for correlation method"
        return build_correlation_adjacency(volume, threshold=threshold,
                                           verbose=verbose)

    else:
        raise ValueError(f"Unknown method '{method}'. "
                         "Choose 'distance', 'correlation', or 'original'.")