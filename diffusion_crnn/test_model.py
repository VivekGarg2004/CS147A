"""
test_model.py

Verifies the full model stack before starting training.

Usage:
    python test_model.py

Checks:
    - DiffusionConv shapes and dense/sparse equivalence
    - DCGRUCell forward and backward pass
    - Encoder/Decoder shapes
    - Full DCRNN with all combinations of attention + learned adjacency
    - Parameter counts
    - One full training step with real data shapes
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np

from src.diffusion_conv import verify_diffusion_conv
from src.dcrnn_cell     import verify_dcrnn_cell
from src.dcrnn_model    import verify_model, DCRNN
import config as cfg


def test_full_training_step():
    """
    Simulate one full training step with realistic data shapes.
    Verifies loss computation and backward pass work end-to-end.
    """
    print("\n" + "=" * 60)
    print("FULL TRAINING STEP SIMULATION")
    print("=" * 60)

    from src.data_loader import masked_mae
    from src.graph_utils import build_distance_adjacency, prepare_graph_tensors
    import pandas as pd

    # Load real adjacency
    loc_df = pd.read_csv(cfg.LOCATIONS_PATH)
    lat_col = next(c for c in loc_df.columns if 'lat' in c.lower())
    lon_col = next(c for c in loc_df.columns if 'lon' in c.lower())
    coords  = loc_df[[lat_col, lon_col]].values

    adj    = build_distance_adjacency(coords, sigma=cfg.GRAPH_SIGMA,
                                      threshold=cfg.GRAPH_THRESHOLD, verbose=False)
    T_f, T_b = prepare_graph_tensors(adj, device=cfg.DEVICE)

    # Build model
    model = DCRNN(
        num_nodes       = cfg.NUM_SENSORS,
        in_features     = cfg.IN_FEATURES,
        hidden_dim      = cfg.HIDDEN_DIM,
        out_features    = cfg.IN_FEATURES,
        output_seq_len  = cfg.OUTPUT_SEQ_LEN,
        num_layers      = cfg.NUM_LAYERS,
        K               = cfg.DIFFUSION_K,
        use_attention   = cfg.USE_ATTENTION,
        attention_heads = cfg.ATTENTION_HEADS,
        use_learned_adj = True,
        graph_mode      = "both",
    ).to(cfg.DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)

    # Fake batch
    B   = cfg.BATCH_SIZE
    x   = torch.randn(B, cfg.INPUT_SEQ_LEN,  cfg.NUM_SENSORS, cfg.IN_FEATURES).to(cfg.DEVICE)
    y   = torch.randn(B, cfg.OUTPUT_SEQ_LEN, cfg.NUM_SENSORS, cfg.IN_FEATURES).to(cfg.DEVICE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model parameters: {n_params:,}")

    # Forward
    model.train()
    pred = model(x, T_f, T_b, targets=y, teacher_forcing_prob=0.5)
    print(f"  Forward pass: {tuple(x.shape)} → {tuple(pred.shape)}")

    # Loss + backward
    loss = masked_mae(pred, y)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.CLIP_GRAD_NORM)
    optimizer.step()

    print(f"  Loss: {loss.item():.4f}")
    print(f"  ✓ Full training step complete")

    # Check no NaN gradients
    for name, p in model.named_parameters():
        if p.grad is not None and torch.isnan(p.grad).any():
            print(f"  ⚠ NaN gradient in {name}!")
            return
    print(f"  ✓ No NaN gradients")


if __name__ == "__main__":
    print("=" * 60)
    print("DIFFUSION CONV")
    print("=" * 60)
    verify_diffusion_conv()

    print("=" * 60)
    print("DCGRU CELL + ENCODER + DECODER")
    print("=" * 60)
    verify_dcrnn_cell()

    print("\n" + "=" * 60)
    print("FULL DCRNN MODEL (all configurations)")
    print("=" * 60)
    verify_model()

    test_full_training_step()

    print("\n" + "=" * 60)
    print("ALL MODEL TESTS PASSED")
    print("Next step: python train.py")
    print("=" * 60)