"""
config.py

Single source of truth for all hyperparameters and paths.
Change things here, not scattered across files.
"""

import torch
import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR        = os.path.join(os.path.dirname(__file__), os.path.join("..", "data"))
OUTPUT_DIR      = os.path.join(os.path.dirname(__file__), "outputs")
VOLUME_PATH     = os.path.join(DATA_DIR, "sensor_volume_150.csv")
ADJ_PKL_PATH    = os.path.join(DATA_DIR, "adj_mat_volume.pkl")
LOCATIONS_PATH  = os.path.join(DATA_DIR, "sensor_location_150.csv")

# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------
# 'distance'    → rebuild from lat/lon (recommended)
# 'correlation' → build from time-series pearson correlation
# 'original'    → use the pkl as-is
GRAPH_METHOD    = "distance"

# Distance-based params
# sigma: Gaussian bandwidth in km. None = auto (std of all pairwise distances)
# threshold: edges below this weight are dropped. Lower = denser graph.
# Tune threshold until avg_neighbors is roughly 3-8 with 0 isolated nodes.
GRAPH_SIGMA     = 2.5
GRAPH_THRESHOLD = 0.1

# Correlation-based params (used if GRAPH_METHOD = 'correlation')
CORR_THRESHOLD  = 0.55

# Diffusion steps K — how many hops of neighbourhood info to aggregate
DIFFUSION_K     = 3

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
NUM_SENSORS     = 150

# Sliding window sizes (in hours — data is hourly)
INPUT_SEQ_LEN   = 12    # use last 12 hours to predict
OUTPUT_SEQ_LEN  = 3    # predict next 3 hours

# Train / val / test split (fractions)
TRAIN_RATIO     = 0.7
VAL_RATIO       = 0.1
TEST_RATIO      = 0.2

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
IN_FEATURES     = 1     # just volume at each sensor
HIDDEN_DIM      = 32    # DCGRU hidden state size
NUM_LAYERS      = 1     # stacked DCGRU layers in encoder and decoder

# Scheduled sampling — probability of feeding decoder its own prediction
# vs ground truth during training. Starts at 0, ramps up to MAX over
# SCHEDULED_SAMPLING_STEPS training steps (curriculum learning)
SCHEDULED_SAMPLING_START = 0.0
SCHEDULED_SAMPLING_MAX   = 0.5
SCHEDULED_SAMPLING_STEPS = 2000   # steps over which to ramp up

# ---------------------------------------------------------------------------
# Attention 
# ---------------------------------------------------------------------------
USE_ATTENTION   = True
ATTENTION_HEADS = 4     # multi-head attention over time dimension

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
BATCH_SIZE      = 32
LEARNING_RATE   = 1e-3
WEIGHT_DECAY    = 1e-4
NUM_EPOCHS      = 50
CLIP_GRAD_NORM  = 5.0   # gradient clipping 

# Learning rate scheduler — reduce on plateau
LR_PATIENCE     = 10
LR_FACTOR       = 0.5
LR_MIN          = 1e-5

# Evaluation horizons (in hours)
EVAL_HORIZONS   = [1, 2, 3]   # evaluate MAE/RMSE/MAPE at these steps

# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------
# Device selection: CUDA > MPS (Apple Silicon) > CPU
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

SEED            = 42
CHECKPOINT_DIR  = os.path.join(OUTPUT_DIR, "checkpoints")
LOG_INTERVAL    = 10