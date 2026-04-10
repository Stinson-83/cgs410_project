"""
Central configuration for the Transformer Attention & Syntactic Dependency Pipeline.

All experimental parameters are defined here for reproducibility.
"""

import os
import torch

# ─── Reproducibility ────────────────────────────────────────────────────────────
RANDOM_SEED = 42

# ─── Model ───────────────────────────────────────────────────────────────────────
MODEL_NAME = "bert-base-uncased"          # HuggingFace model identifier
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─── Dataset Generation ─────────────────────────────────────────────────────────
MAX_DEPTH = 7                             # Maximum recursion depth to test
NUM_SENTENCES_PER_DEPTH = 10              # Sentences generated per depth level

# ─── Attention Graph Pruning ─────────────────────────────────────────────────────
PRUNING_STRATEGY = "top_k"                # "top_k", "threshold", or "mst"
TOP_K = 1                                 # Keep top-k outgoing edges per token
ATTENTION_THRESHOLD = 0.1                 # Minimum attention weight (if using threshold)

# ─── Output Paths ────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

# Ensure output directories exist
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
