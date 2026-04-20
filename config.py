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

# Available models for multi-model experiments
AVAILABLE_MODELS = {
    "bert":     "bert-base-uncased",
    "roberta":  "roberta-base",
    "gpt2":     "gpt2",
    "mbert":    "bert-base-multilingual-cased",
}

# ─── Dataset Generation ─────────────────────────────────────────────────────────
MAX_DEPTH = 7                             # Maximum recursion depth to test
NUM_SENTENCES_PER_DEPTH = 30              # Sentences generated per depth level

# ─── SUD Treebank Paths ─────────────────────────────────────────────────────────
# Paths to SUD CoNLL-U files for cross-lingual evaluation
# Update these paths to match your local SUD treebank installation
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SUD_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "sud_treebanks")

SUD_TREEBANKS = {
    "english": os.path.join(SUD_DATA_DIR, "SUD_English-EWT", "en_ewt-sud-test.conllu"),
    "german":  os.path.join(SUD_DATA_DIR, "SUD_German-GSD", "de_gsd-sud-test.conllu"),
    "hindi":   os.path.join(SUD_DATA_DIR, "SUD_Hindi-HDTB", "hi_hdtb-sud-test.conllu"),
}

# SUD sentence filtering parameters
SUD_MIN_TOKENS = 5
SUD_MAX_TOKENS = 25

# ─── Attention Graph Pruning ─────────────────────────────────────────────────────
PRUNING_STRATEGY = "top_k"                # "top_k", "threshold", or "mst"
TOP_K = 1                                 # Keep top-k outgoing edges per token
ATTENTION_THRESHOLD = 0.1                 # Minimum attention weight (if using threshold)

# ─── Output Paths ────────────────────────────────────────────────────────────────
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

# Ensure output directories exist
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
