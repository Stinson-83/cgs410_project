# Emergence and Limits of Dependency Structure in Transformer Attention

A research pipeline investigating whether transformer attention mechanisms dynamically construct syntactic dependency structures during inference, and how robust this mechanism is as sentences become more recursively complex.

## Research Question

> Do transformer attention mechanisms dynamically construct syntactic dependency structures during inference, and how robust is this mechanism when sentences contain increasingly deep recursive structures?

## Hypothesis

- Intermediate transformer layers partially approximate syntactic dependency trees.
- As recursion depth increases, dependency distances grow and alignment weakens.
- This suggests transformers rely more on **local token interactions** than fully hierarchical representations at higher depths.

## Architecture and Workflow

The project consists of a full experimental pipeline that takes text (either procedurally generated, from SUD treebanks, or user-provided) and processes it through 7 distinct stages:

1. **Text Acquisition**:
   - *Generation Mode*: Uses `data/generator.py` to create controlled recursive sentences across varying depths (e.g., Subject Relative Clauses, Object Relative Clauses, PP Stacking).
   - *SUD Treebank Mode*: Uses `data/sud_loader.py` to load naturalistic sentences from Surface-syntactic Universal Dependencies treebanks (English, German, Hindi), binned by dependency tree depth.
   - *Custom Inference Mode*: Bypasses generation to analyze custom sentences provided directly via CLI.
2. **Gold Dependency Parsing**: Uses `spaCy` (`en_core_web_sm`) inside `parsing/dependency_parser.py` for generated data, or SUD gold annotations directly for treebank data.
3. **Attention Extraction**: Uses HuggingFace models (BERT, RoBERTa, GPT-2, mBERT) via `models/attention_extractor.py` to run inference and extract multi-head attention weights across all layers.
4. **Attention Graph Construction**: Transforms high-dimensional attention weights into directed graphs using various pruning algorithms (MST, Top-K, Threshold) in `graphs/attention_graph.py`.
5. **Metric Computation**: Compares attention-derived graphs to gold trees to quantify syntactic alignment using established metrics in `metrics/comparison.py`.
6. **Statistical Inference**: Runs Spearman correlation, Mann-Whitney U tests, and computes rank-biserial effect sizes via `metrics/statistics.py`.
7. **Visualization & Synthesis**: Generates heatmaps, bar charts, and JSON reports detailing performance across layers and recursion depths in `visualization/plots.py`.

## Project Structure

```text
project/
├── config.py                   # Central configuration parameters
├── main.py                     # CLI entry point orchestrator
├── requirements.txt            # Python dependencies
│
├── data/
│   ├── generator.py            # Controlled recursive sentence generation
│   └── sud_loader.py           # SUD treebank CoNLL-U loader + depth binning
├── parsing/
│   └── dependency_parser.py    # Gold dependency tree extraction (spaCy)
├── models/
│   └── attention_extractor.py  # Multi-model attention extraction + alignment
├── graphs/
│   └── attention_graph.py      # Attention → directed graph conversion (Pruning)
├── metrics/
│   ├── comparison.py           # Evaluation metrics (DERR, USO, TDC, AMTH)
│   └── statistics.py           # Statistical inference (Spearman, Mann-Whitney)
├── experiments/
│   └── runner.py               # Experiment orchestration (generated + SUD)
├── visualization/
│   └── plots.py                # Publication-quality visualizations (Matplotlib)
└── results/                    # Generated outputs (metrics + figures + stats)
```

## Pruning Strategies

To convert raw dense attention mechanisms into distinct graph edges that model syntactic structures, the pipeline supports multiple pruning strategies. Results for different strategies are securely saved into independent directories (e.g., `results/mst/`, `results/top_k/`) to prevent overlapping or overwriting.

- **Maximum Spanning Tree (MST)**: Ideal for finding a connected, tree-like structure from attention weights.
- **Top-K**: Retains the `k` highest probability attention edges per token.
- **Threshold**: Hard-pruning of edges below a specific probability.

## Installation

```bash
# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download the required spaCy model (essential for gold dependency parsing)
python -m spacy download en_core_web_sm
```
*Note: If offline, you can directly pip install the `.whl` files provided in the root directory.*

## How to Run (Usage)

The primary entry point is `main.py`, which is heavily parameterised for flexibility.

### 1. Default Pipeline Run
Runs the full pipeline generating recursive sentences (depth 1-7, 10 sentences each) against BERT and evaluates both `mst` and `top_k` pruning strategies sequentially.
```bash
python main.py
```

### 2. Fast Testing
Ideal for debugging or quickly generating small sample metric sets.
```bash
python main.py --max_depth 3 --num_sentences 3
```

### 3. Executing on Custom Sentences 
To visualize the alignment of custom text rather than using the recursive generator:
```bash
python main.py --custom_sentences "The cat sat on the mat." "Transformers learn contextual representations."
```

### 4. Customizing Pruning Strategies
You can select a specific pruning technique instead of running the default (both `mst` and `top_k`).

```bash
# Evaluate ONLY maximum spanning tree (MST)
python main.py --pruning mst

# Evaluate ONLY top-K edges
python main.py --pruning top_k --top_k 2

# Evaluate based on hard probability thresholds
python main.py --pruning threshold --threshold 0.1
```

### 5. Hardware & Model Overrides
```bash
# Run with GPU acceleration
python main.py --device cuda

# Override HuggingFace models
python main.py --model_name roberta-base
```

## Metrics Explanation

The pipeline computes four key metrics indicating structural alignment:

| Metric | Abbreviation | Range | Description |
|--------|-------------|-------|-------------|
| **Dependency Edge Recovery Rate** | DERR | `[0, 1]` | Fraction of gold dependency edges recovered in the extracted attention graph. |
| **Undirected Structural Overlap** | USO | `[0, 1]` | Jaccard similarity of undirected edge sets between attention and gold tree. |
| **Tree Distance Correlation** | TDC | `[-1, 1]` | Spearman correlation of pairwise shortest-path distances in the tree vs attention graphs. |
| **Attention Mass on True Head** | AMTH | `[0, 1]` | Average model attention weight focused exactly on a token's syntactic head. |

## Outputs

All outcomes are placed in the `results/` directory, separated by the pruning strategy used (e.g., `results/mst/`, `results/top_k/`). An output bundle will contain:

- `metrics_results.json`: Per-depth, per-layer aggregated metrics (DERR, USO, TDC, AMTH).
- `figures/depth_vs_alignment.png`: Shows how syntactic alignment degrades as recursive depth (complexity) increases.
- `figures/layer_vs_alignment_heatmap.png`: Identifies which transformer layers capture syntax best.
- `figures/attention_heatmap_example.png`: Raw attention weights overlapped with gold edges.
- `figures/dependency_overlay_example.png`: Networkx visualizations of the gold tree vs attention graph.
- `figures/head_analysis.png`: Highlights the specific heads responsible for capturing grammar per layer.

## References

- **Hewitt & Manning (2019)**: *A Structural Probe for Finding Syntax in Word Representations*
- **Clark et al. (2019)**: *What Does BERT Look At? An Analysis of BERT's Attention*
- **Tenney et al. (2019)**: *BERT Rediscovers the Classical NLP Pipeline*
- **Jawahar et al. (2019)**: *What Does BERT Learn about the Structure of Language?*
