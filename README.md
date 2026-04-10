# Emergence and Limits of Dependency Structure in Transformer Attention Under Increasing Recursive Depth

A research pipeline investigating whether transformer attention mechanisms dynamically construct syntactic dependency structures during inference, and how robust this mechanism is as sentences become more recursively complex.

## Research Question

> Do transformer attention mechanisms dynamically construct syntactic dependency structures during inference, and how robust is this mechanism when sentences contain increasingly deep recursive structures?

## Hypothesis

- Intermediate transformer layers partially approximate syntactic dependency trees
- As recursion depth increases, dependency distances grow and alignment weakens
- This suggests transformers rely more on **local token interactions** than fully hierarchical representations at higher depths

## Project Structure

```
project/
├── config.py                   # Central configuration
├── main.py                     # CLI entry point
├── requirements.txt            # Dependencies
│
├── data/
│   └── generator.py            # Controlled recursive sentence generation
│
├── parsing/
│   └── dependency_parser.py    # Gold dependency tree extraction (spaCy)
│
├── models/
│   └── attention_extractor.py  # BERT attention weight extraction + alignment
│
├── graphs/
│   └── attention_graph.py      # Attention → directed graph conversion
│
├── metrics/
│   └── comparison.py           # 4 comparison metrics (DERR, USO, TDC, AMTH)
│
├── experiments/
│   └── runner.py               # Experiment orchestrator
│
├── visualization/
│   └── plots.py                # Publication-quality visualizations
│
└── results/                    # Generated outputs (metrics + figures)
    └── figures/
```

## Installation

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm
```

## Usage

```bash
# Full pipeline with defaults (depth 1-7, 10 sentences/depth, BERT)
python main.py

# Quick test run
python main.py --max_depth 3 --num_sentences 3

# Use GPU
python main.py --device cuda

# Custom configuration
python main.py --max_depth 10 --num_sentences 20 --model_name bert-base-uncased

# Threshold-based pruning instead of top-k
python main.py --pruning threshold --threshold 0.1

python main.py --pruning top_k

python main.py --pruning mst
```

## Metrics

| Metric | Abbreviation | Range | Description |
|--------|-------------|-------|-------------|
| Dependency Edge Recovery Rate | DERR | [0, 1] | Fraction of gold edges recovered in attention graph |
| Undirected Structural Overlap | USO | [0, 1] | Jaccard similarity of undirected edge sets |
| Tree Distance Correlation | TDC | [-1, 1] | Spearman correlation of pairwise shortest-path distances |
| Attention Mass on True Head | AMTH | [0, 1] | Mean attention weight on syntactic head |

## Outputs

After running, check `results/` for:
- `metrics_results.json` — Per-depth, per-layer aggregated metrics
- `figures/depth_vs_alignment.png` — How alignment degrades with depth
- `figures/layer_vs_alignment_heatmap.png` — Which layers capture syntax best
- `figures/attention_heatmap_example.png` — Attention weights with gold edges
- `figures/dependency_overlay_example.png` — Gold tree vs attention graph
- `figures/head_analysis.png` — Best heads per layer

## Sentence Templates

Three template families ensure diversity:

1. **Subject Relative Clauses**: "The dog that chased the cat barked."
2. **Object Relative Clauses**: "The man saw the dog that chased the cat."
3. **PP Stacking**: "The boy on the roof near the chimney laughed."

## References

- Hewitt & Manning (2019): *A Structural Probe for Finding Syntax in Word Representations*
- Clark et al. (2019): *What Does BERT Look At? An Analysis of BERT's Attention*
- Tenney et al. (2019): *BERT Rediscovers the Classical NLP Pipeline*
- Jawahar et al. (2019): *What Does BERT Learn about the Structure of Language?*
