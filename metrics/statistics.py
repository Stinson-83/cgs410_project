"""
metrics/statistics.py — Statistical Inference for Hypothesis Testing

Implements three statistical tests used to formally test the degradation
hypothesis (H2):
    1. Spearman rank correlation (metric ~ depth)
    2. Mann-Whitney U test (depth 1 vs depth K distributions)
    3. Rank-biserial effect size

References:
    - scipy.stats.spearmanr
    - scipy.stats.mannwhitneyu
"""

import numpy as np
from scipy.stats import spearmanr, mannwhitneyu
from typing import Dict, List, Any, Optional, Tuple


def spearman_depth_correlation(
    depth_means: Dict[int, float],
) -> Tuple[float, float]:
    """
    Compute Spearman rank correlation between depth and a metric.

    Args:
        depth_means: Dictionary mapping depth (int) -> mean metric value.

    Returns:
        (correlation coefficient rho, p-value)
    """
    depths = sorted(depth_means.keys())
    values = [depth_means[d] for d in depths]

    if len(depths) < 3:
        return (0.0, 1.0)

    rho, p = spearmanr(depths, values)
    return (float(rho), float(p))


def mann_whitney_test(
    values_group1: List[float],
    values_group2: List[float],
) -> Dict[str, float]:
    """
    Mann-Whitney U test comparing two groups (e.g., depth 1 vs depth 7).

    Args:
        values_group1: Metric values for group 1 (e.g., depth 1 sentences).
        values_group2: Metric values for group 2 (e.g., depth 7 sentences).

    Returns:
        Dictionary with keys: "U", "p_value", "rank_biserial"
    """
    if len(values_group1) < 2 or len(values_group2) < 2:
        return {"U": 0.0, "p_value": 1.0, "rank_biserial": 0.0}

    u_stat, p_value = mannwhitneyu(
        values_group1, values_group2, alternative="two-sided"
    )

    # Rank-biserial correlation: r_rb = 1 - (2U) / (n1 * n2)
    n1 = len(values_group1)
    n2 = len(values_group2)
    r_rb = 1 - (2 * u_stat) / (n1 * n2)

    return {
        "U": float(u_stat),
        "p_value": float(p_value),
        "rank_biserial": float(r_rb),
    }


def compute_all_statistical_tests(
    aggregated_results: Dict[str, Any],
    raw_results: Dict[int, list],
    metric_names: Optional[List[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Run all statistical tests for each metric.

    Args:
        aggregated_results: Output of _aggregate_metrics() from runner.py.
        raw_results: Raw per-sentence results keyed by depth.
        metric_names: Metrics to test. Defaults to all four.

    Returns:
        Dictionary: metric_name -> {
            "spearman_rho": float,
            "spearman_p": float,
            "mann_whitney_U": float,
            "mann_whitney_p": float,
            "rank_biserial": float,
        }
    """
    if metric_names is None:
        metric_names = ["DERR", "USO", "TDC", "AMTH"]

    per_depth_best = aggregated_results.get("per_depth_best_layer", {})
    depths = sorted(per_depth_best.keys(), key=int)

    if len(depths) < 2:
        return {}

    results = {}

    for metric in metric_names:
        # ── Spearman correlation ─────────────────────────────────────────
        depth_means = {
            int(d): per_depth_best[d][metric]["mean"] for d in depths
        }
        rho, p_spearman = spearman_depth_correlation(depth_means)

        # ── Mann-Whitney U (depth 1 vs deepest) ─────────────────────────
        d_min = min(depths, key=int)
        d_max = max(depths, key=int)

        values_d1 = _extract_best_layer_values(raw_results, int(d_min), metric)
        values_dk = _extract_best_layer_values(raw_results, int(d_max), metric)

        mw = mann_whitney_test(values_d1, values_dk)

        results[metric] = {
            "spearman_rho": rho,
            "spearman_p": p_spearman,
            "mann_whitney_U": mw["U"],
            "mann_whitney_p": mw["p_value"],
            "rank_biserial": mw["rank_biserial"],
        }

    return results


def _extract_best_layer_values(
    raw_results: Dict[int, list],
    depth: int,
    metric: str,
) -> List[float]:
    """
    Extract per-sentence metric values at the best-performing layer for a depth.

    For each sentence, picks the layer/head combination that maximises
    the given metric, then returns the list of those best values.
    """
    sentences = raw_results.get(depth, [])
    values = []

    for sent_data in sentences:
        metrics_by_layer = sent_data.get("metrics", [])
        if not metrics_by_layer:
            continue

        best_val = -float("inf")
        for layer_metrics in metrics_by_layer:
            for head_metrics in layer_metrics:
                v = head_metrics.get(metric, 0.0)
                if v > best_val:
                    best_val = v

        values.append(best_val)

    return values


def format_statistical_report(
    test_results: Dict[str, Dict[str, Any]],
) -> str:
    """
    Format statistical test results as a readable text report.

    Returns:
        Multi-line string summarising all tests.
    """
    lines = []
    lines.append("=" * 60)
    lines.append("STATISTICAL INFERENCE RESULTS")
    lines.append("=" * 60)

    for metric, res in test_results.items():
        lines.append(f"\n  {metric}:")
        lines.append(
            f"    Spearman rho = {res['spearman_rho']:.3f}, "
            f"p = {res['spearman_p']:.4f}"
        )
        lines.append(
            f"    Mann-Whitney U = {res['mann_whitney_U']:.1f}, "
            f"p = {res['mann_whitney_p']:.4f}"
        )
        lines.append(
            f"    Rank-biserial r_rb = {res['rank_biserial']:.3f}"
        )

        # Interpret
        if res["spearman_p"] < 0.05:
            direction = "negative" if res["spearman_rho"] < 0 else "positive"
            lines.append(
                f"    => Significant {direction} correlation with depth (p < 0.05)"
            )
        else:
            lines.append("    => No significant correlation with depth")

        if abs(res["rank_biserial"]) > 0.5:
            lines.append("    => Large effect size")
        elif abs(res["rank_biserial"]) > 0.3:
            lines.append("    => Medium effect size")
        else:
            lines.append("    => Small effect size")

    lines.append("\n" + "=" * 60)
    return "\n".join(lines)


# ─── CLI test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Quick test with synthetic data
    depth_means = {1: 0.472, 2: 0.190, 3: 0.197, 4: 0.163, 5: 0.141, 6: 0.118, 7: 0.102}
    rho, p = spearman_depth_correlation(depth_means)
    print(f"Spearman test: rho={rho:.3f}, p={p:.4f}")

    g1 = [0.5, 0.45, 0.52, 0.48, 0.41, 0.55, 0.47, 0.50, 0.43, 0.46]
    g2 = [0.10, 0.08, 0.12, 0.09, 0.11, 0.07, 0.13, 0.10, 0.09, 0.08]
    mw = mann_whitney_test(g1, g2)
    print(f"Mann-Whitney test: U={mw['U']:.1f}, p={mw['p_value']:.4f}, r_rb={mw['rank_biserial']:.3f}")
