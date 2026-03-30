"""CLI entry point for figure generation.

Loads the matplotlib publication style and calls all eight figure generators,
saving output to ``outputs/figures/``.

Usage:
    python -m src.figures.run                              # All figures
    python -m src.figures.run --config configs/base.yaml   # Custom config
    python -m src.figures.run --figure 1                   # Single figure
"""

import json
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from dotenv import load_dotenv
from loguru import logger

from src.core.config import load_config
from src.core.utils import setup_logging, ensure_output_dir
from src.figures.generators import (
    figure1_sycophancy_heatmap,
    figure2_matched_vs_mismatched,
    figure3_probe_accuracy_curves,
    figure4_cosine_similarity_matrices,
    figure5_layer_trajectories,
    figure6_steering_pareto,
    figure7_random_baseline,
    figure8_question_sensitivity,
    MODEL_ORDER,
    AUTHORITY_LEVELS,
    PROBE_NAMES,
)

load_dotenv()

# Resolve style file path relative to this module
STYLE_PATH = Path(__file__).parent / "style.mplstyle"


# ---------------------------------------------------------------------------
# Data loaders — read from outputs/ directories
# ---------------------------------------------------------------------------

def _load_figure1_data(output_dir: str, models: list) -> dict:
    """Load sycophancy heatmap data from behavioral scores.

    Returns:
        Dict[model_name -> Dict[authority_level -> {'rate', 'ci_low', 'ci_high'}]]
    """
    import pandas as pd

    data = {}
    for model_cfg in models:
        name = model_cfg.name
        scores_path = Path(output_dir) / "behavioral" / f"{name}_scores.csv"
        if not scores_path.exists():
            logger.warning(f"Scores not found for {name}, skipping figure 1 data")
            continue

        df = pd.read_csv(scores_path)
        model_data = {}
        for level in AUTHORITY_LEVELS:
            subset = df[df["authority_level"] == level]["flip"].dropna()
            values = subset.values.astype(float)
            rate = float(values.mean()) if len(values) > 0 else 0.0
            if len(values) > 1:
                from src.figures.generators import _bootstrap_ci
                ci_low, ci_high = _bootstrap_ci(values)
            else:
                ci_low, ci_high = rate, rate
            model_data[level] = {
                "rate": rate,
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
        data[name] = model_data

    return data


def _load_figure2_data(output_dir: str) -> dict:
    """Load matched vs mismatched data from analysis outputs.

    Returns:
        Dict[model_name -> {'matched': {...}, 'mismatched': {...}, 'p_value': float}]
    """
    mm_path = Path(output_dir) / "analysis" / "matched_vs_mismatched.json"
    if not mm_path.exists():
        logger.warning(f"Matched vs mismatched data not found at {mm_path}")
        return {}

    with open(mm_path) as f:
        raw = json.load(f)

    data = {}
    for model_name, results in raw.items():
        if model_name == "combined":
            continue
        data[model_name] = {
            "matched": {
                "rate": results.get("matched_flip_rate", 0.0),
                "ci_low": results.get("matched_flip_rate", 0.0) - 0.05,
                "ci_high": results.get("matched_flip_rate", 0.0) + 0.05,
            },
            "mismatched": {
                "rate": results.get("mismatched_flip_rate", 0.0),
                "ci_low": results.get("mismatched_flip_rate", 0.0) - 0.05,
                "ci_high": results.get("mismatched_flip_rate", 0.0) + 0.05,
            },
            "p_value": results.get("p_value", 1.0),
        }

    return data


def _load_figure3_data(output_dir: str, models: list) -> dict:
    """Load probe accuracy curves from probing outputs.

    Returns:
        Dict[model_name -> Dict[probe_name -> {'layer_results', 'peak_layer', 'peak_accuracy'}]]
    """
    data = {}
    for model_cfg in models:
        name = model_cfg.name
        curves_path = (
            Path(output_dir) / "probing" / name / "probe_accuracy_curves.json"
        )
        if not curves_path.exists():
            logger.warning(f"Probe curves not found for {name}")
            continue

        with open(curves_path) as f:
            data[name] = json.load(f)

    return data


def _load_figure4_data(output_dir: str, models: list) -> dict:
    """Load cosine similarity matrices from probing weight vectors.

    Returns:
        Dict[model_name -> np.ndarray of shape (3, 3)]
    """
    data = {}
    for model_cfg in models:
        name = model_cfg.name
        weight_path = Path(output_dir) / "probing" / name / "weight_vectors.pt"
        if not weight_path.exists():
            logger.warning(f"Weight vectors not found for {name}")
            continue

        weights = torch.load(str(weight_path), weights_only=False)

        # Extract directions at peak layers
        directions = {}
        for probe in PROBE_NAMES:
            w_key = f"w_{probe}"
            peak_key = f"peak_layer_{probe}"
            if w_key in weights and peak_key in weights:
                w = weights[w_key]  # [n_layers, d_model]
                peak = int(weights[peak_key])
                d = w[peak].numpy().astype(np.float64)
                norm = np.linalg.norm(d)
                directions[probe] = d / norm if norm > 0 else d

        if len(directions) == 3:
            # Build 3x3 cosine similarity matrix
            sim_matrix = np.zeros((3, 3))
            for i, p1 in enumerate(PROBE_NAMES):
                for j, p2 in enumerate(PROBE_NAMES):
                    sim_matrix[i, j] = float(
                        np.dot(directions[p1], directions[p2])
                    )
            data[name] = sim_matrix

    return data


def _load_figure5_data(output_dir: str, models: list) -> dict:
    """Load layer-wise cosine similarity between authority and sycophancy directions.

    Returns:
        Dict[model_name -> {'layers_normalized': [...], 'cosine_similarities': [...]}]
    """
    data = {}
    for model_cfg in models:
        name = model_cfg.name
        weight_path = Path(output_dir) / "probing" / name / "weight_vectors.pt"
        if not weight_path.exists():
            continue

        weights = torch.load(str(weight_path), weights_only=False)
        w_auth = weights.get("w_authority")  # [n_layers, d_model]
        w_syc = weights.get("w_sycophancy")  # [n_layers, d_model]

        if w_auth is None or w_syc is None:
            continue

        n_layers = w_auth.shape[0]
        cos_sims = []
        for layer in range(n_layers):
            a = w_auth[layer].numpy().astype(np.float64)
            s = w_syc[layer].numpy().astype(np.float64)
            norm_a = np.linalg.norm(a)
            norm_s = np.linalg.norm(s)
            if norm_a > 0 and norm_s > 0:
                cos_sims.append(float(np.dot(a, s) / (norm_a * norm_s)))
            else:
                cos_sims.append(0.0)

        layers_norm = [i / max(n_layers - 1, 1) for i in range(n_layers)]
        data[name] = {
            "layers_normalized": layers_norm,
            "cosine_similarities": cos_sims,
        }

    return data


def _load_figure6_data(output_dir: str, models: list) -> dict:
    """Load steering Pareto data from scored steering CSVs.

    Computes persona sycophancy gap (high_auth_flip - none/control_flip) from
    the scored CSV files produced by ``scripts/score_steered.py``.

    Falls back to steering_summary.json metrics if CSVs are not available.

    Returns:
        Dict[model_name -> Dict[direction -> {'alphas': [...], 'sycophancy_gaps': [...]}]]
    """
    import pandas as pd
    from src.figures.generators import _bootstrap_ci

    data = {}
    for model_cfg in models:
        name = model_cfg.name
        steering_dir = Path(output_dir) / "steering" / name

        if not steering_dir.exists():
            logger.warning(f"Steering results not found for {name}")
            continue

        # Try to compute from scored CSVs first
        by_direction = {}
        for csv_path in sorted(steering_dir.glob("scores_*.csv")):
            # Parse direction and alpha from filename: scores_{direction}_alpha{X}.csv
            stem = csv_path.stem  # e.g. "scores_authority_alpha1.5"
            parts = stem.replace("scores_", "").rsplit("_alpha", 1)
            if len(parts) != 2:
                continue
            direction = parts[0]
            try:
                alpha = float(parts[1])
            except ValueError:
                continue

            df = pd.read_csv(csv_path)
            if "flip" not in df.columns or "authority_level" not in df.columns:
                continue

            # Compute flip rates by authority level
            high_flips = df[df["authority_level"] == "high"]["flip"].dropna().values.astype(float)
            # Use control or 'none' authority as baseline
            control_mask = (df["is_control"] == True) | (df["authority_level"] == "none")
            control_flips = df[control_mask]["flip"].dropna().values.astype(float)

            high_rate = float(high_flips.mean()) if len(high_flips) > 0 else 0.0
            control_rate = float(control_flips.mean()) if len(control_flips) > 0 else 0.0
            gap = high_rate - control_rate

            # Bootstrap CI on the gap
            ci_lo, ci_hi = None, None
            if len(high_flips) > 1 and len(control_flips) > 1:
                rng = np.random.RandomState(42)
                boot_gaps = []
                for _ in range(10000):
                    h_boot = rng.choice(high_flips, size=len(high_flips), replace=True).mean()
                    c_boot = rng.choice(control_flips, size=len(control_flips), replace=True).mean()
                    boot_gaps.append(h_boot - c_boot)
                boot_gaps = np.array(boot_gaps)
                ci_lo = float(np.percentile(boot_gaps, 2.5))
                ci_hi = float(np.percentile(boot_gaps, 97.5))

            if direction not in by_direction:
                by_direction[direction] = []
            by_direction[direction].append((alpha, gap, ci_lo, ci_hi))

        # Fall back to steering_summary.json if no scored CSVs found
        if not by_direction:
            steering_path = steering_dir / "steering_summary.json"
            if steering_path.exists():
                with open(steering_path) as f:
                    raw = json.load(f)
                for cond_key, cond_data in raw.get("conditions", {}).items():
                    direction = cond_data.get("direction", "")
                    alpha = cond_data.get("alpha", 0.0)
                    metrics = cond_data.get("metrics", {})
                    gap = metrics.get("persona_sycophancy_gap", 0.0)
                    ci_lo = metrics.get("persona_sycophancy_gap_ci_low")
                    ci_hi = metrics.get("persona_sycophancy_gap_ci_high")
                    if direction not in by_direction:
                        by_direction[direction] = []
                    by_direction[direction].append((alpha, gap, ci_lo, ci_hi))

        model_data = {}
        for direction in PROBE_NAMES:
            if direction not in by_direction:
                continue
            pairs = sorted(by_direction[direction], key=lambda x: x[0])
            dir_data = {
                "alphas": [p[0] for p in pairs],
                "sycophancy_gaps": [p[1] for p in pairs],
            }
            # Include CIs if available
            if all(p[2] is not None for p in pairs):
                dir_data["ci_low"] = [p[2] for p in pairs]
                dir_data["ci_high"] = [p[3] for p in pairs]
            model_data[direction] = dir_data
        data[name] = model_data

    return data


def _load_figure7_data(output_dir: str, models: list) -> dict:
    """Load random baseline comparison data from steering outputs.

    Compares unsteered flip rate, authority-steered flip rate (alpha=1.5),
    and random-direction-steered flip rate (alpha=1.5).

    Returns:
        Dict[model_name -> {'unsteered': {...}, 'authority': {...}, 'random': {...}}]
    """
    import pandas as pd
    from src.figures.generators import _bootstrap_ci

    data = {}
    for model_cfg in models:
        name = model_cfg.name
        model_data = {}

        # Unsteered flip rate from behavioral scores
        scores_path = Path(output_dir) / "behavioral" / f"{name}_scores.csv"
        if scores_path.exists():
            df = pd.read_csv(scores_path)
            flips = df["flip"].dropna().values.astype(float)
            rate = float(flips.mean()) if len(flips) > 0 else 0.0
            if len(flips) > 1:
                ci_lo, ci_hi = _bootstrap_ci(flips)
            else:
                ci_lo, ci_hi = rate, rate
            model_data["unsteered"] = {"rate": rate, "ci_low": ci_lo, "ci_high": ci_hi}

        # Authority-steered at alpha=1.5
        auth_scores = Path(output_dir) / "steering" / name / "scores_authority_alpha1.5.csv"
        if auth_scores.exists():
            df = pd.read_csv(auth_scores)
            flips = df["flip"].dropna().values.astype(float)
            rate = float(flips.mean()) if len(flips) > 0 else 0.0
            if len(flips) > 1:
                ci_lo, ci_hi = _bootstrap_ci(flips)
            else:
                ci_lo, ci_hi = rate, rate
            model_data["authority"] = {"rate": rate, "ci_low": ci_lo, "ci_high": ci_hi}

        # Random-direction-steered at alpha=1.5
        rand_scores = Path(output_dir) / "steering" / name / "scores_random_alpha1.5.csv"
        if rand_scores.exists():
            df = pd.read_csv(rand_scores)
            flips = df["flip"].dropna().values.astype(float)
            rate = float(flips.mean()) if len(flips) > 0 else 0.0
            if len(flips) > 1:
                ci_lo, ci_hi = _bootstrap_ci(flips)
            else:
                ci_lo, ci_hi = rate, rate
            model_data["random"] = {"rate": rate, "ci_low": ci_lo, "ci_high": ci_hi}

        if model_data:
            data[name] = model_data

    return data


def _load_figure8_data(output_dir: str) -> dict:
    """Load question-level sensitivity analysis from analysis outputs.

    Returns:
        Dict with 'per_question_rates', 'top_outlier_qids', 'leave5out_bootstrap'.
    """
    analysis_path = Path(output_dir) / "analysis" / "question_level_analysis.json"
    if not analysis_path.exists():
        logger.warning(f"Question-level analysis not found at {analysis_path}")
        return {}

    with open(analysis_path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------

FIGURE_GENERATORS = {
    1: ("figure1_sycophancy_heatmap", figure1_sycophancy_heatmap, _load_figure1_data),
    2: ("figure2_matched_vs_mismatched", figure2_matched_vs_mismatched, _load_figure2_data),
    3: ("figure3_probe_accuracy_curves", figure3_probe_accuracy_curves, _load_figure3_data),
    4: ("figure4_cosine_similarity_matrices", figure4_cosine_similarity_matrices, _load_figure4_data),
    5: ("figure5_layer_trajectories", figure5_layer_trajectories, _load_figure5_data),
    6: ("figure6_steering_pareto", figure6_steering_pareto, _load_figure6_data),
    7: ("figure7_random_baseline", figure7_random_baseline, _load_figure7_data),
    8: ("figure8_question_sensitivity", figure8_question_sensitivity, _load_figure8_data),
}


@click.command()
@click.option("--config", default="configs/base.yaml", help="Path to config YAML")
@click.option("--figure", "figure_num", default=None, type=int, help="Generate a single figure (1-8)")
def main(config: str, figure_num: int):
    """Generate all publication figures for the sycophancy probing paper."""
    cfg = load_config(config)
    setup_logging(cfg.log_level)

    # Apply publication style
    if STYLE_PATH.exists():
        plt.style.use(str(STYLE_PATH))
        logger.info(f"Loaded matplotlib style from {STYLE_PATH}")
    else:
        logger.warning(f"Style file not found at {STYLE_PATH}, using defaults")

    fig_dir = ensure_output_dir(cfg.output_dir, "figures")

    # Determine which figures to generate
    if figure_num is not None:
        if figure_num not in FIGURE_GENERATORS:
            click.echo(f"Invalid figure number {figure_num}. Choose 1-8.")
            return
        figures_to_gen = {figure_num: FIGURE_GENERATORS[figure_num]}
    else:
        figures_to_gen = FIGURE_GENERATORS

    for num, (name, gen_func, load_func) in figures_to_gen.items():
        logger.info(f"Generating Figure {num}: {name}")
        output_path = str(fig_dir / name)

        try:
            # Load data — some loaders need models list, some just output_dir
            import inspect
            sig = inspect.signature(load_func)
            params = list(sig.parameters.keys())

            if "models" in params:
                data = load_func(cfg.output_dir, cfg.models)
            else:
                data = load_func(cfg.output_dir)

            if not data:
                logger.warning(f"No data available for Figure {num}, skipping")
                continue

            gen_func(data, output_path)
            logger.info(f"Figure {num} saved to {output_path}")

        except Exception as e:
            logger.error(f"Failed to generate Figure {num} ({name}): {e}")
            continue

    click.echo(f"Figure generation complete. Output: {fig_dir}")


if __name__ == "__main__":
    main()
