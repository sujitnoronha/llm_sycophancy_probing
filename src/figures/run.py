"""CLI entry point for figure generation.

Loads the matplotlib publication style and calls all six figure generators,
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
    """Load steering Pareto data from steering outputs.

    Parses the conditions dict from steering_summary.json, which has keys
    like ``authority_alpha0.5``, and groups by direction.

    Returns:
        Dict[model_name -> Dict[direction -> {'alphas': [...], 'sycophancy_gaps': [...]}]]
    """
    data = {}
    for model_cfg in models:
        name = model_cfg.name
        steering_path = Path(output_dir) / "steering" / name / "steering_summary.json"
        if not steering_path.exists():
            logger.warning(f"Steering results not found for {name}")
            continue

        with open(steering_path) as f:
            raw = json.load(f)

        # Group conditions by direction
        by_direction = {}
        for cond_key, cond_data in raw.get("conditions", {}).items():
            direction = cond_data.get("direction", "")
            alpha = cond_data.get("alpha", 0.0)
            metrics = cond_data.get("metrics", {})
            gap = metrics.get("persona_sycophancy_gap", 0.0)

            if direction not in by_direction:
                by_direction[direction] = []
            by_direction[direction].append((alpha, gap))

        model_data = {}
        for direction in PROBE_NAMES:
            if direction not in by_direction:
                continue
            pairs = sorted(by_direction[direction], key=lambda x: x[0])
            model_data[direction] = {
                "alphas": [p[0] for p in pairs],
                "sycophancy_gaps": [p[1] for p in pairs],
            }
        data[name] = model_data

    return data


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
}


@click.command()
@click.option("--config", default="configs/base.yaml", help="Path to config YAML")
@click.option("--figure", "figure_num", default=None, type=int, help="Generate a single figure (1-6)")
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
            click.echo(f"Invalid figure number {figure_num}. Choose 1-6.")
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
