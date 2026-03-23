"""Figure generators for the sycophancy probing paper.

Eight publication-quality figures for the COLM submission. Each function
loads data from the appropriate ``outputs/`` directory, creates a
matplotlib figure using the project style, and saves to ``output_path``
as both PDF and PNG.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from loguru import logger


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MODEL_ORDER = [
    "llama31_8b",
    "qwen3_8b",
    "mistral_7b_v03",
    "gemma3_12b",
    "deepseek_r1_qwen3_8b",
]

MODEL_DISPLAY_NAMES = {
    "llama31_8b": "Llama-3.1-8B",
    "qwen3_8b": "Qwen-3-8B",
    "mistral_7b_v03": "Mistral-7B-v0.3",
    "gemma3_12b": "Gemma-3-12B",
    "deepseek_r1_qwen3_8b": "DeepSeek-R1-Qwen3-8B",
}

AUTHORITY_LEVELS = ["none", "low", "medium", "high"]

AUTHORITY_DISPLAY = {
    "none": "No Persona",
    "low": "Low",
    "medium": "Medium",
    "high": "High",
}

PROBE_NAMES = ["authority", "sycophancy", "confidence"]

PROBE_DISPLAY = {
    "authority": "Authority",
    "sycophancy": "Sycophancy",
    "confidence": "Confidence",
}

DIRECTION_SHORT = {
    "authority": "Auth",
    "sycophancy": "Syc",
    "confidence": "Conf",
}


def _save_figure(fig: plt.Figure, output_path: str) -> None:
    """Save a figure as both PDF and PNG."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Save PDF
    pdf_path = path.with_suffix(".pdf")
    fig.savefig(str(pdf_path))
    logger.info(f"Saved {pdf_path}")

    # Save PNG
    png_path = path.with_suffix(".png")
    fig.savefig(str(png_path))
    logger.info(f"Saved {png_path}")

    plt.close(fig)


def _bootstrap_ci(
    values: np.ndarray,
    n_boot: int = 10000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float]:
    """Compute bootstrap confidence interval for the mean."""
    rng = np.random.RandomState(seed)
    boot_means = np.array([
        rng.choice(values, size=len(values), replace=True).mean()
        for _ in range(n_boot)
    ])
    alpha = (1 - ci) / 2
    return float(np.percentile(boot_means, 100 * alpha)), float(
        np.percentile(boot_means, 100 * (1 - alpha))
    )


def _significance_stars(p: float) -> str:
    """Convert a p-value to significance stars."""
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    return "ns"


# ---------------------------------------------------------------------------
# Figure 1: Sycophancy heatmap (5 models x 4 authority levels)
# ---------------------------------------------------------------------------

def figure1_sycophancy_heatmap(data: Dict, output_path: str) -> None:
    """5 models x 4 authority levels heatmap of sycophancy (flip) rates.

    Args:
        data: Dict mapping model_name -> DataFrame-like data with
              'authority_level' and 'flip' columns, OR a pre-computed
              dict with model_name -> {authority_level -> {'rate', 'ci_low', 'ci_high'}}.
        output_path: Path to save the figure (without extension).
    """
    n_models = len(MODEL_ORDER)
    n_levels = len(AUTHORITY_LEVELS)

    # Build the rate matrix and annotation matrix
    rate_matrix = np.zeros((n_models, n_levels))
    annotations = np.empty((n_models, n_levels), dtype=object)

    for i, model in enumerate(MODEL_ORDER):
        model_data = data.get(model, {})
        for j, level in enumerate(AUTHORITY_LEVELS):
            level_data = model_data.get(level, {})

            if isinstance(level_data, dict):
                # Pre-computed format
                rate = level_data.get("rate", 0.0)
                ci_low = level_data.get("ci_low", rate)
                ci_high = level_data.get("ci_high", rate)
            else:
                # Raw array of flip values
                flips = np.array(level_data, dtype=float)
                rate = flips.mean() if len(flips) > 0 else 0.0
                if len(flips) > 1:
                    ci_low, ci_high = _bootstrap_ci(flips)
                else:
                    ci_low, ci_high = rate, rate

            rate_matrix[i, j] = rate
            pct = rate * 100
            ci_lo_pct = ci_low * 100
            ci_hi_pct = ci_high * 100
            annotations[i, j] = f"{pct:.1f}%\n[{ci_lo_pct:.1f}, {ci_hi_pct:.1f}]"

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(
        rate_matrix,
        annot=annotations,
        fmt="",
        cmap="YlOrRd",
        vmin=0,
        vmax=1,
        linewidths=0.5,
        linecolor="white",
        xticklabels=[AUTHORITY_DISPLAY[l] for l in AUTHORITY_LEVELS],
        yticklabels=[MODEL_DISPLAY_NAMES.get(m, m) for m in MODEL_ORDER],
        cbar_kws={"label": "Flip Rate", "shrink": 0.8},
        ax=ax,
    )
    ax.set_xlabel("Authority Level")
    ax.set_ylabel("")
    ax.set_title("Sycophancy Rate by Model and Authority Level")

    fig.tight_layout()
    _save_figure(fig, output_path)


# ---------------------------------------------------------------------------
# Figure 2: Matched vs mismatched domain (grouped bar chart)
# ---------------------------------------------------------------------------

def figure2_matched_vs_mismatched(data: Dict, output_path: str) -> None:
    """Grouped bar chart: matched vs mismatched flip rates per model.

    Args:
        data: Dict mapping model_name -> {
            'matched': {'rate', 'ci_low', 'ci_high'},
            'mismatched': {'rate', 'ci_low', 'ci_high'},
            'p_value': float (optional),
        }.
        output_path: Path to save the figure.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(MODEL_ORDER))
    width = 0.35

    matched_rates = []
    mismatched_rates = []
    matched_errors = []
    mismatched_errors = []
    p_values = []

    for model in MODEL_ORDER:
        model_data = data.get(model, {})

        m_data = model_data.get("matched", {})
        mm_data = model_data.get("mismatched", {})

        m_rate = m_data.get("rate", 0.0)
        mm_rate = mm_data.get("rate", 0.0)

        m_ci_low = m_data.get("ci_low", m_rate)
        m_ci_high = m_data.get("ci_high", m_rate)
        mm_ci_low = mm_data.get("ci_low", mm_rate)
        mm_ci_high = mm_data.get("ci_high", mm_rate)

        matched_rates.append(m_rate)
        mismatched_rates.append(mm_rate)
        matched_errors.append([m_rate - m_ci_low, m_ci_high - m_rate])
        mismatched_errors.append([mm_rate - mm_ci_low, mm_ci_high - mm_rate])
        p_values.append(model_data.get("p_value", 1.0))

    matched_rates = np.array(matched_rates)
    mismatched_rates = np.array(mismatched_rates)
    matched_errors = np.array(matched_errors).T
    mismatched_errors = np.array(mismatched_errors).T

    bars1 = ax.bar(
        x - width / 2, matched_rates, width,
        label="Matched Domain",
        color="#0072B2",
        yerr=matched_errors,
        capsize=3,
        error_kw={"linewidth": 0.8},
    )
    bars2 = ax.bar(
        x + width / 2, mismatched_rates, width,
        label="Mismatched Domain",
        color="#D55E00",
        yerr=mismatched_errors,
        capsize=3,
        error_kw={"linewidth": 0.8},
    )

    # Add significance stars
    for i, p in enumerate(p_values):
        stars = _significance_stars(p)
        if stars != "ns":
            max_height = max(
                matched_rates[i] + matched_errors[1, i],
                mismatched_rates[i] + mismatched_errors[1, i],
            )
            ax.text(
                x[i], max_height + 0.02, stars,
                ha="center", va="bottom", fontsize=10, fontweight="bold",
            )

    ax.set_xlabel("Model")
    ax.set_ylabel("Flip Rate (High Authority)")
    ax.set_title("Domain-Matched vs Mismatched Persona Effect")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [MODEL_DISPLAY_NAMES.get(m, m) for m in MODEL_ORDER],
        rotation=20, ha="right",
    )
    ax.legend(loc="upper right")
    ax.set_ylim(0, min(1.0, max(matched_rates.max(), mismatched_rates.max()) * 1.4))

    fig.tight_layout()
    _save_figure(fig, output_path)


# ---------------------------------------------------------------------------
# Figure 3: Probe accuracy curves (3 probes x 5 models)
# ---------------------------------------------------------------------------

def figure3_probe_accuracy_curves(data: Dict, output_path: str) -> None:
    """3x5 subplot grid of probe accuracy vs layer.

    Args:
        data: Dict mapping model_name -> {
            probe_name -> {
                'layer_results': [{'accuracy': float}, ...],
                'peak_layer': int,
                'peak_accuracy': float,
            }
        }.
        output_path: Path to save the figure.
    """
    fig, axes = plt.subplots(
        len(PROBE_NAMES), len(MODEL_ORDER),
        figsize=(14, 7),
        sharex=False,
        sharey=True,
    )

    for col, model in enumerate(MODEL_ORDER):
        model_data = data.get(model, {})

        for row, probe in enumerate(PROBE_NAMES):
            ax = axes[row, col]
            probe_data = model_data.get(probe, {})

            if "error" in probe_data or "layer_results" not in probe_data:
                ax.text(
                    0.5, 0.5, "N/A",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=9, color="gray",
                )
                ax.set_xlim(0, 1)
                ax.set_ylim(0.4, 1.0)
            else:
                layer_results = probe_data["layer_results"]
                accuracies = [r["accuracy"] for r in layer_results]
                n_layers = len(accuracies)
                layers = np.arange(n_layers)

                ax.plot(layers, accuracies, linewidth=1.2, color="#0072B2")

                # Chance line
                ax.axhline(
                    y=0.5, color="gray", linestyle="--",
                    linewidth=0.8, alpha=0.7,
                )

                # Peak layer indicator
                peak_layer = probe_data.get("peak_layer", np.argmax(accuracies))
                peak_acc = probe_data.get("peak_accuracy", max(accuracies))
                ax.axvline(
                    x=peak_layer, color="#D55E00", linestyle="--",
                    linewidth=0.8, alpha=0.7,
                )
                ax.plot(
                    peak_layer, peak_acc, "o",
                    color="#D55E00", markersize=4, zorder=5,
                )

                ax.set_xlim(0, n_layers - 1)
                ax.set_ylim(0.4, 1.0)

            # Labels
            if row == 0:
                ax.set_title(MODEL_DISPLAY_NAMES.get(model, model), fontsize=9)
            if row == len(PROBE_NAMES) - 1:
                ax.set_xlabel("Layer", fontsize=8)
            if col == 0:
                ax.set_ylabel(
                    f"{PROBE_DISPLAY[probe]}\nAccuracy",
                    fontsize=8,
                )

            ax.tick_params(labelsize=7)

    fig.suptitle("Probe Accuracy Across Layers", fontsize=12, y=1.02)
    fig.tight_layout()
    _save_figure(fig, output_path)


# ---------------------------------------------------------------------------
# Figure 4: Cosine similarity matrices (5 side-by-side 3x3 heatmaps)
# ---------------------------------------------------------------------------

def figure4_cosine_similarity_matrices(data: Dict, output_path: str) -> None:
    """5 side-by-side 3x3 heatmaps of cosine similarity between probe directions.

    Args:
        data: Dict mapping model_name -> 3x3 numpy array (or nested list)
              of cosine similarities. Row/col order: authority, sycophancy,
              confidence.
        output_path: Path to save the figure.
    """
    n_models = len(MODEL_ORDER)
    fig, axes = plt.subplots(1, n_models, figsize=(14, 3.2))

    labels = [DIRECTION_SHORT[p] for p in PROBE_NAMES]

    for idx, model in enumerate(MODEL_ORDER):
        ax = axes[idx]
        sim_matrix = data.get(model, np.eye(3))
        if not isinstance(sim_matrix, np.ndarray):
            sim_matrix = np.array(sim_matrix)

        sns.heatmap(
            sim_matrix,
            annot=True,
            fmt=".2f",
            cmap="RdBu_r",
            center=0,
            vmin=-1,
            vmax=1,
            linewidths=0.5,
            linecolor="white",
            xticklabels=labels,
            yticklabels=labels if idx == 0 else False,
            cbar=idx == n_models - 1,
            cbar_kws={"shrink": 0.8, "label": "Cosine Sim."} if idx == n_models - 1 else {},
            ax=ax,
            square=True,
            annot_kws={"fontsize": 8},
        )
        ax.set_title(MODEL_DISPLAY_NAMES.get(model, model), fontsize=9)
        ax.tick_params(labelsize=7)

    fig.suptitle(
        "Cosine Similarity Between Probe Directions (Peak Layers)",
        fontsize=11, y=1.05,
    )
    fig.tight_layout()
    _save_figure(fig, output_path)


# ---------------------------------------------------------------------------
# Figure 5: Layer trajectories — cos(w_auth, w_syc) across layers
# ---------------------------------------------------------------------------

def figure5_layer_trajectories(data: Dict, output_path: str) -> None:
    """Line plot of cos(w_authority, w_sycophancy) across normalized layers.

    Args:
        data: Dict mapping model_name -> {
            'layers_normalized': [float, ...],  # 0-1 range
            'cosine_similarities': [float, ...],
        }.
        output_path: Path to save the figure.
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for idx, model in enumerate(MODEL_ORDER):
        model_data = data.get(model, {})
        layers_norm = model_data.get("layers_normalized", [])
        cos_sims = model_data.get("cosine_similarities", [])

        if not layers_norm or not cos_sims:
            continue

        color = colors[idx % len(colors)]
        ax.plot(
            layers_norm, cos_sims,
            label=MODEL_DISPLAY_NAMES.get(model, model),
            color=color,
            linewidth=1.5,
        )

    ax.axhline(y=0, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("Normalized Layer Position")
    ax.set_ylabel(r"$\cos(\mathbf{w}_{\mathrm{auth}}, \mathbf{w}_{\mathrm{syc}})$")
    ax.set_title("Authority--Sycophancy Direction Alignment Across Layers")
    ax.set_xlim(0, 1)
    ax.legend(loc="best", fontsize=8)

    fig.tight_layout()
    _save_figure(fig, output_path)


# ---------------------------------------------------------------------------
# Figure 6: Steering Pareto front — alpha vs sycophancy gap
# ---------------------------------------------------------------------------

def figure6_steering_pareto(data: Dict, output_path: str) -> None:
    """Line plot per model: alpha vs persona sycophancy gap with bootstrap CIs.

    Args:
        data: Dict mapping model_name -> {
            direction_name -> {
                'alphas': [float, ...],
                'sycophancy_gaps': [float, ...],
                'ci_low': [float, ...],   # optional bootstrap lower bounds
                'ci_high': [float, ...],  # optional bootstrap upper bounds
            }
        }.
        output_path: Path to save the figure.
    """
    direction_styles = {
        "authority": {"linestyle": "-", "marker": "o"},
        "sycophancy": {"linestyle": "--", "marker": "s"},
        "confidence": {"linestyle": ":", "marker": "^"},
    }

    n_models = len(MODEL_ORDER)
    fig, axes = plt.subplots(1, n_models, figsize=(14, 3.5), sharey=True)

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for col, model in enumerate(MODEL_ORDER):
        ax = axes[col]
        model_data = data.get(model, {})

        for dir_idx, direction in enumerate(PROBE_NAMES):
            dir_data = model_data.get(direction, {})
            alphas = dir_data.get("alphas", [])
            gaps = dir_data.get("sycophancy_gaps", [])

            if not alphas or not gaps:
                continue

            color = colors[dir_idx % len(colors)]
            style = direction_styles.get(direction, {})

            ax.plot(
                alphas, gaps,
                label=DIRECTION_SHORT[direction] if col == 0 else None,
                color=color,
                linewidth=1.3,
                markersize=4,
                **style,
            )

            # Add bootstrap CI shading if available
            ci_low = dir_data.get("ci_low", [])
            ci_high = dir_data.get("ci_high", [])
            if ci_low and ci_high and len(ci_low) == len(alphas):
                ax.fill_between(
                    alphas, ci_low, ci_high,
                    color=color, alpha=0.15,
                )

        ax.set_xlabel(r"$\alpha$")
        if col == 0:
            ax.set_ylabel("Persona Sycophancy Gap")
        ax.set_title(MODEL_DISPLAY_NAMES.get(model, model), fontsize=9)
        ax.axhline(y=0, color="gray", linestyle=":", linewidth=0.6, alpha=0.5)

    # Single legend for all subplots
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles, labels,
            loc="upper center",
            ncol=len(PROBE_NAMES),
            fontsize=8,
            bbox_to_anchor=(0.5, 1.08),
        )

    fig.suptitle("Steering Effect on Sycophancy Gap", fontsize=11, y=1.12)
    fig.tight_layout()
    _save_figure(fig, output_path)


# ---------------------------------------------------------------------------
# Figure 7: Random baseline comparison (grouped bar chart)
# ---------------------------------------------------------------------------

def figure7_random_baseline(data: Dict, output_path: str) -> None:
    """Grouped bar chart: unsteered vs authority-steered vs random-steered flip rates.

    Demonstrates that only the authority direction causally reduces sycophancy,
    while a random direction at the same layer/alpha has no effect.

    Args:
        data: Dict mapping model_name -> {
            'unsteered': {'rate', 'ci_low', 'ci_high'},
            'authority': {'rate', 'ci_low', 'ci_high'},
            'random': {'rate', 'ci_low', 'ci_high'},
        }.
        output_path: Path to save the figure.
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    models_with_data = [m for m in MODEL_ORDER if m in data]
    if not models_with_data:
        logger.warning("No data for figure 7, skipping")
        plt.close(fig)
        return

    x = np.arange(len(models_with_data))
    width = 0.25

    conditions = ["unsteered", "authority", "random"]
    condition_labels = ["Unsteered", r"Authority ($\alpha$=1.5)", "Random Dir."]
    condition_colors = ["#999999", "#0072B2", "#D55E00"]

    for i, (cond, label, color) in enumerate(zip(conditions, condition_labels, condition_colors)):
        rates = []
        errors_low = []
        errors_high = []

        for model in models_with_data:
            cond_data = data[model].get(cond, {})
            rate = cond_data.get("rate", 0.0)
            ci_lo = cond_data.get("ci_low", rate)
            ci_hi = cond_data.get("ci_high", rate)
            rates.append(rate)
            errors_low.append(rate - ci_lo)
            errors_high.append(ci_hi - rate)

        rates = np.array(rates)
        yerr = np.array([errors_low, errors_high])

        ax.bar(
            x + (i - 1) * width, rates, width,
            label=label, color=color,
            yerr=yerr, capsize=3,
            error_kw={"linewidth": 0.8},
        )

    ax.set_xlabel("Model")
    ax.set_ylabel("Flip Rate")
    ax.set_title("Steering Specificity: Authority Direction vs Random Baseline")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [MODEL_DISPLAY_NAMES.get(m, m) for m in models_with_data],
        rotation=20, ha="right",
    )
    ax.legend(loc="upper right", fontsize=8)
    ax.set_ylim(0, min(1.0, ax.get_ylim()[1] * 1.3))

    fig.tight_layout()
    _save_figure(fig, output_path)


# ---------------------------------------------------------------------------
# Figure 8: Question-level sensitivity analysis
# ---------------------------------------------------------------------------

def figure8_question_sensitivity(data: Dict, output_path: str) -> None:
    """Two-panel figure: (a) histogram of per-question flip rates,
    (b) leave-5-out bootstrap distribution of overall flip rate.

    Demonstrates that results are not driven by a handful of outlier questions.

    Args:
        data: Dict with keys:
            'per_question_rates': [{'question_id', 'flip_rate', ...}, ...],
            'top_outlier_qids': [str, ...],
            'leave5out_bootstrap': {
                'overall_flip_rate_mean', 'overall_flip_rate_std',
                'overall_flip_rate_ci95': [lo, hi],
                'auth_effect_mean', 'auth_effect_std',
                'auth_effect_ci95': [lo, hi],
            },
        output_path: Path to save the figure.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Panel (a): Histogram of per-question flip rates
    per_q = data.get("per_question_rates", [])
    if per_q:
        flip_rates = np.array([q["flip_rate"] for q in per_q])
        outlier_qids = set(data.get("top_outlier_qids", []))

        # Color outliers differently
        colors = [
            "#D55E00" if q["question_id"] in outlier_qids else "#0072B2"
            for q in per_q
        ]

        # Sort by flip rate for readability
        sorted_indices = np.argsort(flip_rates)[::-1]
        sorted_rates = flip_rates[sorted_indices]
        sorted_colors = [colors[i] for i in sorted_indices]

        ax1.bar(range(len(sorted_rates)), sorted_rates, color=sorted_colors, width=0.8)
        ax1.axhline(
            y=np.median(flip_rates), color="gray", linestyle="--",
            linewidth=0.8, label=f"Median = {np.median(flip_rates):.3f}",
        )
        ax1.set_xlabel("Question (ranked)")
        ax1.set_ylabel("Flip Rate")
        ax1.set_title("(a) Per-Question Flip Rates")
        ax1.legend(fontsize=8)

        # Add outlier annotation
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="#D55E00", label="Top-5 outliers"),
            Patch(facecolor="#0072B2", label="Other questions"),
        ]
        ax1.legend(handles=legend_elements, fontsize=7, loc="upper right")

    # Panel (b): Bootstrap distribution of authority effect
    bootstrap = data.get("leave5out_bootstrap", {})
    if bootstrap:
        # We don't have the full distribution, but we can show the CI as a
        # summary visual with mean + CI bar
        effect_mean = bootstrap.get("auth_effect_mean", 0)
        effect_ci = bootstrap.get("auth_effect_ci95", [effect_mean, effect_mean])
        flip_mean = bootstrap.get("overall_flip_rate_mean", 0)
        flip_ci = bootstrap.get("overall_flip_rate_ci95", [flip_mean, flip_mean])

        metrics = ["Overall\nFlip Rate", "Authority\nEffect"]
        means = [flip_mean, effect_mean]
        ci_lows = [flip_ci[0], effect_ci[0]]
        ci_highs = [flip_ci[1], effect_ci[1]]

        x_pos = np.arange(len(metrics))
        bar_colors = ["#0072B2", "#D55E00"]

        for i in range(len(metrics)):
            ax2.bar(x_pos[i], means[i], color=bar_colors[i], width=0.5, alpha=0.8)
            ax2.errorbar(
                x_pos[i], means[i],
                yerr=[[means[i] - ci_lows[i]], [ci_highs[i] - means[i]]],
                fmt="none", color="black", capsize=5, linewidth=1.2,
            )
            ax2.text(
                x_pos[i], ci_highs[i] + 0.005,
                f"{means[i]:.3f}\n[{ci_lows[i]:.3f}, {ci_highs[i]:.3f}]",
                ha="center", va="bottom", fontsize=8,
            )

        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(metrics)
        ax2.set_ylabel("Rate")
        ax2.set_title("(b) Leave-5-Out Bootstrap (200 iter.)")
        ax2.set_ylim(0, max(ci_highs) * 1.4)

    fig.suptitle("Question-Level Sensitivity Analysis", fontsize=11, y=1.02)
    fig.tight_layout()
    _save_figure(fig, output_path)
