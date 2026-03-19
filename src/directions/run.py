"""CLI entry point for probe direction analysis.

Loads all models' probe results from outputs/probing/{model}/ and runs
the full suite of mechanistic direction analyses: cosine similarity matrices,
layer trajectories, subspace decompositions, cross-model comparisons,
and permutation significance tests.

Usage:
    python -m src.directions.run                               # All models
    python -m src.directions.run --model llama3_8b             # Single model
    python -m src.directions.run --permtest                    # With permutation tests
"""

import json
from pathlib import Path

import click
import numpy as np
import torch
from dotenv import load_dotenv
from loguru import logger

from src.core.config import load_config, get_model_config
from src.core.utils import (
    set_all_seeds,
    setup_logging,
    ensure_output_dir,
    check_checkpoint,
    save_checkpoint,
)
from src.directions.analysis import (
    compute_cosine_similarity_matrix,
    compute_layer_trajectory,
    subspace_decomposition,
    cross_model_direction_comparison,
    permutation_test_cosine,
)

load_dotenv()

MODULE_NAME = "directions"


def load_probe_results(output_dir: str, model_name: str) -> dict:
    """Load probe weight vectors and summary for a single model.

    Args:
        output_dir: Base output directory (e.g., 'outputs').
        model_name: Model name (e.g., 'llama3_8b').

    Returns:
        Dict with 'weight_vectors' (torch tensors), 'summary' (JSON dict),
        and derived numpy arrays.

    Raises:
        FileNotFoundError: If required probe output files are missing.
    """
    probe_dir = Path(output_dir) / "probing" / model_name

    wv_path = probe_dir / "weight_vectors.pt"
    summary_path = probe_dir / "probe_summary.json"

    if not wv_path.exists():
        raise FileNotFoundError(
            f"Weight vectors not found at {wv_path}. Run probing first."
        )
    if not summary_path.exists():
        raise FileNotFoundError(
            f"Probe summary not found at {summary_path}. Run probing first."
        )

    weight_vectors = torch.load(str(wv_path), weights_only=False)
    with open(summary_path) as f:
        summary = json.load(f)

    logger.info(
        f"Loaded probe results for {model_name}: "
        f"{summary.get('n_layers', '?')} layers, "
        f"{summary.get('d_model', '?')} d_model"
    )

    return {
        "weight_vectors_raw": weight_vectors,
        "summary": summary,
    }


def extract_peak_directions(probe_results: dict) -> dict:
    """Extract unit-normalized peak-layer directions from probe results.

    Pulls the weight vector at each probe's peak accuracy layer and
    returns them as numpy arrays keyed by the standard direction names.

    Args:
        probe_results: Output of load_probe_results().

    Returns:
        Dict with 'w_auth', 'w_syc', 'w_conf' numpy arrays of shape [d_model],
        plus 'peak_layers' dict mapping probe name to peak layer index.
    """
    wv = probe_results["weight_vectors_raw"]
    summary = probe_results["summary"]

    directions = {}
    peak_layers = {}

    probe_map = {
        "w_auth": ("w_authority", "authority"),
        "w_syc": ("w_sycophancy", "sycophancy"),
        "w_conf": ("w_confidence", "confidence"),
    }

    for dir_key, (wv_key, probe_name) in probe_map.items():
        if wv_key not in wv:
            logger.warning(f"Missing weight vectors for '{wv_key}', skipping")
            continue

        # Get peak layer from weight_vectors.pt metadata or summary
        peak_layer_key = f"peak_layer_{probe_name}"
        if peak_layer_key in wv:
            peak_layer = int(wv[peak_layer_key])
        elif probe_name in summary and "peak_layer" in summary[probe_name]:
            peak_layer = summary[probe_name]["peak_layer"]
        else:
            # Fallback: use the middle layer
            w_all = wv[wv_key].numpy()
            peak_layer = w_all.shape[0] // 2
            logger.warning(
                f"No peak layer info for {probe_name}, defaulting to layer {peak_layer}"
            )

        w_all = wv[wv_key].numpy()  # [n_layers, d_model]
        w_peak = w_all[peak_layer].astype(np.float64)

        # Ensure unit normalization
        norm = np.linalg.norm(w_peak)
        if norm > 1e-10:
            w_peak = w_peak / norm

        directions[dir_key] = w_peak
        peak_layers[probe_name] = peak_layer

    return {**directions, "peak_layers": peak_layers}


def extract_all_layer_vectors(probe_results: dict) -> dict:
    """Extract weight vectors at all layers as numpy arrays.

    Args:
        probe_results: Output of load_probe_results().

    Returns:
        Dict mapping 'w_authority', 'w_sycophancy', 'w_confidence' to
        numpy arrays of shape [n_layers, d_model].
    """
    wv = probe_results["weight_vectors_raw"]
    vectors = {}

    for key in ["w_authority", "w_sycophancy", "w_confidence"]:
        if key in wv:
            vectors[key] = wv[key].numpy().astype(np.float64)

    return vectors


def run_single_model_analysis(
    model_name: str,
    probe_results: dict,
    model_output_dir: Path,
) -> dict:
    """Run all direction analyses for a single model.

    Args:
        model_name: Name of the model.
        probe_results: Output of load_probe_results().
        model_output_dir: Directory to save results.

    Returns:
        Summary dict of analysis results.
    """
    logger.info(f"Running direction analysis for {model_name}")

    peak_dirs = extract_peak_directions(probe_results)
    all_layer_vecs = extract_all_layer_vectors(probe_results)

    results = {
        "model_name": model_name,
        "peak_layers": peak_dirs.get("peak_layers", {}),
    }

    # --- Analysis 1: Cosine similarity matrix at peak layers ---
    if all(k in peak_dirs for k in ["w_auth", "w_syc", "w_conf"]):
        sim_matrix = compute_cosine_similarity_matrix({
            "w_auth": peak_dirs["w_auth"],
            "w_syc": peak_dirs["w_syc"],
            "w_conf": peak_dirs["w_conf"],
        })
        results["cosine_similarity_matrix"] = sim_matrix.tolist()
        results["cosine_auth_syc"] = float(sim_matrix[0, 1])
        results["cosine_auth_conf"] = float(sim_matrix[0, 2])
        results["cosine_syc_conf"] = float(sim_matrix[1, 2])

        np.save(str(model_output_dir / "cosine_similarity_matrix.npy"), sim_matrix)
        logger.info(f"Saved cosine similarity matrix for {model_name}")
    else:
        logger.warning(f"Cannot compute similarity matrix for {model_name}: missing directions")

    # --- Analysis 2: Layer trajectories ---
    trajectory_pairs = [
        ("w_authority", "w_sycophancy"),
        ("w_authority", "w_confidence"),
        ("w_sycophancy", "w_confidence"),
    ]

    trajectories = {}
    for probe_a, probe_b in trajectory_pairs:
        if probe_a in all_layer_vecs and probe_b in all_layer_vecs:
            traj = compute_layer_trajectory(all_layer_vecs, probe_a, probe_b)
            pair_name = f"{probe_a}_vs_{probe_b}"
            trajectories[pair_name] = traj.tolist()
            np.save(str(model_output_dir / f"trajectory_{pair_name}.npy"), traj)

    results["trajectories"] = trajectories
    logger.info(f"Computed {len(trajectories)} layer trajectories for {model_name}")

    # --- Analysis 3: Subspace decomposition ---
    if all(k in peak_dirs for k in ["w_auth", "w_syc", "w_conf"]):
        decomp = subspace_decomposition(
            w_syc=peak_dirs["w_syc"],
            w_auth=peak_dirs["w_auth"],
            w_conf=peak_dirs["w_conf"],
        )
        results["subspace_decomposition"] = {
            "variance_auth": decomp["variance_auth"],
            "variance_conf": decomp["variance_conf"],
            "variance_residual": decomp["variance_residual"],
        }

        # Save the component vectors for potential downstream use
        np.savez(
            str(model_output_dir / "subspace_components.npz"),
            component_auth=decomp["component_auth"],
            component_conf_orth=decomp["component_conf_orth"],
            component_residual=decomp["component_residual"],
            w_conf_orthogonalized=decomp["w_conf_orthogonalized"],
        )
        logger.info(f"Saved subspace decomposition for {model_name}")
    else:
        logger.warning(f"Cannot compute subspace decomposition for {model_name}: missing directions")

    return results


def run_permutation_tests_for_model(
    model_name: str,
    output_dir: str,
    model_output_dir: Path,
    probe_results: dict,
    n_permutations: int = 1000,
    seed: int = 42,
) -> dict:
    """Run permutation tests for direction cosine significance.

    Args:
        model_name: Name of the model.
        output_dir: Base output directory.
        model_output_dir: Directory to save results.
        probe_results: Output of load_probe_results().
        n_permutations: Number of permutations per test.
        seed: Random seed.

    Returns:
        Dict of permutation test results.
    """
    # Load activations for this model
    act_path = Path(output_dir) / "activations" / f"{model_name}_activations.pt"
    if not act_path.exists():
        logger.warning(f"Activations not found at {act_path}, skipping permutation tests")
        return {}

    logger.info(f"Loading activations from {act_path} for permutation tests")
    data = torch.load(str(act_path), weights_only=False)
    activations = data["activations"].numpy()
    labels = {k: v.numpy() for k, v in data["labels"].items()}

    summary = probe_results["summary"]
    perm_results = {}

    # Test pairs: authority-sycophancy, authority-confidence, sycophancy-confidence
    test_pairs = [
        ("authority", "sycophancy"),
        ("authority", "confidence"),
        ("sycophancy", "confidence"),
    ]

    for label_a_name, label_b_name in test_pairs:
        if label_a_name not in labels or label_b_name not in labels:
            logger.warning(f"Missing labels for {label_a_name} or {label_b_name}, skipping")
            continue

        # Use the peak layer of probe_a for the test
        if label_a_name in summary and "peak_layer" in summary[label_a_name]:
            layer_idx = summary[label_a_name]["peak_layer"]
        else:
            layer_idx = activations.shape[1] // 2
            logger.warning(f"No peak layer for {label_a_name}, using layer {layer_idx}")

        pair_name = f"{label_a_name}_vs_{label_b_name}"
        logger.info(
            f"Running permutation test: {pair_name} at layer {layer_idx} "
            f"({n_permutations} permutations)"
        )

        perm = permutation_test_cosine(
            activations=activations,
            labels_a=labels[label_a_name],
            labels_b=labels[label_b_name],
            layer_idx=layer_idx,
            n_permutations=n_permutations,
            seed=seed,
        )
        perm_results[pair_name] = perm

        logger.info(
            f"  {pair_name}: observed={perm['observed_cosine']:.4f}, "
            f"p={perm['p_value']:.4f}"
        )

    if perm_results:
        with open(model_output_dir / "permutation_test_cosine.json", "w") as f:
            json.dump(perm_results, f, indent=2)
        logger.info(f"Saved permutation test results for {model_name}")

    return perm_results


@click.command()
@click.option("--config", default="configs/base.yaml", help="Path to config YAML")
@click.option("--model", "model_name", default=None, help="Run for a single model")
@click.option("--permtest", is_flag=True, help="Run permutation tests for cosine significance (slow)")
@click.option("--n-permutations", default=1000, help="Number of permutations for significance tests")
def main(config: str, model_name: str, permtest: bool, n_permutations: int):
    """Run probe direction analysis across all models."""
    cfg = load_config(config)
    setup_logging(cfg.log_level)
    set_all_seeds(cfg.seed)

    models_to_run = (
        [get_model_config(cfg, model_name)] if model_name else cfg.models
    )

    all_results = {}
    directions_by_model = {}

    # --- Phase 1: Per-model analysis ---
    for model_cfg in models_to_run:
        if check_checkpoint(cfg.output_dir, MODULE_NAME, model_cfg.name):
            logger.info(f"Skipping {model_cfg.name} (checkpoint exists)")

            # Still load results for cross-model comparison
            model_dir = Path(cfg.output_dir) / MODULE_NAME / model_cfg.name
            results_path = model_dir / "direction_analysis.json"
            if results_path.exists():
                with open(results_path) as f:
                    all_results[model_cfg.name] = json.load(f)

            # Load weight vectors for cross-model comparison
            try:
                probe_results = load_probe_results(cfg.output_dir, model_cfg.name)
                directions_by_model[model_cfg.name] = extract_all_layer_vectors(probe_results)
            except FileNotFoundError:
                pass

            continue

        try:
            probe_results = load_probe_results(cfg.output_dir, model_cfg.name)
        except FileNotFoundError as e:
            logger.warning(f"Skipping {model_cfg.name}: {e}")
            continue

        model_output_dir = ensure_output_dir(cfg.output_dir, f"{MODULE_NAME}/{model_cfg.name}")

        # Run per-model analyses
        results = run_single_model_analysis(
            model_name=model_cfg.name,
            probe_results=probe_results,
            model_output_dir=model_output_dir,
        )

        # Optional: permutation significance tests
        if permtest:
            perm_results = run_permutation_tests_for_model(
                model_name=model_cfg.name,
                output_dir=cfg.output_dir,
                model_output_dir=model_output_dir,
                probe_results=probe_results,
                n_permutations=n_permutations,
                seed=cfg.seed,
            )
            results["permutation_tests"] = {
                k: {kk: vv for kk, vv in v.items()}
                for k, v in perm_results.items()
            }

        # Save per-model results
        with open(model_output_dir / "direction_analysis.json", "w") as f:
            json.dump(results, f, indent=2, default=str)

        all_results[model_cfg.name] = results
        directions_by_model[model_cfg.name] = extract_all_layer_vectors(probe_results)

        save_checkpoint(cfg.output_dir, MODULE_NAME, model_cfg.name)
        logger.info(f"Direction analysis complete for {model_cfg.name}")

    # --- Phase 2: Cross-model comparison (Llama architecture) ---
    cross_model_dir = ensure_output_dir(cfg.output_dir, MODULE_NAME)

    if "llama3_8b" in directions_by_model and "deepseek_r1_llama8b" in directions_by_model:
        logger.info("Running cross-model direction comparison (Llama architecture)")
        try:
            cross_results = cross_model_direction_comparison(directions_by_model)

            cross_results_serializable = {
                "per_layer_cosine": cross_results["per_layer_cosine"].tolist(),
                "mean_cosine": cross_results["mean_cosine"],
                "peak_cosine": cross_results["peak_cosine"],
                "peak_layer": cross_results["peak_layer"],
                "models_compared": list(cross_results["models_compared"]),
            }

            with open(cross_model_dir / "cross_model_comparison.json", "w") as f:
                json.dump(cross_results_serializable, f, indent=2)

            np.save(
                str(cross_model_dir / "cross_model_per_layer_cosine.npy"),
                cross_results["per_layer_cosine"],
            )

            logger.info("Cross-model comparison saved")
        except (ValueError, KeyError) as e:
            logger.warning(f"Cross-model comparison failed: {e}")
    else:
        available = list(directions_by_model.keys())
        logger.info(
            f"Skipping cross-model comparison: need both llama3_8b and deepseek_r1_llama8b. "
            f"Available: {available}"
        )

    # --- Save aggregate summary ---
    aggregate = {
        "models_analyzed": list(all_results.keys()),
        "per_model_summary": {},
    }

    for model_name, results in all_results.items():
        model_summary = {
            "peak_layers": results.get("peak_layers", {}),
        }
        if "cosine_similarity_matrix" in results:
            model_summary["cosine_auth_syc"] = results.get("cosine_auth_syc")
            model_summary["cosine_auth_conf"] = results.get("cosine_auth_conf")
            model_summary["cosine_syc_conf"] = results.get("cosine_syc_conf")
        if "subspace_decomposition" in results:
            model_summary["subspace_decomposition"] = results["subspace_decomposition"]
        aggregate["per_model_summary"][model_name] = model_summary

    with open(cross_model_dir / "direction_analysis_summary.json", "w") as f:
        json.dump(aggregate, f, indent=2)

    logger.info(f"Aggregate summary saved to {cross_model_dir / 'direction_analysis_summary.json'}")
    click.echo("Direction analysis complete.")


if __name__ == "__main__":
    main()
