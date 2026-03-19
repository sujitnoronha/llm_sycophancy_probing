"""CLI entry point for MMLU capability evaluation.

Runs unsteered and steered MMLU evaluation to verify that activation
steering does not catastrophically degrade general model capabilities.

Usage:
    python -m src.mmlu.run                              # All models
    python -m src.mmlu.run --model llama3_8b             # Single model
    python -m src.mmlu.run --config configs/base.yaml    # Custom config
"""

import json
from pathlib import Path

import click
import torch
from dotenv import load_dotenv
from loguru import logger

from src.core.config import load_config, get_model_config
from src.core.models import load_model_context
from src.core.utils import (
    set_all_seeds,
    setup_logging,
    ensure_output_dir,
    check_checkpoint,
    save_checkpoint,
)
from src.mmlu.eval_mmlu import load_mmlu_questions, evaluate_mmlu, evaluate_mmlu_steered

load_dotenv()

MODULE_NAME = "mmlu"


def _load_authority_direction(output_dir: str, model_name: str) -> dict:
    """Load the authority direction vector and peak layer from probing outputs.

    Returns:
        Dict with 'direction' (torch.Tensor) and 'peak_layer' (int).
    """
    weight_path = Path(output_dir) / "probing" / model_name / "weight_vectors.pt"
    if not weight_path.exists():
        raise FileNotFoundError(
            f"Weight vectors not found at {weight_path}. "
            f"Run probing first: python -m src.probing.run --model {model_name}"
        )

    data = torch.load(str(weight_path), weights_only=False)
    direction = data["w_authority"]  # [n_layers, d_model]
    peak_layer = int(data["peak_layer_authority"])

    # Extract the direction at the peak layer
    direction_vec = direction[peak_layer]  # [d_model]

    # Ensure unit norm
    norm = direction_vec.norm()
    if norm > 0:
        direction_vec = direction_vec / norm

    logger.info(
        f"Loaded authority direction for {model_name}: "
        f"peak_layer={peak_layer}, ||w||={norm:.4f}"
    )
    return {
        "direction": direction_vec,
        "peak_layer": peak_layer,
    }


@click.command()
@click.option("--config", default="configs/base.yaml", help="Path to config YAML")
@click.option("--model", "model_name", default=None, help="Run for a single model")
def main(config: str, model_name: str):
    """Run MMLU capability evaluation (unsteered + steered)."""
    cfg = load_config(config)
    setup_logging(cfg.log_level)
    set_all_seeds(cfg.seed)

    # Load MMLU questions
    questions = load_mmlu_questions()
    if cfg.mmlu.n_questions < len(questions):
        questions = questions[: cfg.mmlu.n_questions]
    logger.info(f"Using {len(questions)} MMLU questions ({cfg.mmlu.n_shot}-shot)")

    models_to_run = (
        [get_model_config(cfg, model_name)] if model_name else cfg.models
    )

    for model_cfg in models_to_run:
        if check_checkpoint(cfg.output_dir, MODULE_NAME, model_cfg.name):
            logger.info(f"Skipping {model_cfg.name} (checkpoint exists)")
            continue

        model_output_dir = ensure_output_dir(cfg.output_dir, f"mmlu/{model_cfg.name}")
        all_results = {"model_name": model_cfg.name, "n_shot": cfg.mmlu.n_shot}

        # --- Phase 1: Unsteered MMLU ---
        logger.info(f"Running unsteered MMLU for {model_cfg.name}")
        with load_model_context(model_cfg, cfg, need_hooks=False) as (model, tokenizer):
            unsteered = evaluate_mmlu(
                model=model,
                tokenizer=tokenizer,
                questions=questions,
                n_shot=cfg.mmlu.n_shot,
                max_new_tokens=5,
            )
            all_results["unsteered"] = {
                "accuracy": unsteered["accuracy"],
                "n_correct": unsteered["n_correct"],
                "n_total": unsteered["n_total"],
                "per_question": unsteered["per_question"],
            }

        # --- Phase 2: Steered MMLU (authority direction at each alpha) ---
        try:
            auth_data = _load_authority_direction(cfg.output_dir, model_cfg.name)
        except FileNotFoundError as e:
            logger.warning(f"Skipping steered MMLU for {model_cfg.name}: {e}")
            all_results["steered"] = {"error": str(e)}
            # Save what we have and continue
            _save_results(all_results, model_output_dir)
            save_checkpoint(cfg.output_dir, MODULE_NAME, model_cfg.name)
            continue

        direction = auth_data["direction"]
        peak_layer = auth_data["peak_layer"]
        steered_results = {}

        with load_model_context(model_cfg, cfg, need_hooks=False) as (model, tokenizer):
            for alpha in cfg.steering.alpha_values:
                logger.info(
                    f"Running steered MMLU for {model_cfg.name} "
                    f"(alpha={alpha}, layer={peak_layer})"
                )
                steered = evaluate_mmlu_steered(
                    model=model,
                    tokenizer=tokenizer,
                    questions=questions,
                    direction=direction,
                    alpha=alpha,
                    target_layer=peak_layer,
                    n_shot=cfg.mmlu.n_shot,
                    max_new_tokens=5,
                )
                steered_results[str(alpha)] = {
                    "accuracy": steered["accuracy"],
                    "n_correct": steered["n_correct"],
                    "n_total": steered["n_total"],
                    "steering_config": steered["steering_config"],
                    "per_question": steered["per_question"],
                }

        all_results["steered"] = steered_results

        # --- Phase 3: Compare and summarize ---
        comparison = _build_comparison(all_results)
        all_results["comparison"] = comparison

        # Save
        _save_results(all_results, model_output_dir)
        save_checkpoint(cfg.output_dir, MODULE_NAME, model_cfg.name)
        logger.info(f"MMLU evaluation complete for {model_cfg.name}")

    click.echo("MMLU capability evaluation complete.")


def _build_comparison(results: dict) -> dict:
    """Build a comparison summary between unsteered and steered accuracy."""
    unsteered_acc = results.get("unsteered", {}).get("accuracy", 0.0)
    comparison = {"unsteered_accuracy": unsteered_acc}

    steered = results.get("steered", {})
    if isinstance(steered, dict) and "error" not in steered:
        for alpha_str, steered_data in steered.items():
            acc = steered_data.get("accuracy", 0.0)
            drop = unsteered_acc - acc
            comparison[f"alpha_{alpha_str}"] = {
                "steered_accuracy": acc,
                "accuracy_drop": round(drop, 4),
                "relative_drop_pct": round(100 * drop / max(unsteered_acc, 1e-9), 2),
            }

    return comparison


def _save_results(results: dict, output_dir: Path) -> None:
    """Save MMLU results to JSON."""
    output_path = output_dir / "mmlu_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"MMLU results saved to {output_path}")


if __name__ == "__main__":
    main()
