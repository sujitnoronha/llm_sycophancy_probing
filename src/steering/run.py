"""CLI entry point for activation steering experiments.

Iterates over models, runs the full steering sweep (3 directions x 6 alphas),
saves steered conversations and summary metrics. Includes checkpointing
so interrupted runs can be resumed.

Usage:
    python -m src.steering.run                                 # All models
    python -m src.steering.run --model llama3_8b               # Single model
    python -m src.steering.run --config configs/custom.yaml    # Custom config
"""

import json
from pathlib import Path

import click
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
from src.steering.steer import run_full_sweep, run_steered_eval, generate_random_direction, _load_probe_directions, _load_conversations

load_dotenv()

MODULE_NAME = "steering"


def save_steered_conversations(conversations: list, output_path: Path) -> None:
    """Save steered conversations to JSONL format.

    Args:
        conversations: List of conversation dicts with steered Turn 4.
        output_path: Path to the output JSONL file.
    """
    with open(output_path, "w") as f:
        for conv in conversations:
            f.write(json.dumps(conv, default=str) + "\n")
    logger.info(f"Saved {len(conversations)} steered conversations to {output_path}")


@click.command()
@click.option("--config", default="configs/base.yaml", help="Path to config YAML")
@click.option("--model", "model_name", default=None, help="Run for a single model")
@click.option("--random-baseline", is_flag=True, help="Run random direction baseline (control experiment)")
def main(config: str, model_name: str, random_baseline: bool):
    """Run activation steering experiments across models.

    For each model, loads the TransformerLens model once, then sweeps
    over all (direction, alpha) combinations specified in the config.
    Results are saved per-model with checkpointing for resumability.
    """
    cfg = load_config(config)
    setup_logging(cfg.log_level)
    set_all_seeds(cfg.seed)

    models_to_run = (
        [get_model_config(cfg, model_name)] if model_name else cfg.models
    )

    logger.info(
        f"Steering experiment: {len(models_to_run)} models, "
        f"{len(cfg.steering.directions)} directions, "
        f"{len(cfg.steering.alpha_values)} alpha values"
    )

    all_summaries = {}

    for model_cfg in models_to_run:
        # Check for existing checkpoint
        if check_checkpoint(cfg.output_dir, MODULE_NAME, model_cfg.name):
            logger.info(f"Skipping {model_cfg.name} (checkpoint exists)")

            # Load existing results for the final summary
            model_dir = Path(cfg.output_dir) / MODULE_NAME / model_cfg.name
            summary_path = model_dir / "steering_summary.json"
            if summary_path.exists():
                with open(summary_path) as f:
                    all_summaries[model_cfg.name] = json.load(f)

            continue

        logger.info(f"Starting steering sweep for {model_cfg.name}")
        model_output_dir = ensure_output_dir(cfg.output_dir, f"{MODULE_NAME}/{model_cfg.name}")

        try:
            # Run the full sweep (loads model, runs all conditions, unloads)
            sweep_results = run_full_sweep(
                model_name=model_cfg.name,
                config=cfg,
            )

            # Save the full sweep results
            with open(model_output_dir / "steering_summary.json", "w") as f:
                json.dump(sweep_results, f, indent=2, default=str)

            # Save per-condition steered conversation files
            for condition_key, condition_data in sweep_results.get("conditions", {}).items():
                # The steered conversations are in the condition data if available
                conv_path = model_output_dir / f"steered_{condition_key}.jsonl"

                # Note: run_full_sweep returns summary metrics, not raw conversations.
                # Raw conversations are saved inside run_steered_eval if needed.
                # Here we save the condition metadata for reference.
                condition_meta_path = model_output_dir / f"condition_{condition_key}.json"
                with open(condition_meta_path, "w") as f:
                    json.dump(condition_data, f, indent=2, default=str)

            all_summaries[model_cfg.name] = sweep_results

            # Mark this model as complete
            save_checkpoint(cfg.output_dir, MODULE_NAME, model_cfg.name)
            logger.info(f"Steering sweep complete for {model_cfg.name}")

        except FileNotFoundError as e:
            logger.error(
                f"Cannot run steering for {model_cfg.name}: {e}. "
                f"Ensure probing and behavioral eval have been run first."
            )
            continue
        except Exception as e:
            logger.error(f"Steering failed for {model_cfg.name}: {e}", exc_info=True)
            # Do NOT save checkpoint on failure so the model can be retried
            continue

    # --- Random direction baseline (reviewer concern: causal specificity) ---
    if random_baseline:
        from src.core.models import load_model_context

        logger.info("Running random direction baseline")
        random_seed = 12345

        for model_cfg in models_to_run:
            model_output_dir = ensure_output_dir(
                cfg.output_dir, f"{MODULE_NAME}/{model_cfg.name}"
            )
            random_conv_path = model_output_dir / "steered_random_alpha1.5.jsonl"

            if random_conv_path.exists():
                logger.info(f"Random baseline exists for {model_cfg.name}, skipping")
                continue

            try:
                # Generate random direction
                random_dir = generate_random_direction(
                    model_cfg.d_model, seed=random_seed
                )

                # Load conversations
                conversations = _load_conversations(cfg.output_dir, model_cfg.name)

                # Get peak layer from authority probe (same layer for fair comparison)
                probe_directions = _load_probe_directions(
                    cfg.output_dir, model_cfg.name
                )
                target_layer = probe_directions["authority"]["peak_layer"]

                logger.info(
                    f"Random baseline for {model_cfg.name}: "
                    f"alpha=1.5, layer={target_layer}, d_model={model_cfg.d_model}"
                )

                # Load model and run steered eval
                with load_model_context(model_cfg, cfg, need_hooks=True) as (
                    model,
                    tokenizer,
                ):
                    steered_convs = run_steered_eval(
                        model=model,
                        tokenizer=tokenizer,
                        conversations=conversations,
                        direction=random_dir,
                        alpha=1.5,
                        target_layer=target_layer,
                        config=cfg,
                    )

                # Save
                with open(random_conv_path, "w") as f:
                    for conv in steered_convs:
                        f.write(json.dumps(conv, default=str) + "\n")
                logger.info(
                    f"Saved {len(steered_convs)} random-baseline convs for {model_cfg.name}"
                )

            except FileNotFoundError as e:
                logger.error(f"Cannot run random baseline for {model_cfg.name}: {e}")
            except Exception as e:
                logger.error(
                    f"Random baseline failed for {model_cfg.name}: {e}", exc_info=True
                )

    # Save aggregate summary across all models
    aggregate_dir = ensure_output_dir(cfg.output_dir, MODULE_NAME)
    aggregate = {
        "models_completed": list(all_summaries.keys()),
        "config": {
            "directions": cfg.steering.directions,
            "alpha_values": cfg.steering.alpha_values,
            "layer_selection": cfg.steering.layer_selection,
        },
        "per_model": {},
    }

    for model_name_key, sweep in all_summaries.items():
        model_summary = {
            "n_conversations": sweep.get("n_conversations", 0),
            "n_conditions": len(sweep.get("conditions", {})),
            "conditions": {},
        }
        for cond_key, cond_data in sweep.get("conditions", {}).items():
            model_summary["conditions"][cond_key] = {
                "direction": cond_data.get("direction"),
                "alpha": cond_data.get("alpha"),
                "target_layer": cond_data.get("target_layer"),
                "n_generated": cond_data.get("n_generated", 0),
                "metrics": cond_data.get("metrics", {}),
            }
        aggregate["per_model"][model_name_key] = model_summary

    with open(aggregate_dir / "steering_aggregate_summary.json", "w") as f:
        json.dump(aggregate, f, indent=2, default=str)

    logger.info(
        f"Aggregate summary saved to {aggregate_dir / 'steering_aggregate_summary.json'}"
    )
    click.echo("Activation steering experiments complete.")


if __name__ == "__main__":
    main()
