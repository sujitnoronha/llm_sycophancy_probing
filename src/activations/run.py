"""CLI entry point for activation collection.

Usage:
    python -m src.activations.run                              # All models
    python -m src.activations.run --model llama3_8b            # Single model
"""

import json

import click
import pandas as pd
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
from src.activations.collector import (
    select_probing_subset,
    collect_activations_transformerlens,
    save_activations,
)

load_dotenv()

MODULE_NAME = "activations"


@click.command()
@click.option("--config", default="configs/base.yaml", help="Path to config YAML")
@click.option("--model", "model_name", default=None, help="Run for a single model")
def main(config: str, model_name: str):
    cfg = load_config(config)
    setup_logging(cfg.log_level)
    set_all_seeds(cfg.seed)

    output_dir = ensure_output_dir(cfg.output_dir, MODULE_NAME)

    models_to_run = (
        [get_model_config(cfg, model_name)] if model_name else cfg.models
    )

    for model_cfg in models_to_run:
        if check_checkpoint(cfg.output_dir, MODULE_NAME, model_cfg.name):
            logger.info(f"Skipping {model_cfg.name} (checkpoint exists)")
            continue

        # Load completed conversations for this model
        conv_path = f"{cfg.output_dir}/behavioral/{model_cfg.name}_conversations.jsonl"
        conversations = []
        with open(conv_path) as f:
            for line in f:
                if line.strip():
                    conversations.append(json.loads(line))

        # Load scores for sycophancy labels
        scores_path = f"{cfg.output_dir}/behavioral/{model_cfg.name}_scores.csv"
        scores_df = pd.read_csv(scores_path)

        # Select probing subset (200 prompts)
        key_conditions = [kc.model_dump() for kc in cfg.probing.key_conditions]
        probing_subset = select_probing_subset(
            conversations, scores_df, key_conditions
        )

        if len(probing_subset) == 0:
            logger.error(f"No probing prompts selected for {model_cfg.name}")
            continue

        logger.info(
            f"Collecting activations for {model_cfg.name}: "
            f"{len(probing_subset)} prompts, {model_cfg.n_layers} layers"
        )

        # Load model with hooks enabled
        with load_model_context(model_cfg, cfg, need_hooks=True) as (model, tokenizer):
            activations, labels, metadata = collect_activations_transformerlens(
                model=model,
                tokenizer=tokenizer,
                probing_subset=probing_subset,
                n_layers=model_cfg.n_layers,
                d_model=model_cfg.d_model,
                system_prompt=cfg.experiment.system_prompt,
            )

        # Save
        output_path = output_dir / f"{model_cfg.name}_activations.pt"
        metadata["model_name"] = model_cfg.name
        save_activations(activations, labels, metadata, str(output_path))
        save_checkpoint(cfg.output_dir, MODULE_NAME, model_cfg.name)

    click.echo("Activation collection complete.")


if __name__ == "__main__":
    main()
