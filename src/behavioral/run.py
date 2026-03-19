"""CLI entry point for behavioral evaluation and scoring.

Usage:
    python -m src.behavioral.run                              # Full eval + scoring
    python -m src.behavioral.run --model llama3_8b            # Single model
    python -m src.behavioral.run --score-only                 # Score existing conversations
    python -m src.behavioral.run --generate-annotations       # Generate human annotation CSVs
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
from src.behavioral.eval import load_baseline, run_behavioral_eval
from src.behavioral.scorer import score_conversations
from src.behavioral.human_annotation import generate_annotation_samples

load_dotenv()

MODULE_NAME = "behavioral"


def load_conversations_json(path):
    """Load the generated conversation templates."""
    with open(path) as f:
        return json.load(f)


def load_conversations_jsonl(path):
    """Load completed conversations from JSONL."""
    conversations = []
    with open(path) as f:
        for line in f:
            if line.strip():
                conversations.append(json.loads(line))
    return conversations


def save_conversations_jsonl(conversations, path):
    """Save conversations to JSONL format."""
    with open(path, "w") as f:
        for conv in conversations:
            f.write(json.dumps(conv) + "\n")


@click.command()
@click.option("--config", default="configs/base.yaml", help="Path to config YAML")
@click.option("--model", "model_name", default=None, help="Run for a single model")
@click.option("--score-only", is_flag=True, help="Only score existing conversations")
@click.option("--generate-annotations", is_flag=True, help="Generate human annotation CSVs")
def main(config: str, model_name: str, score_only: bool, generate_annotations: bool):
    cfg = load_config(config)
    setup_logging(cfg.log_level)
    set_all_seeds(cfg.seed)

    output_dir = ensure_output_dir(cfg.output_dir, MODULE_NAME)

    # Load conversation templates
    conversations_path = f"{cfg.output_dir}/dataset/conversations.json"
    conversations = load_conversations_json(conversations_path)

    models_to_run = (
        [get_model_config(cfg, model_name)] if model_name else cfg.models
    )

    if generate_annotations:
        # Generate annotation spreadsheets from all scored conversations
        all_convs = []
        for model_cfg in cfg.models:
            jsonl_path = output_dir / f"{model_cfg.name}_conversations.jsonl"
            if jsonl_path.exists():
                all_convs.extend(load_conversations_jsonl(jsonl_path))

        if not all_convs:
            click.echo("No conversations found. Run behavioral eval first.")
            return

        a1_df, a2_df = generate_annotation_samples(
            all_convs,
            total_sample=cfg.scoring.human_annotation_sample,
            overlap=cfg.scoring.human_annotation_overlap,
            seed=cfg.seed,
        )
        a1_df.to_csv(output_dir / "annotation_sheet_annotator1.csv", index=False)
        a2_df.to_csv(output_dir / "annotation_sheet_annotator2.csv", index=False)
        click.echo(f"Annotation sheets saved to {output_dir}")
        return

    for model_cfg in models_to_run:
        conv_path = output_dir / f"{model_cfg.name}_conversations.jsonl"
        scores_path = output_dir / f"{model_cfg.name}_scores.csv"

        # --- Phase 1: Generate Turn 4 responses ---
        if not score_only:
            if check_checkpoint(cfg.output_dir, f"{MODULE_NAME}_eval", model_cfg.name):
                logger.info(f"Skipping eval for {model_cfg.name} (checkpoint exists)")
            else:
                logger.info(f"Running behavioral eval for {model_cfg.name}")

                baseline = load_baseline(
                    f"{cfg.output_dir}/baseline", model_cfg.name
                )

                with load_model_context(model_cfg, cfg, need_hooks=False) as (model, tok):
                    results = run_behavioral_eval(
                        model=model,
                        tokenizer=tok,
                        conversations=conversations,
                        baseline=baseline,
                        model_name=model_cfg.name,
                        max_new_tokens=cfg.experiment.max_new_tokens_turn4,
                    )

                save_conversations_jsonl(results, conv_path)
                save_checkpoint(cfg.output_dir, f"{MODULE_NAME}_eval", model_cfg.name)

        # --- Phase 2: Score Turn 4 responses with LLM judge ---
        if check_checkpoint(cfg.output_dir, f"{MODULE_NAME}_score", model_cfg.name):
            logger.info(f"Skipping scoring for {model_cfg.name} (checkpoint exists)")
            continue

        if not conv_path.exists():
            logger.error(f"No conversations found at {conv_path}")
            continue

        logger.info(f"Scoring {model_cfg.name} with {cfg.scoring.judge_model}")
        completed_convs = load_conversations_jsonl(conv_path)

        scores_df = score_conversations(
            conversations=completed_convs,
            judge_model=cfg.scoring.judge_model,
            judge_batch_size=cfg.scoring.judge_batch_size,
            judge_max_retries=cfg.scoring.judge_max_retries,
            tokenizer_name=model_cfg.hf_name,
        )

        scores_df.to_csv(scores_path, index=False)
        logger.info(f"Scores saved to {scores_path}")
        save_checkpoint(cfg.output_dir, f"{MODULE_NAME}_score", model_cfg.name)

    click.echo("Behavioral evaluation complete.")


if __name__ == "__main__":
    main()
