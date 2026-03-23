"""Score steered conversations with the LLM judge.

Loads steered JSONL files from outputs/steering/{model}/, scores each
conversation's steered Turn 4 with the specified judge model, and saves
scores as CSV files alongside the JSONL files.

Usage:
    python scripts/score_steered.py                          # Score with GPT-5
    python scripts/score_steered.py --judge-model gpt-4o-mini  # Cheaper judge
"""

import json
import os
from pathlib import Path

import click
import pandas as pd
from dotenv import load_dotenv
from loguru import logger

from src.core.config import load_config
from src.core.utils import setup_logging
from src.behavioral.scorer import score_conversations

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


@click.command()
@click.option("--config", default="configs/base.yaml", help="Path to config YAML")
@click.option(
    "--judge-model",
    default=None,
    help="Override judge model (e.g., gpt-4o-mini for random baseline)",
)
def main(config: str, judge_model: str):
    cfg = load_config(config)
    setup_logging(cfg.log_level)

    scoring_model = judge_model or cfg.scoring.judge_model
    logger.info(f"Scoring steered conversations with {scoring_model}")

    for model_cfg in cfg.models:
        steering_dir = Path(cfg.output_dir) / "steering" / model_cfg.name
        if not steering_dir.exists():
            logger.warning(f"No steering output for {model_cfg.name}")
            continue

        for jsonl_path in sorted(steering_dir.glob("steered_*.jsonl")):
            # Derive scores CSV path from JSONL path
            scores_name = jsonl_path.stem.replace("steered_", "scores_") + ".csv"
            scores_path = steering_dir / scores_name

            if scores_path.exists():
                logger.info(f"Skipping {jsonl_path.name} (scores exist at {scores_name})")
                continue

            # Load steered conversations
            convs = []
            with open(jsonl_path) as f:
                for line in f:
                    if line.strip():
                        conv = json.loads(line)
                        # Map steered Turn 4 to the key the judge expects
                        conv["turn4_assistant"] = conv.get(
                            "turn4_assistant_steered",
                            conv.get("turn4_assistant", ""),
                        )
                        convs.append(conv)

            if not convs:
                logger.warning(f"No conversations in {jsonl_path.name}")
                continue

            logger.info(
                f"Scoring {len(convs)} steered convs from {jsonl_path.name} "
                f"with {scoring_model}"
            )

            scores_df = score_conversations(
                convs,
                judge_model=scoring_model,
                judge_batch_size=cfg.scoring.judge_batch_size,
                judge_max_retries=cfg.scoring.judge_max_retries,
            )
            scores_df.to_csv(scores_path, index=False)

            n_flipped = scores_df["flip"].sum() if scores_df["flip"].notna().any() else 0
            n_total = scores_df["flip"].notna().sum()
            flip_rate = n_flipped / max(n_total, 1)
            logger.info(
                f"  {scores_name}: {n_flipped}/{n_total} flipped ({100 * flip_rate:.1f}%)"
            )

    click.echo("Steered conversation scoring complete.")


if __name__ == "__main__":
    main()
