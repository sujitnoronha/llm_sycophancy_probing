"""Validate gpt-4o-mini against GPT-5 on a small subset.

Scores 50 random conversations with BOTH gpt-4o-mini and GPT-5, then
computes Cohen's kappa to verify that gpt-4o-mini is reliable enough
for scoring the random baseline.

Usage:
    python scripts/validate_mini_judge.py
    python scripts/validate_mini_judge.py --n-samples 100
"""

import json
import random
from pathlib import Path

import click
import pandas as pd
from dotenv import load_dotenv
from loguru import logger
from sklearn.metrics import cohen_kappa_score

from src.core.config import load_config
from src.core.utils import setup_logging
from src.behavioral.scorer import score_conversations

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


@click.command()
@click.option("--config", default="configs/base.yaml", help="Path to config YAML")
@click.option("--n-samples", default=50, help="Number of conversations to validate")
def main(config: str, n_samples: int):
    cfg = load_config(config)
    setup_logging(cfg.log_level)

    # Find first model's random baseline
    random_path = None
    for model_cfg in cfg.models:
        candidate = (
            Path(cfg.output_dir)
            / "steering"
            / model_cfg.name
            / "steered_random_alpha1.5.jsonl"
        )
        if candidate.exists():
            random_path = candidate
            break

    if random_path is None:
        click.echo("No random baseline found. Run --random-baseline first.")
        return

    logger.info(f"Loading random baseline from {random_path}")
    convs = []
    with open(random_path) as f:
        for line in f:
            if line.strip():
                conv = json.loads(line)
                conv["turn4_assistant"] = conv.get(
                    "turn4_assistant_steered", conv.get("turn4_assistant", "")
                )
                convs.append(conv)

    rng = random.Random(42)
    sample = rng.sample(convs, min(n_samples, len(convs)))
    logger.info(f"Scoring {len(sample)} conversations with both judges")

    # Score with both models
    scores_mini = score_conversations(sample, judge_model="gpt-4o-mini")
    scores_gpt5 = score_conversations(sample, judge_model="gpt-5-2025-08-07")

    # Compute agreement
    mini_flip = scores_mini["flip"].dropna().astype(int)
    gpt5_flip = scores_gpt5["flip"].dropna().astype(int)

    common_idx = mini_flip.index.intersection(gpt5_flip.index)
    if len(common_idx) == 0:
        click.echo("No overlapping scored conversations.")
        return

    kappa = cohen_kappa_score(mini_flip[common_idx], gpt5_flip[common_idx])
    agreement = (mini_flip[common_idx] == gpt5_flip[common_idx]).mean()

    result = {
        "n_samples": int(len(common_idx)),
        "raw_agreement": float(agreement),
        "cohens_kappa": float(kappa),
        "mini_flip_rate": float(mini_flip[common_idx].mean()),
        "gpt5_flip_rate": float(gpt5_flip[common_idx].mean()),
    }

    output_path = Path(cfg.output_dir) / "steering" / "judge_validation.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    click.echo(f"Judge validation results:")
    for k, v in result.items():
        click.echo(f"  {k}: {v}")


if __name__ == "__main__":
    main()
