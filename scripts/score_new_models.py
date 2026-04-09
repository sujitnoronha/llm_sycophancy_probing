#!/usr/bin/env python3
"""Score Turn 4 responses for the 5 new OpenRouter models using the LLM judge.

Defaults to OpenRouter with openai/gpt-4o-mini to avoid OpenAI quota issues.
The openai library reads OPENAI_BASE_URL + OPENAI_API_KEY from the environment,
so routing through OpenRouter requires no changes to judge.py.

Usage:
    python scripts/score_new_models.py                     # All remaining models via OpenRouter
    python scripts/score_new_models.py --model llama33_70b  # Single model
    python scripts/score_new_models.py --judge openai/gpt-4o  # Different judge
"""

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

# Route judge calls through OpenRouter using the existing OPENROUTER_API_KEY.
# The openai library picks up OPENAI_BASE_URL and OPENAI_API_KEY automatically.
OPENROUTER_KEY = os.environ.get("OPENROUTER_API_KEY", "")
if OPENROUTER_KEY:
    os.environ["OPENAI_API_KEY"] = OPENROUTER_KEY
    os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"
    logger.info("Using OpenRouter as judge backend (OPENROUTER_API_KEY found)")

BEHAVIORAL_DIR = PROJECT_ROOT / "outputs" / "behavioral"

NEW_MODELS = [
    "gemma3_27b",
    "qwen3_30b",
    "llama33_70b",
    "deepseek_r1_70b",
    "qwen3_235b",
]


def load_jsonl(path: Path):
    conversations = []
    with open(path) as f:
        for line in f:
            if line.strip():
                conversations.append(json.loads(line))
    return conversations


def main():
    parser = argparse.ArgumentParser(description="Score new model conversations")
    parser.add_argument("--model", default=None, help="Single model short name to score")
    parser.add_argument(
        "--judge",
        default="openai/gpt-4o-mini",
        help="Judge model (default: openai/gpt-4o-mini via OpenRouter)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=20, help="Judge API batch size"
    )
    args = parser.parse_args()

    from src.behavioral.scorer import score_conversations

    models = [args.model] if args.model else NEW_MODELS

    for model_name in models:
        conv_path = BEHAVIORAL_DIR / f"{model_name}_conversations.jsonl"
        scores_path = BEHAVIORAL_DIR / f"{model_name}_scores.csv"

        if scores_path.exists():
            logger.info(f"Skipping {model_name} — scores already exist at {scores_path}")
            continue

        if not conv_path.exists():
            logger.error(f"No conversations found for {model_name} at {conv_path}")
            continue

        logger.info(f"\nScoring {model_name} with {args.judge}...")
        conversations = load_jsonl(conv_path)
        logger.info(f"  Loaded {len(conversations)} conversations")

        scores_df = score_conversations(
            conversations=conversations,
            judge_model=args.judge,
            judge_batch_size=args.batch_size,
            judge_max_retries=3,
            tokenizer_name=None,  # Use word count fallback for remote models
        )

        scores_df.to_csv(scores_path, index=False)
        logger.info(f"  Saved scores to {scores_path}")

        n_flipped = scores_df["flip"].sum()
        n_total = scores_df["flip"].notna().sum()
        logger.info(
            f"  Flip rate: {n_flipped}/{n_total} = {100*n_flipped/max(n_total,1):.1f}%"
        )

    logger.info("\nDone scoring all models.")


if __name__ == "__main__":
    main()
