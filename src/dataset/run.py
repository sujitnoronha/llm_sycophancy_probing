"""CLI entry point for dataset construction.

Usage: python -m src.dataset.run [--config configs/base.yaml]
"""

import json

import click
from dotenv import load_dotenv

from src.core.config import load_config
from src.core.utils import set_all_seeds, setup_logging, ensure_output_dir
from src.dataset.build_dataset import (
    load_questions,
    load_personas,
    generate_all_conversations,
)

load_dotenv()


@click.command()
@click.option("--config", default="configs/base.yaml", help="Path to config YAML")
def main(config: str):
    cfg = load_config(config)
    setup_logging(cfg.log_level)
    set_all_seeds(cfg.seed)

    questions = load_questions()
    personas = load_personas()

    conversations = generate_all_conversations(
        questions=questions,
        personas=personas,
        system_prompt=cfg.experiment.system_prompt,
    )

    output_dir = ensure_output_dir(cfg.output_dir, "dataset")
    output_path = output_dir / "conversations.json"
    with open(output_path, "w") as f:
        json.dump(conversations, f, indent=2)

    click.echo(f"Saved {len(conversations)} conversations to {output_path}")


if __name__ == "__main__":
    main()
