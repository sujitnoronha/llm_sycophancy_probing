"""CLI entry point for baseline validation using vLLM.

Usage:
    python -m src.baseline.run                    # Validate all models
    python -m src.baseline.run --model llama3_8b  # Validate one model
"""

import gc
import json
import multiprocessing
import os
import random

import click
import numpy as np
from dotenv import load_dotenv
from loguru import logger

from src.core.config import load_config, get_model_config
from src.core.utils import (
    setup_logging,
    ensure_output_dir,
    check_checkpoint,
    save_checkpoint,
)
from src.dataset.build_dataset import load_questions
from src.baseline.validate import validate_single_model_vllm

load_dotenv()

MODULE_NAME = "baseline"


def set_seeds_no_cuda(seed: int) -> None:
    """Set seeds for CPU RNG sources only (no CUDA init)."""
    random.seed(seed)
    np.random.seed(seed)


@click.command()
@click.option("--config", default="configs/base.yaml", help="Path to config YAML")
@click.option("--model", "model_name", default=None, help="Run for a single model")
def main(config: str, model_name: str):
    cfg = load_config(config)
    setup_logging(cfg.log_level)
    set_seeds_no_cuda(cfg.seed)

    questions = load_questions()
    output_dir = ensure_output_dir(cfg.output_dir, MODULE_NAME)

    models_to_run = (
        [get_model_config(cfg, model_name)] if model_name else cfg.models
    )

    for model_cfg in models_to_run:
        if check_checkpoint(cfg.output_dir, MODULE_NAME, model_cfg.name):
            logger.info(f"Skipping {model_cfg.name} (checkpoint exists)")
            continue

        logger.info(f"Validating baseline for {model_cfg.name} ({model_cfg.hf_name})")

        from vllm import LLM

        llm = LLM(
            model=model_cfg.hf_name,
            dtype=cfg.dtype,
            seed=cfg.seed,
            max_model_len=4096,
            gpu_memory_utilization=0.85,
            max_num_seqs=128,
        )

        results = validate_single_model_vllm(
            llm=llm,
            questions=questions,
            system_prompt=cfg.experiment.system_prompt,
            max_new_tokens=cfg.experiment.max_new_tokens_turn2,
            hf_model_name=model_cfg.hf_name,
        )

        # Cleanup vLLM model before loading the next one
        del llm
        gc.collect()

        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Save results
        output_path = output_dir / f"{model_cfg.name}_baseline.json"
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        n_correct = sum(1 for r in results if r["is_correct"])
        logger.info(f"{model_cfg.name}: {n_correct}/{len(questions)} correct → {output_path}")

        save_checkpoint(cfg.output_dir, MODULE_NAME, model_cfg.name)

    click.echo("Baseline validation complete.")


if __name__ == "__main__":
    # Must set spawn before any CUDA init for vLLM multiprocessing
    multiprocessing.set_start_method("spawn", force=True)
    main()
