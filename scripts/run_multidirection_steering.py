"""Multi-direction activation steering experiment.

Subtracts all 3 probe directions (authority + sycophancy + confidence)
simultaneously, testing whether the single-direction steering failure
on RLHF models was due to using too few directions.

Usage:
    PYTHONPATH=. python scripts/run_multidirection_steering.py --model llama31_8b
    PYTHONPATH=. python scripts/run_multidirection_steering.py --model qwen3_8b
"""

import json
from pathlib import Path

import click
import numpy as np
import torch
from loguru import logger

from src.core.config import load_config, get_model_config
from src.core.models import load_model_context
from src.core.utils import setup_logging, log_gpu_memory
from src.steering.steer import run_steered_eval


@click.command()
@click.option("--config", default="configs/base.yaml")
@click.option("--model", "model_name", required=True)
@click.option("--alpha", default=1.5, type=float)
def main(config: str, model_name: str, alpha: float):
    cfg = load_config(config)
    setup_logging(cfg.log_level)

    model_cfg = get_model_config(cfg, model_name)

    # Load all 3 probe directions
    wv_path = Path(cfg.output_dir) / "probing" / model_name / "weight_vectors.pt"
    weight_vectors = torch.load(str(wv_path), weights_only=False)

    target_layer = model_cfg.n_layers // 2

    directions = []
    for probe_name in ["authority", "sycophancy", "confidence"]:
        w_all = weight_vectors[f"w_{probe_name}"].numpy()
        w = w_all[target_layer].astype(np.float64)
        norm = np.linalg.norm(w)
        if norm > 1e-10:
            w = w / norm
        directions.append(w)
        logger.info(f"Loaded {probe_name} direction at layer {target_layer}")

    # Sum and re-normalize the combined direction
    combined = directions[0] + directions[1] + directions[2]
    combined = combined / np.linalg.norm(combined)
    logger.info(
        f"Combined 3 directions into single unit vector, "
        f"cos(combined, auth)={np.dot(combined, directions[0]):.3f}, "
        f"cos(combined, syc)={np.dot(combined, directions[1]):.3f}, "
        f"cos(combined, conf)={np.dot(combined, directions[2]):.3f}"
    )

    # Load conversations
    conv_path = Path(cfg.output_dir) / "behavioral" / f"{model_name}_conversations.jsonl"
    conversations = []
    with open(conv_path) as f:
        for line in f:
            if line.strip():
                conversations.append(json.loads(line))
    conversations = conversations[:200]
    logger.info(f"Loaded {len(conversations)} conversations")

    # Load model and run steering
    with load_model_context(model_cfg, cfg, need_hooks=False) as (model, tokenizer):
        log_gpu_memory(f"Model loaded ({model_name}): ")

        steered = run_steered_eval(
            model=model,
            tokenizer=tokenizer,
            conversations=conversations,
            direction=combined,
            alpha=alpha,
            target_layer=target_layer,
            config=cfg,
            use_hf=True,
        )

    # Save results
    out_dir = Path(cfg.output_dir) / "steering" / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"steered_multidirection_alpha{alpha}.jsonl"
    with open(out_path, "w") as f:
        for conv in steered:
            f.write(json.dumps(conv, default=str) + "\n")
    logger.info(f"Saved {len(steered)} conversations to {out_path}")


if __name__ == "__main__":
    main()
