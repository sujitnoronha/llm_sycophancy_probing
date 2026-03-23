"""CLI entry point for probe training.

Usage:
    python -m src.probing.run                              # All models
    python -m src.probing.run --model llama3_8b            # Single model
    python -m src.probing.run --model llama3_8b --permtest # With permutation tests
"""

import json

import click
import numpy as np
import torch
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
from src.activations.collector import load_activations
from src.probing.trainer import train_all_probes_single_model, run_permutation_test, train_confound_probes

load_dotenv()

MODULE_NAME = "probing"


@click.command()
@click.option("--config", default="configs/base.yaml", help="Path to config YAML")
@click.option("--model", "model_name", default=None, help="Run for a single model")
@click.option("--permtest", is_flag=True, help="Run permutation tests (slow)")
def main(config: str, model_name: str, permtest: bool):
    cfg = load_config(config)
    setup_logging(cfg.log_level)
    set_all_seeds(cfg.seed)

    models_to_run = (
        [get_model_config(cfg, model_name)] if model_name else cfg.models
    )

    probing_config = {
        "train_fraction": cfg.probing.train_fraction,
        "probe_regularization": cfg.probing.probe_regularization,
        "max_iter": cfg.probing.max_iter,
        "solver": cfg.probing.solver,
        "seed": cfg.seed,
    }

    for model_cfg in models_to_run:
        if check_checkpoint(cfg.output_dir, MODULE_NAME, model_cfg.name):
            logger.info(f"Skipping {model_cfg.name} (checkpoint exists)")
            continue

        # Load activations
        act_path = f"{cfg.output_dir}/activations/{model_cfg.name}_activations.pt"
        logger.info(f"Loading activations from {act_path}")
        data = load_activations(act_path)

        activations = data["activations"].numpy()
        labels = {k: v.numpy() for k, v in data["labels"].items()}
        domains = data["metadata"]["domains"]

        # Train all probes
        logger.info(f"Training probes for {model_cfg.name}")
        results = train_all_probes_single_model(
            activations, labels, domains, probing_config
        )

        # Save results
        model_output_dir = ensure_output_dir(cfg.output_dir, f"probing/{model_cfg.name}")

        # Save accuracy curves (JSON-serializable)
        accuracy_curves = {}
        for probe_name, probe_data in results["probes"].items():
            if "error" in probe_data:
                accuracy_curves[probe_name] = {"error": probe_data["error"]}
                continue
            accuracy_curves[probe_name] = {
                "layer_results": probe_data["layer_results"],
                "peak_layer": probe_data["peak_layer"],
                "peak_accuracy": probe_data["peak_accuracy"],
            }

        with open(model_output_dir / "probe_accuracy_curves.json", "w") as f:
            json.dump(accuracy_curves, f, indent=2)

        # Save weight vectors (torch tensors)
        weight_vectors = {}
        for probe_name, probe_data in results["probes"].items():
            if "error" in probe_data:
                continue
            weight_vectors[f"w_{probe_name}"] = torch.from_numpy(
                probe_data["weight_vectors"]
            )
            weight_vectors[f"peak_layer_{probe_name}"] = probe_data["peak_layer"]

        torch.save(weight_vectors, str(model_output_dir / "weight_vectors.pt"))

        # Save summary
        summary = {
            "model_name": model_cfg.name,
            "n_prompts": results["n_prompts"],
            "n_layers": results["n_layers"],
            "d_model": results["d_model"],
        }
        for probe_name, probe_data in results["probes"].items():
            if "error" in probe_data:
                summary[probe_name] = {"error": probe_data["error"]}
            else:
                summary[probe_name] = {
                    "peak_layer": probe_data["peak_layer"],
                    "peak_accuracy": probe_data["peak_accuracy"],
                    "n_positive": probe_data["n_positive"],
                    "n_negative": probe_data["n_negative"],
                }

        with open(model_output_dir / "probe_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        # Optional: permutation tests
        if permtest:
            perm_results = {}
            for probe_name, probe_data in results["probes"].items():
                if "error" in probe_data:
                    continue
                peak_layer = probe_data["peak_layer"]
                logger.info(
                    f"Running permutation test for {probe_name} at layer {peak_layer}"
                )
                perm = run_permutation_test(
                    activations,
                    labels[probe_name],
                    domains,
                    layer_idx=peak_layer,
                    n_permutations=1000,
                    config_probing=probing_config,
                )
                perm_results[probe_name] = perm
                logger.info(
                    f"  {probe_name}: p={perm['p_value']:.4f} "
                    f"(observed={perm['observed_accuracy']:.3f}, "
                    f"null={perm['null_mean']:.3f}±{perm['null_std']:.3f})"
                )

            with open(model_output_dir / "permutation_results.json", "w") as f:
                json.dump(perm_results, f, indent=2)

        # Confound control probes (reviewer concern: surface confounds)
        logger.info(f"Training confound probes for {model_cfg.name}")
        confound_results = train_confound_probes(
            activations, data["metadata"], probing_config
        )

        confound_summary = {}
        confound_weight_vectors = {}
        for conf_name, conf_data in confound_results.items():
            if "error" in conf_data:
                confound_summary[conf_name] = {"error": conf_data["error"]}
                continue
            confound_summary[conf_name] = {
                "peak_layer": conf_data["peak_layer"],
                "peak_accuracy": conf_data["peak_accuracy"],
                "n_positive": conf_data["n_positive"],
                "n_negative": conf_data["n_negative"],
            }
            confound_weight_vectors[f"w_{conf_name}"] = torch.from_numpy(
                conf_data["weight_vectors"]
            )
            confound_weight_vectors[f"peak_layer_{conf_name}"] = conf_data["peak_layer"]

        with open(model_output_dir / "confound_probe_summary.json", "w") as f:
            json.dump(confound_summary, f, indent=2)

        if confound_weight_vectors:
            torch.save(
                confound_weight_vectors,
                str(model_output_dir / "confound_weight_vectors.pt"),
            )

        save_checkpoint(cfg.output_dir, MODULE_NAME, model_cfg.name)
        logger.info(f"Probing complete for {model_cfg.name}")

    click.echo("Probe training complete.")


if __name__ == "__main__":
    main()
