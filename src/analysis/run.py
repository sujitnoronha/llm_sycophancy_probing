"""CLI entry point for statistical analysis of behavioral data.

Usage: python -m src.analysis.run [--config configs/base.yaml]
"""

import json

import click
import pandas as pd
from dotenv import load_dotenv
from loguru import logger

from src.core.config import load_config
from src.core.utils import set_all_seeds, setup_logging, ensure_output_dir
from src.analysis.behavioral_stats import (
    run_factorial_anova,
    run_logistic_regression,
    compute_all_effect_sizes,
    matched_vs_mismatched_analysis,
    build_cross_model_summary,
)

load_dotenv()


@click.command()
@click.option("--config", default="configs/base.yaml", help="Path to config YAML")
def main(config: str):
    cfg = load_config(config)
    setup_logging(cfg.log_level)
    set_all_seeds(cfg.seed)

    output_dir = ensure_output_dir(cfg.output_dir, "analysis")

    # Load scores from all models
    all_scores = {}
    for model_cfg in cfg.models:
        scores_path = f"{cfg.output_dir}/behavioral/{model_cfg.name}_scores.csv"
        try:
            df = pd.read_csv(scores_path)
            all_scores[model_cfg.name] = df
            logger.info(f"Loaded {len(df)} scores for {model_cfg.name}")
        except FileNotFoundError:
            logger.warning(f"Scores not found for {model_cfg.name} at {scores_path}")

    if not all_scores:
        click.echo("No scores found. Run behavioral evaluation first.")
        return

    # Combine all scores
    combined = pd.concat(all_scores.values(), ignore_index=True)
    # Filter to non-control conditions for factorial analysis
    factorial_df = combined[~combined["is_control"]].dropna(subset=["flip"])

    # 1. Factorial ANOVA for each DV
    anova_results = {}
    for dv in ["flip", "hedge_score", "deference_ratio"]:
        if dv in factorial_df.columns:
            anova_results[dv] = run_factorial_anova(factorial_df, dv)
            logger.info(f"ANOVA for {dv}: completed")

    with open(output_dir / "anova_results.json", "w") as f:
        json.dump(anova_results, f, indent=2)

    # 2. Logistic regression for flip
    logreg_results = run_logistic_regression(factorial_df)
    with open(output_dir / "logistic_regression.json", "w") as f:
        json.dump(logreg_results, f, indent=2)

    # 3. Effect sizes
    effect_sizes = {}
    for model_name, df in all_scores.items():
        non_control = df[~df["is_control"]].dropna(subset=["flip"])
        effect_sizes[model_name] = compute_all_effect_sizes(non_control)
    # Also compute on combined
    effect_sizes["combined"] = compute_all_effect_sizes(factorial_df)

    with open(output_dir / "effect_sizes.json", "w") as f:
        json.dump(effect_sizes, f, indent=2)

    # 4. Matched vs mismatched analysis (the key novelty test)
    matched_analysis = {}
    for model_name, df in all_scores.items():
        non_control = df[~df["is_control"]].dropna(subset=["flip"])
        matched_analysis[model_name] = matched_vs_mismatched_analysis(non_control)
    matched_analysis["combined"] = matched_vs_mismatched_analysis(factorial_df)

    with open(output_dir / "matched_vs_mismatched.json", "w") as f:
        json.dump(matched_analysis, f, indent=2)

    # 5. Cross-model summary
    summary_df = build_cross_model_summary(all_scores)
    summary_df.to_csv(output_dir / "cross_model_summary.csv", index=False)

    logger.info(f"All analysis results saved to {output_dir}")
    click.echo("Statistical analysis complete.")


if __name__ == "__main__":
    main()
