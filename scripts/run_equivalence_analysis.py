#!/usr/bin/env python3
"""Cluster 5 — equivalence analysis for the matched-vs-mismatched (blind deference) claim.

Reviewers KUMj & zMkt: "failure to reject a difference is not proof of equivalence."
This script replaces that inferential gap with the proper tools, computed over the
existing behavioral score CSVs (no model inference required):

  1. TOST equivalence test (per model + pooled), pre-registered margin (default +/-5pp)
  2. Newcombe 95% CI on the matched-minus-mismatched flip-rate difference
  3. Hierarchical (mixed-effects) logistic fit with random intercepts for question_id
     and model_name (also addresses Cluster 6 item-dependence)

Outputs:
  outputs/analysis/equivalence_results.json   (full results)
  outputs/analysis/equivalence_table.md        (rebuttal-ready table)

Usage:
  python scripts/run_equivalence_analysis.py
  python scripts/run_equivalence_analysis.py --margin 0.05 --pooled-authority
"""

import argparse
import glob
import json
import sys
from pathlib import Path

import pandas as pd
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.behavioral_stats import (  # noqa: E402
    equivalence_analysis,
    fit_mixed_effects_logit,
)

SCORES_GLOB = str(PROJECT_ROOT / "outputs" / "behavioral" / "*_scores.csv")
OUT_DIR = PROJECT_ROOT / "outputs" / "analysis"

# Pretty display names for the CSV stems.
DISPLAY = {
    "deepseek_r1_qwen3_8b": "DeepSeek-R1-8B",
    "deepseek_r1_70b": "DeepSeek-R1-70B",
    "qwen3_8b": "Qwen3-8B",
    "qwen3_30b": "Qwen3-30B",
    "qwen3_235b": "Qwen3-235B",
    "llama31_8b": "Llama-3.1-8B",
    "llama33_70b": "Llama-3.3-70B",
    "mistral_7b_v03": "Mistral-7B",
    "gemma3_12b": "Gemma-3-12B",
    "gemma3_27b": "Gemma-3-27B",
}


def load_all_scores():
    paths = sorted(glob.glob(SCORES_GLOB))
    out = {}
    for p in paths:
        stem = Path(p).stem.replace("_scores", "")
        df = pd.read_csv(p)
        if "flip" not in df.columns:
            logger.warning(f"{stem}: no 'flip' column, skipping")
            continue
        df["flip"] = pd.to_numeric(df["flip"], errors="coerce")
        df["model_name"] = stem
        out[stem] = df
        logger.info(f"Loaded {len(df):>5} rows  {stem}")
    return out


def fmt_pct(x):
    return f"{x*100:.1f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--margin", type=float, default=0.05,
                    help="Practical-equivalence margin on the flip-rate scale (default 0.05 = 5pp)")
    ap.add_argument("--high-authority-only", action="store_true",
                    help="Restrict to high authority (clearest-signal subset). "
                         "Default pools across authority levels, matching the paper's Table 2 / BDI.")
    ap.add_argument("--skip-mixed", action="store_true",
                    help="Skip the (slower) mixed-effects logistic fit")
    args = ap.parse_args()

    high_only = args.high_authority_only
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_scores = load_all_scores()
    if not all_scores:
        logger.error(f"No score CSVs found at {SCORES_GLOB}")
        sys.exit(1)

    results = {
        "margin": args.margin,
        "high_authority_only": high_only,
        "per_model": {},
        "pooled": {},
    }

    # ── Per-model equivalence ──────────────────────────────────────────────────
    for stem, df in all_scores.items():
        nc = df[~df["is_control"]].dropna(subset=["flip"]) if "is_control" in df.columns else df
        results["per_model"][stem] = equivalence_analysis(
            nc, margin=args.margin, high_authority_only=high_only
        )

    # ── Pooled equivalence ─────────────────────────────────────────────────────
    combined = pd.concat(all_scores.values(), ignore_index=True)
    nc_combined = (combined[~combined["is_control"]].dropna(subset=["flip"])
                   if "is_control" in combined.columns else combined)
    results["pooled"]["equivalence"] = equivalence_analysis(
        nc_combined, margin=args.margin, high_authority_only=high_only
    )

    # ── Hierarchical mixed-effects logistic (random intercepts) ────────────────
    if not args.skip_mixed:
        logger.info("Fitting hierarchical mixed-effects logistic model (random intercepts: question_id + model_name)...")
        results["pooled"]["mixed_effects"] = fit_mixed_effects_logit(nc_combined)

    # ── Save JSON ──────────────────────────────────────────────────────────────
    with open(OUT_DIR / "equivalence_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # ── Markdown table ─────────────────────────────────────────────────────────
    scope = "high authority only" if high_only else "pooled across authority"
    lines = [
        "# Matched vs. Mismatched — Equivalence Tests (Cluster 5)",
        "",
        f"Practical-equivalence margin: **±{args.margin*100:.0f}pp**  ·  scope: **{scope}**  ·  "
        "TOST α = 0.05.",
        "",
        "Verdict key: **equivalent** = TOST significant + NHST n.s.; "
        "**trivially_different** = detectable but within margin; "
        "**inconclusive** = underpowered; **meaningfully_different** = real gap.",
        "",
        "| Model | Matched | Mismatched | Diff (pp) | 95% CI (pp) | TOST p | Verdict |",
        "|---|--:|--:|--:|:--:|--:|:--|",
    ]

    def row(name, r):
        if "error" in r:
            return f"| {name} | — | — | — | — | — | {r['error']} |"
        diff_pp = r["difference"] * 100
        lo, hi = [x * 100 for x in r["diff_ci95_newcombe"]]
        return (f"| {name} | {fmt_pct(r['matched_flip_rate'])} | "
                f"{fmt_pct(r['mismatched_flip_rate'])} | {diff_pp:+.1f} | "
                f"[{lo:+.1f}, {hi:+.1f}] | {r['tost']['p_tost']:.3f} | **{r['verdict']}** |")

    # Sort per-model rows by mismatched flip rate for readability.
    ordered = sorted(results["per_model"].items(),
                     key=lambda kv: kv[1].get("mismatched_flip_rate", 0))
    for stem, r in ordered:
        lines.append(row(DISPLAY.get(stem, stem), r))
    lines.append(row("**Pooled (all models)**", results["pooled"]["equivalence"]))

    # Mixed-effects summary
    me = results["pooled"].get("mixed_effects")
    if me and "domain_match_effect" in me:
        lines += ["", "## Hierarchical mixed-effects logistic (random intercepts: question + model)", ""]
        lines.append(f"_n = {me['n_observations']:,}; groups = {', '.join(me['random_groups'])}._")
        lines.append("")
        lines.append("| Term | Odds ratio | 95% CI (OR) | log-odds ± SD |")
        lines.append("|---|--:|:--:|:--:|")
        for term, c in me["domain_match_effect"].items():
            lo, hi = c["ci95_odds_ratio"]
            lines.append(f"| {term} | {c['odds_ratio']:.3f} | [{lo:.3f}, {hi:.3f}] | "
                         f"{c['log_odds']:+.3f} ± {c['sd']:.3f} |")
        lines.append("")
        lines.append("An odds ratio ≈ 1 with a tight CI spanning 1 means domain match does not "
                     "shift flip odds once item- and model-level variation is accounted for — "
                     "supporting blind deference without relying on a null result.")
    elif me:
        lines += ["", f"_Mixed-effects fit unavailable: {me.get('error','?')}_"]

    with open(OUT_DIR / "equivalence_table.md", "w") as f:
        f.write("\n".join(lines) + "\n")

    # ── Console summary ────────────────────────────────────────────────────────
    print("\n".join(lines))
    print(f"\nSaved: {OUT_DIR/'equivalence_results.json'}")
    print(f"Saved: {OUT_DIR/'equivalence_table.md'}")


if __name__ == "__main__":
    main()
