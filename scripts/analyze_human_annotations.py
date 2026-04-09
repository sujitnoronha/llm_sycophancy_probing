#!/usr/bin/env python3
"""
Human annotation analysis script.

Matches Label Studio annotation exports to the correct model's scored conversations
by comparing turn4_assistant text, then computes inter-rater reliability and
human-vs-GPT-5 agreement statistics.
"""

import json
import os
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ANN_DIR = ROOT / "annotation_package" / "annotations_results"
BEH_DIR = ROOT / "outputs" / "behavioral"

SCORED_MODELS = [
    "llama31_8b", "mistral_7b_v03", "qwen3_8b",
    "gemma3_12b", "deepseek_r1_qwen3_8b", "gemma3_27b",
    "qwen3_30b", "llama33_70b"
]

# --------------------------------------------------------------------------- #
# 1.  Load Label Studio exports
# --------------------------------------------------------------------------- #

def parse_label_studio(json_path):
    """Parse a Label Studio JSON export into a DataFrame."""
    with open(json_path) as f:
        data = json.load(f)

    rows = []
    for item in data:
        d = item["data"]
        flip_label = hedge_label = None
        for ann in item.get("annotations", []):
            for r in ann.get("result", []):
                name = r.get("from_name", "")
                choices = r.get("value", {}).get("choices", [])
                if name == "annotator_flip" and choices:
                    flip_label = choices[0]          # "TRUE" or "FALSE"
                elif name == "annotator_hedge_score" and choices:
                    hedge_label = choices[0]         # "0 - Firm, no hedging" etc.
        if flip_label is None:
            continue  # skip unannotated tasks
        rows.append({
            "conversation_id": d["conversation_id"],
            "turn4_assistant": d["turn4_assistant"],
            "human_flip_raw": flip_label,
            "human_hedge_raw": hedge_label,
        })
    df = pd.DataFrame(rows)
    df["human_flip"] = (df["human_flip_raw"].str.upper() == "TRUE").astype(int)
    # Hedge score: extract leading digit
    df["human_hedge"] = df["human_hedge_raw"].str[0].astype(float)
    return df


print("Loading Label Studio exports …")
ann1 = parse_label_studio(ANN_DIR / "project-7-at-2026-03-30-22-55-26fd2987.json")
ann2 = parse_label_studio(ANN_DIR / "project-1-at-2026-03-31-23-59-5c83a553.json")

print(f"  Annotator 1: {len(ann1)} annotations")
print(f"  Annotator 2: {len(ann2)} annotations")

# --------------------------------------------------------------------------- #
# 2.  Build a turn4 → (model, conversation_id) lookup from all JSONL files
# --------------------------------------------------------------------------- #

print("\nBuilding turn4 text → model lookup …")

# Key: normalised turn4 text (strip whitespace), Value: (model_key, conversation_id)
turn4_index = {}

for model in SCORED_MODELS:
    jsonl_path = BEH_DIR / f"{model}_conversations.jsonl"
    if not jsonl_path.exists():
        print(f"  WARNING: {jsonl_path} not found, skipping")
        continue
    count = 0
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            t4 = rec.get("turn4_assistant", "")
            if t4:
                key = t4.strip()
                turn4_index[key] = (model, rec["conversation_id"])
                count += 1
    print(f"  {model}: {count} records indexed")

print(f"  Total unique turn4 texts: {len(turn4_index)}")

# --------------------------------------------------------------------------- #
# 3.  Match each annotation to its model
# --------------------------------------------------------------------------- #

def match_annotations(ann_df, source_name):
    matched = []
    unmatched = 0
    for _, row in ann_df.iterrows():
        key = row["turn4_assistant"].strip()
        if key in turn4_index:
            model_key, conv_id = turn4_index[key]
            matched.append({**row.to_dict(),
                            "matched_model": model_key,
                            "matched_conv_id": conv_id})
        else:
            # Try prefix matching (truncation at 500 chars)
            found = False
            for t4key, val in turn4_index.items():
                if key[:200] == t4key[:200]:
                    matched.append({**row.to_dict(),
                                    "matched_model": val[0],
                                    "matched_conv_id": val[1]})
                    found = True
                    break
            if not found:
                unmatched += 1
                print(f"  [{source_name}] UNMATCHED conv_id={row['conversation_id']}")
    print(f"  [{source_name}] Matched {len(matched)}/{len(ann_df)}, unmatched {unmatched}")
    return pd.DataFrame(matched)


print("\nMatching annotations to models …")
matched1 = match_annotations(ann1, "annotator1")
matched2 = match_annotations(ann2, "annotator2")

# --------------------------------------------------------------------------- #
# 4.  Load GPT-5 scores for matched conversations
# --------------------------------------------------------------------------- #

print("\nLoading GPT-5 scores …")

all_scores = []
for model in SCORED_MODELS:
    csv_path = BEH_DIR / f"{model}_scores.csv"
    if not csv_path.exists():
        continue
    df = pd.read_csv(csv_path)
    df["model_key"] = model
    all_scores.append(df)

scores_df = pd.concat(all_scores, ignore_index=True)
print(f"  Loaded {len(scores_df)} score rows across {len(all_scores)} models")

# --------------------------------------------------------------------------- #
# 5.  Merge annotations with GPT-5 scores
# --------------------------------------------------------------------------- #

def merge_with_scores(matched_df, annotator_name):
    merged = matched_df.merge(
        scores_df[["conversation_id", "model_key", "flip", "hedge_score"]],
        left_on=["matched_conv_id", "matched_model"],
        right_on=["conversation_id", "model_key"],
        how="left"
    )
    merged = merged.rename(columns={"flip": "gpt5_flip", "hedge_score": "gpt5_hedge"})
    n_missing = merged["gpt5_flip"].isna().sum()
    if n_missing:
        print(f"  [{annotator_name}] WARNING: {n_missing} rows have no GPT-5 score")
    merged = merged.dropna(subset=["gpt5_flip"])
    merged["gpt5_flip"] = merged["gpt5_flip"].astype(int)
    print(f"  [{annotator_name}]: {len(merged)} rows with complete data")
    return merged


merged1 = merge_with_scores(matched1, "annotator1")
merged2 = merge_with_scores(matched2, "annotator2")

all_merged = pd.concat([merged1, merged2], ignore_index=True)
print(f"\nCombined: {len(all_merged)} annotated conversations")

# --------------------------------------------------------------------------- #
# 6.  Agreement statistics
# --------------------------------------------------------------------------- #

from sklearn.metrics import cohen_kappa_score, confusion_matrix
from scipy.stats import spearmanr

print("\n" + "="*60)
print("HUMAN vs GPT-5 AGREEMENT (combined annotators)")
print("="*60)

h = all_merged["human_flip"].values
g = all_merged["gpt5_flip"].values

kappa = cohen_kappa_score(h, g)
cm = confusion_matrix(h, g, labels=[0, 1])
tn, fp, fn, tp = cm.ravel()
precision = tp / (tp + fp) if (tp + fp) else 0
recall    = tp / (tp + fn) if (tp + fn) else 0
f1        = 2*precision*recall/(precision+recall) if (precision+recall) else 0
accuracy  = (tp + tn) / len(h)

print(f"\nFlip label agreement (human=1 → flip, gpt5=1 → flip)")
print(f"  N = {len(h)}")
print(f"  Human flip rate:  {h.mean()*100:.1f}%")
print(f"  GPT-5 flip rate:  {g.mean()*100:.1f}%")
print(f"  Cohen's κ:        {kappa:.3f}")
print(f"  Accuracy:         {accuracy*100:.1f}%")
print(f"  Confusion matrix (rows=human, cols=gpt5):")
print(f"    TN={tn}  FP={fp}")
print(f"    FN={fn}  TP={tp}")
print(f"  Precision (GPT-5 for flip=1): {precision:.3f}")
print(f"  Recall    (GPT-5 for flip=1): {recall:.3f}")
print(f"  F1:                           {f1:.3f}")

# Hedge score correlation
h_hed = all_merged["human_hedge"].values
g_hed = all_merged["gpt5_hedge"].values
rho, pval = spearmanr(h_hed, g_hed)
print(f"\nHedge score correlation")
print(f"  Spearman ρ = {rho:.3f}  (p={pval:.4f}, n={len(h_hed)})")

# --------------------------------------------------------------------------- #
# 6b. Per-annotator breakdown
# --------------------------------------------------------------------------- #

print("\n" + "-"*40)
print("PER-ANNOTATOR breakdown")
for name, df in [("Annotator 1", merged1), ("Annotator 2", merged2)]:
    h_ = df["human_flip"].values
    g_ = df["gpt5_flip"].values
    if len(h_) < 2:
        continue
    k_ = cohen_kappa_score(h_, g_)
    acc_ = (h_ == g_).mean()
    print(f"  {name}: N={len(h_)}, κ={k_:.3f}, acc={acc_*100:.1f}%, "
          f"human_flip={h_.mean()*100:.1f}%, gpt5_flip={g_.mean()*100:.1f}%")

# --------------------------------------------------------------------------- #
# 6c. Inter-annotator agreement on overlapping items
# --------------------------------------------------------------------------- #

overlap = merged1.merge(
    merged2[["matched_conv_id", "human_flip", "human_hedge"]],
    on="matched_conv_id", suffixes=("_a1", "_a2")
)
print(f"\n  Overlapping items between annotators: {len(overlap)}")
if len(overlap) >= 10:
    k_iaa = cohen_kappa_score(overlap["human_flip_a1"], overlap["human_flip_a2"])
    print(f"  Inter-annotator κ (flip): {k_iaa:.3f}")
    rho_iaa, _ = spearmanr(overlap["human_hedge_a1"], overlap["human_hedge_a2"])
    print(f"  Inter-annotator ρ (hedge): {rho_iaa:.3f}")

# --------------------------------------------------------------------------- #
# 7.  Model distribution of annotations
# --------------------------------------------------------------------------- #

print("\n" + "-"*40)
print("Model distribution of annotated conversations")
print(all_merged["matched_model"].value_counts().to_string())

# --------------------------------------------------------------------------- #
# 8.  Save matched dataset
# --------------------------------------------------------------------------- #

out_path = ROOT / "outputs" / "behavioral" / "human_annotations_matched.csv"
all_merged.to_csv(out_path, index=False)
print(f"\nMatched dataset saved: {out_path}")

# --------------------------------------------------------------------------- #
# 9.  Summary dict for paper
# --------------------------------------------------------------------------- #

summary = {
    "n_annotations": int(len(all_merged)),
    "human_flip_rate_pct": float(round(h.mean()*100, 1)),
    "gpt5_flip_rate_pct": float(round(g.mean()*100, 1)),
    "cohens_kappa": float(round(kappa, 3)),
    "accuracy_pct": float(round(accuracy*100, 1)),
    "precision": float(round(precision, 3)),
    "recall": float(round(recall, 3)),
    "f1": float(round(f1, 3)),
    "spearman_rho_hedge": float(round(rho, 3)),
    "spearman_p_hedge": float(round(pval, 4)),
    "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
}

summary_path = ROOT / "outputs" / "behavioral" / "annotation_agreement_summary.json"
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"Summary JSON saved:    {summary_path}")
print("\nDone.")
