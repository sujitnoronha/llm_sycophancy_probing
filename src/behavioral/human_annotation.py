"""Human annotation tools for inter-rater reliability.

Generates annotation spreadsheets for 2 annotators and computes
Cohen's kappa (human-human and human-LLM agreement).
"""

import random
from typing import Dict, List, Tuple

import pandas as pd
from loguru import logger


def generate_annotation_samples(
    conversations: List[Dict],
    total_sample: int = 300,
    overlap: int = 100,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate annotation spreadsheets for 2 annotators.

    Randomly samples conversations, with a specified number of overlapping
    items for computing inter-annotator agreement.

    Args:
        conversations: All scored conversations.
        total_sample: Total unique conversations to sample.
        overlap: Number of conversations both annotators score (for kappa).
        seed: Random seed for reproducibility.

    Returns:
        (annotator1_df, annotator2_df) — DataFrames ready for export to CSV.
    """
    rng = random.Random(seed)

    # Sample conversations
    sample_ids = rng.sample(range(len(conversations)), min(total_sample, len(conversations)))
    sampled = [conversations[i] for i in sample_ids]

    # Split: first `overlap` items go to both, rest split evenly
    shared = sampled[:overlap]
    remaining = sampled[overlap:]
    split_point = len(remaining) // 2
    only_a = remaining[:split_point]
    only_b = remaining[split_point:]

    annotator1_convs = shared + only_a
    annotator2_convs = shared + only_b

    # Shuffle so annotators don't know which items overlap
    rng.shuffle(annotator1_convs)
    rng.shuffle(annotator2_convs)

    def make_annotation_df(convs: List[Dict]) -> pd.DataFrame:
        rows = []
        for conv in convs:
            rows.append({
                "conversation_id": conv["conversation_id"],
                "question_id": conv["question_id"],
                "domain": conv["domain"],
                "correct_answer": conv["correct_answer"],
                "wrong_answer": conv["wrong_answer"],
                "turn2_assistant": conv.get("turn2_assistant", ""),
                "turn3_user": conv.get("turn3_user", ""),
                "turn4_assistant": conv.get("turn4_assistant", ""),
                # Columns for annotator to fill:
                "annotator_flip": "",  # true/false
                "annotator_hedge_score": "",  # 0-3
                "annotator_notes": "",
            })
        return pd.DataFrame(rows)

    logger.info(
        f"Annotation samples: {len(annotator1_convs)} for A1, "
        f"{len(annotator2_convs)} for A2, {overlap} overlapping"
    )

    return make_annotation_df(annotator1_convs), make_annotation_df(annotator2_convs)


def compute_agreement(
    annotations_a: pd.DataFrame,
    annotations_b: pd.DataFrame,
    llm_scores: pd.DataFrame,
) -> Dict:
    """Compute Cohen's kappa for human-human and human-LLM agreement.

    Args:
        annotations_a: Annotator 1's completed annotations.
        annotations_b: Annotator 2's completed annotations.
        llm_scores: LLM judge scores for the same conversations.

    Returns:
        Dict with kappa values and agreement statistics.
    """
    from sklearn.metrics import cohen_kappa_score

    # Find overlapping conversations between annotators
    ids_a = set(annotations_a["conversation_id"])
    ids_b = set(annotations_b["conversation_id"])
    overlap_ids = ids_a & ids_b

    if not overlap_ids:
        logger.warning("No overlapping annotations found between annotators")
        return {"error": "no_overlap"}

    # Human-human agreement on overlapping items
    a_overlap = annotations_a[annotations_a["conversation_id"].isin(overlap_ids)].set_index(
        "conversation_id"
    )
    b_overlap = annotations_b[annotations_b["conversation_id"].isin(overlap_ids)].set_index(
        "conversation_id"
    )

    # Align by conversation_id
    common_ids = sorted(overlap_ids)
    a_aligned = a_overlap.loc[common_ids]
    b_aligned = b_overlap.loc[common_ids]

    results = {}

    # Kappa for flip (binary)
    if "annotator_flip" in a_aligned.columns:
        a_flip = a_aligned["annotator_flip"].astype(bool).astype(int)
        b_flip = b_aligned["annotator_flip"].astype(bool).astype(int)
        results["human_human_kappa_flip"] = cohen_kappa_score(a_flip, b_flip)

    # Kappa for hedge score (ordinal, weighted)
    if "annotator_hedge_score" in a_aligned.columns:
        a_hedge = a_aligned["annotator_hedge_score"].astype(int)
        b_hedge = b_aligned["annotator_hedge_score"].astype(int)
        results["human_human_kappa_hedge"] = cohen_kappa_score(
            a_hedge, b_hedge, weights="linear"
        )

    # Human-LLM agreement (using all annotated items from both annotators)
    all_annotated = pd.concat([annotations_a, annotations_b]).drop_duplicates(
        subset=["conversation_id"]
    )
    merged = all_annotated.merge(
        llm_scores[["conversation_id", "flip", "hedge_score"]],
        on="conversation_id",
        how="inner",
    )

    if len(merged) > 0:
        human_flip = merged["annotator_flip"].astype(bool).astype(int)
        llm_flip = merged["flip"].astype(int)
        results["human_llm_kappa_flip"] = cohen_kappa_score(human_flip, llm_flip)

        human_hedge = merged["annotator_hedge_score"].astype(int)
        llm_hedge = merged["hedge_score"].astype(int)
        results["human_llm_kappa_hedge"] = cohen_kappa_score(
            human_hedge, llm_hedge, weights="linear"
        )

    results["n_overlap"] = len(overlap_ids)
    results["n_total_annotated"] = len(all_annotated)

    logger.info(f"Agreement results: {results}")
    return results
