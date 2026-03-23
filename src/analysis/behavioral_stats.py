"""Statistical analysis of behavioral evaluation results.

Implements factorial ANOVA, logistic regression, effect sizes,
and the critical matched vs mismatched domain analysis.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats


def run_factorial_anova(
    df: pd.DataFrame,
    dv: str,
    factors: List[str] = None,
) -> Dict:
    """Run factorial ANOVA on a dependent variable.

    Uses statsmodels OLS with Type III sum of squares.

    Args:
        df: Scores DataFrame (filtered to non-control conditions).
        dv: Dependent variable column name.
        factors: Factor columns. Defaults to authority, domain_match, confidence.

    Returns:
        Dict with F-statistics, p-values, eta-squared for all effects.
    """
    import statsmodels.api as sm
    from statsmodels.formula.api import ols

    if factors is None:
        factors = ["authority_level", "domain_match", "confidence_level"]

    # Build formula with all main effects and interactions
    formula = f"{dv} ~ " + " * ".join([f"C({f})" for f in factors])

    try:
        model = ols(formula, data=df).fit()
        anova_table = sm.stats.anova_lm(model, typ=3)

        results = {
            "dv": dv,
            "n_observations": len(df),
            "r_squared": model.rsquared,
            "effects": {},
        }

        for idx, row in anova_table.iterrows():
            if idx == "Residual" or idx == "Intercept":
                continue
            results["effects"][idx] = {
                "sum_sq": float(row["sum_sq"]),
                "df": float(row["df"]),
                "F": float(row["F"]) if not np.isnan(row["F"]) else None,
                "p_value": float(row["PR(>F)"]) if not np.isnan(row["PR(>F)"]) else None,
            }

        # Compute eta-squared for each effect
        ss_total = anova_table["sum_sq"].sum()
        for effect_name in results["effects"]:
            ss_effect = results["effects"][effect_name]["sum_sq"]
            results["effects"][effect_name]["eta_squared"] = ss_effect / ss_total

        return results

    except Exception as e:
        logger.error(f"ANOVA failed for {dv}: {e}")
        return {"dv": dv, "error": str(e)}


def run_logistic_regression(df: pd.DataFrame) -> Dict:
    """Logistic regression predicting flip from all factors + interactions.

    Returns odds ratios with 95% CIs.
    """
    import statsmodels.api as sm

    # Clean subset: ensure proper types before dummification
    df_clean = df[["flip", "authority_level", "domain_match", "confidence_level"]].copy()
    df_clean["flip"] = pd.to_numeric(df_clean["flip"], errors="coerce")
    df_clean = df_clean.dropna(subset=["flip"])

    for col in ["authority_level", "domain_match", "confidence_level"]:
        df_clean[col] = df_clean[col].astype(str)

    # Encode categorical variables
    df_encoded = pd.get_dummies(
        df_clean,
        columns=["authority_level", "domain_match", "confidence_level"],
        drop_first=True,
    )

    y = df_encoded["flip"].astype(float)
    X = df_encoded.drop(columns=["flip"]).astype(float)
    X = sm.add_constant(X)

    try:
        model = sm.Logit(y, X).fit(disp=0)

        results = {
            "n_observations": len(df),
            "pseudo_r_squared": model.prsquared,
            "llr_pvalue": model.llr_pvalue,
            "coefficients": {},
        }

        for name, coef, pval, ci_low, ci_high in zip(
            model.params.index,
            model.params,
            model.pvalues,
            model.conf_int()[0],
            model.conf_int()[1],
        ):
            results["coefficients"][name] = {
                "coef": float(coef),
                "odds_ratio": float(np.exp(coef)),
                "p_value": float(pval),
                "ci_95_lower": float(np.exp(ci_low)),
                "ci_95_upper": float(np.exp(ci_high)),
            }

        return results

    except Exception as e:
        logger.error(f"Logistic regression failed: {e}")
        return {"error": str(e)}


def compute_effect_size(
    group_a: np.ndarray,
    group_b: np.ndarray,
) -> Tuple[float, Tuple[float, float]]:
    """Compute Cohen's d with 95% bootstrap CI.

    Returns (d, (ci_lower, ci_upper)).
    """
    n_a, n_b = len(group_a), len(group_b)
    mean_diff = np.mean(group_a) - np.mean(group_b)

    # Pooled standard deviation
    pooled_std = np.sqrt(
        ((n_a - 1) * np.var(group_a, ddof=1) + (n_b - 1) * np.var(group_b, ddof=1))
        / (n_a + n_b - 2)
    )

    d = mean_diff / pooled_std if pooled_std > 0 else 0.0

    # Bootstrap CI
    rng = np.random.RandomState(42)
    boot_ds = []
    for _ in range(10000):
        boot_a = rng.choice(group_a, size=n_a, replace=True)
        boot_b = rng.choice(group_b, size=n_b, replace=True)
        boot_diff = np.mean(boot_a) - np.mean(boot_b)
        boot_pooled = np.sqrt(
            ((n_a - 1) * np.var(boot_a, ddof=1) + (n_b - 1) * np.var(boot_b, ddof=1))
            / (n_a + n_b - 2)
        )
        boot_d = boot_diff / boot_pooled if boot_pooled > 0 else 0.0
        boot_ds.append(boot_d)

    ci = (float(np.percentile(boot_ds, 2.5)), float(np.percentile(boot_ds, 97.5)))
    return float(d), ci


def compute_all_effect_sizes(df: pd.DataFrame) -> Dict:
    """Compute Cohen's d for key comparisons."""
    results = {}

    # 1. High authority vs no persona
    high = df[df["authority_level"] == "high"]["flip"].values.astype(float)
    none_ = df[df["authority_level"] == "none"]["flip"].values.astype(float)
    if len(high) > 0 and len(none_) > 0:
        d, ci = compute_effect_size(high, none_)
        results["high_vs_none"] = {"cohens_d": d, "ci_95": ci, "n_high": len(high), "n_none": len(none_)}

    # 2. Matched vs mismatched domain
    matched = df[df["domain_match"] == "matched"]["flip"].values.astype(float)
    mismatched = df[df["domain_match"] == "mismatched"]["flip"].values.astype(float)
    if len(matched) > 0 and len(mismatched) > 0:
        d, ci = compute_effect_size(matched, mismatched)
        results["matched_vs_mismatched"] = {"cohens_d": d, "ci_95": ci}

    # 3. High confidence vs low confidence
    high_conf = df[df["confidence_level"] == "high"]["flip"].values.astype(float)
    low_conf = df[df["confidence_level"] == "low"]["flip"].values.astype(float)
    if len(high_conf) > 0 and len(low_conf) > 0:
        d, ci = compute_effect_size(high_conf, low_conf)
        results["highconf_vs_lowconf"] = {"cohens_d": d, "ci_95": ci}

    # 4. Extreme comparison: high auth + high conf vs low auth + low conf
    extreme_high = df[
        (df["authority_level"] == "high") & (df["confidence_level"] == "high")
    ]["flip"].values.astype(float)
    extreme_low = df[
        (df["authority_level"] == "low") & (df["confidence_level"] == "low")
    ]["flip"].values.astype(float)
    if len(extreme_high) > 0 and len(extreme_low) > 0:
        d, ci = compute_effect_size(extreme_high, extreme_low)
        results["extreme_high_vs_low"] = {"cohens_d": d, "ci_95": ci}

    return results


def matched_vs_mismatched_analysis(df: pd.DataFrame) -> Dict:
    """The critical novelty test: blind authority deference vs rational expertise weighting.

    Tests whether the authority effect is the same for domain-matched and
    domain-mismatched personas.
    """
    # Filter to high authority only (clearest signal)
    high_auth = df[df["authority_level"] == "high"]

    matched = high_auth[high_auth["domain_match"] == "matched"]
    mismatched = high_auth[high_auth["domain_match"] == "mismatched"]

    matched_flip_rate = matched["flip"].mean() if len(matched) > 0 else 0
    mismatched_flip_rate = mismatched["flip"].mean() if len(mismatched) > 0 else 0

    # Two-proportion z-test
    n1, n2 = len(matched), len(mismatched)
    p1, p2 = matched_flip_rate, mismatched_flip_rate

    if n1 > 0 and n2 > 0:
        p_pooled = (p1 * n1 + p2 * n2) / (n1 + n2)
        if p_pooled > 0 and p_pooled < 1:
            se = np.sqrt(p_pooled * (1 - p_pooled) * (1 / n1 + 1 / n2))
            z_stat = (p1 - p2) / se if se > 0 else 0
            p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        else:
            z_stat, p_value = 0.0, 1.0
    else:
        z_stat, p_value = 0.0, 1.0

    return {
        "matched_flip_rate": float(matched_flip_rate),
        "mismatched_flip_rate": float(mismatched_flip_rate),
        "difference": float(matched_flip_rate - mismatched_flip_rate),
        "z_statistic": float(z_stat),
        "p_value": float(p_value),
        "n_matched": n1,
        "n_mismatched": n2,
        "interpretation": (
            "blind_authority_deference"
            if p_value > 0.05
            else "rational_expertise_weighting"
        ),
    }


def build_cross_model_summary(all_scores: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build a cross-model summary table of key metrics."""
    rows = []
    for model_name, df in all_scores.items():
        row = {
            "model": model_name,
            "n_conversations": len(df),
            "overall_flip_rate": df["flip"].mean(),
            "high_auth_flip_rate": df[df["authority_level"] == "high"]["flip"].mean(),
            "medium_auth_flip_rate": df[df["authority_level"] == "medium"]["flip"].mean(),
            "low_auth_flip_rate": df[df["authority_level"] == "low"]["flip"].mean(),
            "no_persona_flip_rate": df[df["authority_level"] == "none"]["flip"].mean(),
            "mean_hedge_score": df["hedge_score"].mean(),
            "mean_deference_ratio": df["deference_ratio"].mean(),
        }
        rows.append(row)

    return pd.DataFrame(rows)


def question_level_analysis(
    df: pd.DataFrame,
    n_leave_out: int = 5,
    n_bootstrap: int = 200,
    seed: int = 42,
) -> Dict:
    """Question-level sensitivity analysis.

    Addresses reviewer concern about narrow dataset (50 questions):
    a) Per-question flip rates across all models
    b) Identify outlier questions (top/bottom by flip rate)
    c) Re-run ANOVA excluding top-5 outlier questions
    d) Leave-5-out bootstrap for robustness

    Args:
        df: Combined scores DataFrame (all models, including controls).
        n_leave_out: Number of questions to drop per bootstrap iteration.
        n_bootstrap: Number of bootstrap iterations.
        seed: Random seed.

    Returns:
        Dict with per-question rates, outliers, robust ANOVA, and bootstrap CIs.
    """
    rng = np.random.RandomState(seed)

    # a) Per-question flip rates
    q_rates = (
        df.dropna(subset=["flip"])
        .groupby("question_id")["flip"]
        .agg(["mean", "count", "sum"])
        .reset_index()
    )
    q_rates.columns = ["question_id", "flip_rate", "n_conversations", "n_flips"]
    q_rates = q_rates.sort_values("flip_rate", ascending=False)

    # b) Outlier identification (top 5 and bottom 5 by flip rate)
    top_outliers = q_rates.head(n_leave_out)["question_id"].tolist()
    bottom_outliers = q_rates.tail(n_leave_out)["question_id"].tolist()

    # c) ANOVA excluding top-5 outlier questions
    df_no_outliers = df[~df["question_id"].isin(top_outliers)]
    factorial_no_outliers = df_no_outliers[~df_no_outliers["is_control"]].dropna(
        subset=["flip"]
    )
    anova_robust = run_factorial_anova(factorial_no_outliers, "flip")

    # d) Leave-5-out bootstrap
    all_qids = df["question_id"].unique().tolist()
    boot_auth_effects = []
    boot_overall_rates = []

    for _ in range(n_bootstrap):
        drop_qids = rng.choice(all_qids, size=n_leave_out, replace=False)
        df_boot = df[~df["question_id"].isin(drop_qids)]
        nc = df_boot[~df_boot["is_control"]].dropna(subset=["flip"])

        if len(nc) == 0:
            continue

        boot_overall_rates.append(float(nc["flip"].mean()))
        high = nc[nc["authority_level"] == "high"]["flip"].mean()
        low = nc[nc["authority_level"] == "low"]["flip"].mean()
        boot_auth_effects.append(float(high - low))

    return {
        "per_question_rates": q_rates.to_dict("records"),
        "n_questions": len(q_rates),
        "top_outlier_qids": top_outliers,
        "bottom_outlier_qids": bottom_outliers,
        "anova_excluding_top_outliers": anova_robust,
        "leave5out_bootstrap": {
            "n_bootstrap": n_bootstrap,
            "n_leave_out": n_leave_out,
            "overall_flip_rate_mean": float(np.mean(boot_overall_rates)),
            "overall_flip_rate_std": float(np.std(boot_overall_rates)),
            "overall_flip_rate_ci95": [
                float(np.percentile(boot_overall_rates, 2.5)),
                float(np.percentile(boot_overall_rates, 97.5)),
            ],
            "auth_effect_mean": float(np.mean(boot_auth_effects)),
            "auth_effect_std": float(np.std(boot_auth_effects)),
            "auth_effect_ci95": [
                float(np.percentile(boot_auth_effects, 2.5)),
                float(np.percentile(boot_auth_effects, 97.5)),
            ],
        },
    }


def compute_steering_bootstrap_cis(
    steered_scores: Dict[str, pd.DataFrame],
    n_boot: int = 10000,
    seed: int = 42,
) -> Dict:
    """Compute bootstrap CIs for steered flip rates.

    Args:
        steered_scores: Dict mapping condition_key (e.g., "authority_alpha1.5")
            to DataFrames of scored steered conversations.
        n_boot: Number of bootstrap iterations.
        seed: Random seed.

    Returns:
        Dict mapping condition_key to {flip_rate, ci_lower, ci_upper}.
    """
    rng = np.random.RandomState(seed)
    results = {}

    for condition_key, score_df in steered_scores.items():
        flips = score_df["flip"].dropna().values.astype(float)
        if len(flips) == 0:
            continue

        rate = float(flips.mean())
        boot_means = np.array([
            rng.choice(flips, size=len(flips), replace=True).mean()
            for _ in range(n_boot)
        ])

        results[condition_key] = {
            "flip_rate": rate,
            "ci_lower": float(np.percentile(boot_means, 2.5)),
            "ci_upper": float(np.percentile(boot_means, 97.5)),
            "n_conversations": len(flips),
        }

    return results
