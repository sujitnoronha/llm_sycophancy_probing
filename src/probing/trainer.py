"""Probe trainer: trains linear probes for authority, sycophancy, and confidence detection.

Trains LogisticRegression probes at every layer for each model, extracts
the weight vectors (directions), and runs permutation tests for significance.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler


def prepare_probe_data(
    activations: np.ndarray,
    labels: np.ndarray,
    domains: List[str],
    train_fraction: float = 0.8,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split activations into train/test, stratified by label and domain.

    Args:
        activations: [n_prompts, d_model] activations at a single layer.
        labels: [n_prompts] binary labels.
        domains: [n_prompts] domain strings for stratification.
        train_fraction: Fraction for training.
        seed: Random seed.

    Returns:
        (X_train, X_test, y_train, y_test, train_idx, test_idx)
    """
    # Create compound stratification key
    strat_labels = np.array([f"{y}_{d}" for y, d in zip(labels, domains)])

    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=1 - train_fraction, random_state=seed
    )

    try:
        train_idx, test_idx = next(splitter.split(activations, strat_labels))
    except ValueError:
        # Fallback to label-only stratification if compound fails
        logger.warning("Compound stratification failed, using label-only")
        splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=1 - train_fraction, random_state=seed
        )
        train_idx, test_idx = next(splitter.split(activations, labels))

    return (
        activations[train_idx],
        activations[test_idx],
        labels[train_idx],
        labels[test_idx],
        train_idx,
        test_idx,
    )


def train_probe_single_layer(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    C: float = 1.0,
    solver: str = "lbfgs",
    max_iter: int = 1000,
    seed: int = 42,
    class_weight: Optional[str] = None,
) -> Dict:
    """Train a single logistic regression probe at one layer.

    Returns dict with metrics, weight vector, and trained model.
    """
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train probe
    probe = LogisticRegression(
        C=C,
        solver=solver,
        max_iter=max_iter,
        random_state=seed,
        class_weight=class_weight,
    )
    probe.fit(X_train_scaled, y_train)

    # Evaluate
    y_pred = probe.predict(X_test_scaled)
    y_proba = probe.predict_proba(X_test_scaled)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
    }

    # AUC-ROC (only if both classes present in test set)
    if len(np.unique(y_test)) > 1:
        metrics["auc_roc"] = float(roc_auc_score(y_test, y_proba[:, 1]))
    else:
        metrics["auc_roc"] = None

    # Extract direction in ORIGINAL (unscaled) activation space
    # Probe learned: P(y=1|x) = sigmoid(w_scaled . scaler(x) + b)
    # w_scaled . ((x - mean) / std) = (w_scaled / std) . x + const
    # So direction in original space: w_original = w_scaled / std
    w_scaled = probe.coef_[0]
    w_original = w_scaled / scaler.scale_
    w_norm = np.linalg.norm(w_original)
    w_normalized = w_original / w_norm if w_norm > 0 else w_original

    return {
        "metrics": metrics,
        "weight_vector": w_normalized,
        "weight_vector_unnormalized": w_original,
        "weight_norm": float(w_norm),
        "probe": probe,
        "scaler": scaler,
    }


def determine_class_weight(labels: np.ndarray) -> Optional[str]:
    """Determine if class_weight='balanced' is needed."""
    if len(labels) == 0:
        return None
    positive_rate = labels.mean()
    if positive_rate < 0.15 or positive_rate > 0.85:
        logger.info(f"Label imbalance detected ({positive_rate:.2f}), using balanced weights")
        return "balanced"
    return None


def train_all_probes_single_model(
    activations: np.ndarray,
    labels: Dict[str, np.ndarray],
    domains: List[str],
    config_probing: Dict,
) -> Dict:
    """Train probes A (authority), B (sycophancy), C (confidence) at every layer.

    Args:
        activations: [n_prompts, n_layers, d_model] tensor.
        labels: Dict with 'authority', 'sycophancy', 'confidence' arrays.
        domains: List of domain strings for stratification.
        config_probing: Probing config dict.

    Returns:
        Comprehensive results dict with accuracy curves, weight vectors, peak info.
    """
    n_prompts, n_layers, d_model = activations.shape
    probe_types = {
        "authority": labels["authority"],
        "sycophancy": labels["sycophancy"],
        "confidence": labels["confidence"],
    }

    results = {
        "n_prompts": n_prompts,
        "n_layers": n_layers,
        "d_model": d_model,
        "probes": {},
    }

    for probe_name, probe_labels in probe_types.items():
        logger.info(f"Training {probe_name} probe across {n_layers} layers")

        # Check label balance
        n_positive = int(probe_labels.sum())
        n_negative = len(probe_labels) - n_positive
        logger.info(f"  Labels: {n_positive} positive, {n_negative} negative")

        if n_positive < 5 or n_negative < 5:
            logger.warning(f"  Insufficient label diversity for {probe_name} probe, skipping")
            results["probes"][probe_name] = {
                "error": "insufficient_label_diversity",
                "n_positive": n_positive,
                "n_negative": n_negative,
            }
            continue

        class_weight = determine_class_weight(probe_labels)

        layer_results = []
        weight_vectors = np.zeros((n_layers, d_model), dtype=np.float32)

        for layer_idx in range(n_layers):
            X = activations[:, layer_idx, :]

            X_train, X_test, y_train, y_test, _, _ = prepare_probe_data(
                X, probe_labels, domains,
                train_fraction=config_probing.get("train_fraction", 0.8),
                seed=config_probing.get("seed", 42),
            )

            result = train_probe_single_layer(
                X_train, y_train, X_test, y_test,
                C=config_probing.get("probe_regularization", 1.0),
                solver=config_probing.get("solver", "lbfgs"),
                max_iter=config_probing.get("max_iter", 1000),
                seed=config_probing.get("seed", 42),
                class_weight=class_weight,
            )

            layer_results.append(result["metrics"])
            weight_vectors[layer_idx] = result["weight_vector"]

        # Find peak accuracy layer
        accuracies = [r["accuracy"] for r in layer_results]
        peak_layer = int(np.argmax(accuracies))
        peak_accuracy = accuracies[peak_layer]

        results["probes"][probe_name] = {
            "layer_results": layer_results,
            "weight_vectors": weight_vectors,
            "peak_layer": peak_layer,
            "peak_accuracy": peak_accuracy,
            "n_positive": n_positive,
            "n_negative": n_negative,
            "class_weight": class_weight,
        }

        logger.info(
            f"  {probe_name}: peak accuracy {peak_accuracy:.3f} at layer {peak_layer}"
        )

    return results


def run_permutation_test(
    activations: np.ndarray,
    labels: np.ndarray,
    domains: List[str],
    layer_idx: int,
    n_permutations: int = 1000,
    config_probing: Optional[Dict] = None,
) -> Dict:
    """Permutation test for probe accuracy significance.

    Shuffles labels n_permutations times, re-trains probe each time,
    computes null distribution of accuracy.

    Returns dict with observed accuracy, null distribution stats, and p-value.
    """
    if config_probing is None:
        config_probing = {}

    X = activations[:, layer_idx, :]
    rng = np.random.RandomState(42)

    # Train on true labels
    X_train, X_test, y_train, y_test, _, _ = prepare_probe_data(
        X, labels, domains,
        train_fraction=config_probing.get("train_fraction", 0.8),
    )
    true_result = train_probe_single_layer(X_train, y_train, X_test, y_test)
    observed_acc = true_result["metrics"]["accuracy"]

    # Null distribution
    null_accs = []
    for i in range(n_permutations):
        shuffled_labels = rng.permutation(labels)
        X_train_s, X_test_s, y_train_s, y_test_s, _, _ = prepare_probe_data(
            X, shuffled_labels, domains,
            train_fraction=config_probing.get("train_fraction", 0.8),
            seed=42 + i,
        )
        null_result = train_probe_single_layer(
            X_train_s, y_train_s, X_test_s, y_test_s,
            seed=42 + i,
        )
        null_accs.append(null_result["metrics"]["accuracy"])

    null_accs = np.array(null_accs)
    p_value = float(np.mean(null_accs >= observed_acc))

    return {
        "observed_accuracy": float(observed_acc),
        "null_mean": float(null_accs.mean()),
        "null_std": float(null_accs.std()),
        "null_95th": float(np.percentile(null_accs, 95)),
        "p_value": p_value,
        "n_permutations": n_permutations,
    }
