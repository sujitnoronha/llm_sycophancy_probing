"""Core mechanistic analysis of probe weight directions.

Computes cosine similarity matrices, layer-wise trajectories, subspace
decompositions, cross-model comparisons, and permutation-based significance
tests for the relationship between authority, sycophancy, and confidence
probe directions.
"""

from typing import Dict, Tuple

import numpy as np
from loguru import logger
from scipy.spatial.distance import cosine as cosine_distance
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def compute_cosine_similarity_matrix(directions: Dict[str, np.ndarray]) -> np.ndarray:
    """Compute 3x3 pairwise cosine similarity matrix between probe directions.

    Args:
        directions: Dict with keys 'w_auth', 'w_syc', 'w_conf', each a
            unit-normalized numpy array of shape [d_model].

    Returns:
        3x3 numpy array where entry (i, j) is the cosine similarity between
        direction i and direction j. Row/column order: [w_auth, w_syc, w_conf].
    """
    keys = ["w_auth", "w_syc", "w_conf"]
    for key in keys:
        if key not in directions:
            raise ValueError(f"Missing required direction '{key}' in directions dict")

    vecs = [directions[k] for k in keys]
    n = len(vecs)
    sim_matrix = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        for j in range(n):
            if i == j:
                sim_matrix[i, j] = 1.0
            else:
                # scipy cosine returns DISTANCE, similarity = 1 - distance
                sim_matrix[i, j] = 1.0 - cosine_distance(vecs[i], vecs[j])

    logger.info(
        f"Cosine similarity matrix:\n"
        f"  auth-syc:  {sim_matrix[0, 1]:.4f}\n"
        f"  auth-conf: {sim_matrix[0, 2]:.4f}\n"
        f"  syc-conf:  {sim_matrix[1, 2]:.4f}"
    )

    return sim_matrix


def compute_layer_trajectory(
    weight_vectors_all_layers: Dict[str, np.ndarray],
    probe_a: str,
    probe_b: str,
) -> np.ndarray:
    """Compute layer-by-layer cosine similarity between two probe directions.

    At each layer L, computes cosine similarity between the weight vectors
    of probe_a and probe_b at that layer.

    Args:
        weight_vectors_all_layers: Dict mapping probe names to arrays of
            shape [n_layers, d_model]. Expected keys include 'w_authority',
            'w_sycophancy', 'w_confidence'.
        probe_a: Name of the first probe (e.g., 'w_authority').
        probe_b: Name of the second probe (e.g., 'w_sycophancy').

    Returns:
        Array of shape [n_layers] containing the cosine similarity at each layer.

    Raises:
        ValueError: If probe_a or probe_b not found in weight_vectors_all_layers.
    """
    if probe_a not in weight_vectors_all_layers:
        raise ValueError(
            f"Probe '{probe_a}' not found. Available: {list(weight_vectors_all_layers.keys())}"
        )
    if probe_b not in weight_vectors_all_layers:
        raise ValueError(
            f"Probe '{probe_b}' not found. Available: {list(weight_vectors_all_layers.keys())}"
        )

    vecs_a = weight_vectors_all_layers[probe_a]  # [n_layers, d_model]
    vecs_b = weight_vectors_all_layers[probe_b]  # [n_layers, d_model]

    if vecs_a.shape[0] != vecs_b.shape[0]:
        raise ValueError(
            f"Layer count mismatch: {probe_a} has {vecs_a.shape[0]} layers, "
            f"{probe_b} has {vecs_b.shape[0]} layers"
        )

    n_layers = vecs_a.shape[0]
    trajectory = np.zeros(n_layers, dtype=np.float64)

    for layer_idx in range(n_layers):
        w_a = vecs_a[layer_idx]
        w_b = vecs_b[layer_idx]

        # Skip layers where either vector is zero (e.g., degenerate probes)
        norm_a = np.linalg.norm(w_a)
        norm_b = np.linalg.norm(w_b)
        if norm_a < 1e-10 or norm_b < 1e-10:
            trajectory[layer_idx] = 0.0
            logger.warning(f"Near-zero weight vector at layer {layer_idx}, setting similarity to 0")
            continue

        trajectory[layer_idx] = 1.0 - cosine_distance(w_a, w_b)

    logger.info(
        f"Layer trajectory {probe_a} vs {probe_b}: "
        f"mean={trajectory.mean():.4f}, max={trajectory.max():.4f} "
        f"(layer {trajectory.argmax()}), min={trajectory.min():.4f} "
        f"(layer {trajectory.argmin()})"
    )

    return trajectory


def subspace_decomposition(
    w_syc: np.ndarray,
    w_auth: np.ndarray,
    w_conf: np.ndarray,
) -> Dict:
    """Decompose the sycophancy direction into authority, confidence, and residual components.

    Projects w_syc onto:
    (a) w_auth component: projection of w_syc onto w_auth
    (b) w_conf_orth component: projection of w_syc onto w_conf after orthogonalizing
        w_conf against w_auth (Gram-Schmidt)
    (c) residual: whatever remains after removing (a) and (b)

    Reports the variance fraction (squared norm of each component divided by
    squared norm of w_syc). For unit-normalized inputs these should sum to ~1.0.

    Args:
        w_syc: Sycophancy probe weight vector, shape [d_model]. Should be unit-normalized.
        w_auth: Authority probe weight vector, shape [d_model]. Should be unit-normalized.
        w_conf: Confidence probe weight vector, shape [d_model]. Should be unit-normalized.

    Returns:
        Dict with keys:
            - variance_auth: fraction of w_syc variance explained by w_auth
            - variance_conf: fraction explained by orthogonalized w_conf
            - variance_residual: fraction in the residual
            - component_auth: the w_auth component vector
            - component_conf_orth: the orthogonalized w_conf component vector
            - component_residual: the residual component vector
            - w_conf_orthogonalized: w_conf after Gram-Schmidt against w_auth
    """
    # Ensure float64 for numerical stability
    w_syc = np.asarray(w_syc, dtype=np.float64)
    w_auth = np.asarray(w_auth, dtype=np.float64)
    w_conf = np.asarray(w_conf, dtype=np.float64)

    syc_norm_sq = np.dot(w_syc, w_syc)
    if syc_norm_sq < 1e-15:
        logger.warning("w_syc has near-zero norm, returning zero decomposition")
        zero = np.zeros_like(w_syc)
        return {
            "variance_auth": 0.0,
            "variance_conf": 0.0,
            "variance_residual": 0.0,
            "component_auth": zero,
            "component_conf_orth": zero,
            "component_residual": zero,
            "w_conf_orthogonalized": zero,
        }

    # (a) Project w_syc onto w_auth
    # component_auth = (w_syc . w_auth) * w_auth  (for unit-norm w_auth)
    auth_norm_sq = np.dot(w_auth, w_auth)
    proj_coeff_auth = np.dot(w_syc, w_auth) / auth_norm_sq
    component_auth = proj_coeff_auth * w_auth

    # (b) Orthogonalize w_conf against w_auth via Gram-Schmidt
    conf_proj_on_auth = (np.dot(w_conf, w_auth) / auth_norm_sq) * w_auth
    w_conf_orth = w_conf - conf_proj_on_auth
    conf_orth_norm_sq = np.dot(w_conf_orth, w_conf_orth)

    if conf_orth_norm_sq < 1e-15:
        # w_conf is nearly parallel to w_auth, no independent conf direction
        logger.warning("w_conf is nearly parallel to w_auth after orthogonalization")
        component_conf_orth = np.zeros_like(w_syc)
    else:
        proj_coeff_conf = np.dot(w_syc, w_conf_orth) / conf_orth_norm_sq
        component_conf_orth = proj_coeff_conf * w_conf_orth

    # (c) Residual
    component_residual = w_syc - component_auth - component_conf_orth

    # Variance fractions (squared norms relative to ||w_syc||^2)
    variance_auth = float(np.dot(component_auth, component_auth) / syc_norm_sq)
    variance_conf = float(np.dot(component_conf_orth, component_conf_orth) / syc_norm_sq)
    variance_residual = float(np.dot(component_residual, component_residual) / syc_norm_sq)

    total = variance_auth + variance_conf + variance_residual
    logger.info(
        f"Subspace decomposition of w_syc:\n"
        f"  Authority component:  {variance_auth:.4f} ({variance_auth * 100:.1f}%)\n"
        f"  Confidence component: {variance_conf:.4f} ({variance_conf * 100:.1f}%)\n"
        f"  Residual component:   {variance_residual:.4f} ({variance_residual * 100:.1f}%)\n"
        f"  Total:                {total:.4f} (should be ~1.0)"
    )

    return {
        "variance_auth": variance_auth,
        "variance_conf": variance_conf,
        "variance_residual": variance_residual,
        "component_auth": component_auth,
        "component_conf_orth": component_conf_orth,
        "component_residual": component_residual,
        "w_conf_orthogonalized": w_conf_orth,
    }


def cross_model_direction_comparison(
    directions_by_model: Dict[str, Dict[str, np.ndarray]],
) -> Dict:
    """Compare authority probe directions across models sharing Llama architecture.

    For models built on the Llama architecture (llama3_8b and deepseek_r1_llama8b),
    compares the authority probe directions at matched layers using cosine similarity.

    Args:
        directions_by_model: Dict mapping model name to a dict with at least
            'w_authority' key, an array of shape [n_layers, d_model] containing
            the authority probe weight vectors at each layer.

    Returns:
        Dict with:
            - per_layer_cosine: array of shape [n_layers] with cosine similarity
              at each matched layer
            - mean_cosine: mean cosine similarity across layers
            - peak_cosine: max cosine similarity and corresponding layer
            - models_compared: tuple of model names compared

    Raises:
        ValueError: If the required Llama-based models are not present or have
            mismatched layer counts.
    """
    llama_models = ["llama3_8b", "deepseek_r1_llama8b"]

    missing = [m for m in llama_models if m not in directions_by_model]
    if missing:
        raise ValueError(
            f"Missing Llama-architecture models for comparison: {missing}. "
            f"Available: {list(directions_by_model.keys())}"
        )

    w_llama = directions_by_model["llama3_8b"]["w_authority"]
    w_deepseek = directions_by_model["deepseek_r1_llama8b"]["w_authority"]

    if w_llama.shape != w_deepseek.shape:
        raise ValueError(
            f"Shape mismatch: llama3_8b has {w_llama.shape}, "
            f"deepseek_r1_llama8b has {w_deepseek.shape}. "
            f"Cross-model comparison requires matched layer counts and d_model."
        )

    n_layers = w_llama.shape[0]
    per_layer_cosine = np.zeros(n_layers, dtype=np.float64)

    for layer_idx in range(n_layers):
        norm_l = np.linalg.norm(w_llama[layer_idx])
        norm_d = np.linalg.norm(w_deepseek[layer_idx])

        if norm_l < 1e-10 or norm_d < 1e-10:
            per_layer_cosine[layer_idx] = 0.0
            logger.warning(
                f"Near-zero weight vector at layer {layer_idx} for cross-model comparison"
            )
            continue

        per_layer_cosine[layer_idx] = 1.0 - cosine_distance(
            w_llama[layer_idx], w_deepseek[layer_idx]
        )

    mean_cosine = float(per_layer_cosine.mean())
    peak_layer = int(per_layer_cosine.argmax())
    peak_cosine = float(per_layer_cosine[peak_layer])

    logger.info(
        f"Cross-model direction comparison (llama3_8b vs deepseek_r1_llama8b):\n"
        f"  Mean cosine similarity: {mean_cosine:.4f}\n"
        f"  Peak cosine: {peak_cosine:.4f} at layer {peak_layer}\n"
        f"  Min cosine:  {per_layer_cosine.min():.4f} at layer {per_layer_cosine.argmin()}"
    )

    return {
        "per_layer_cosine": per_layer_cosine,
        "mean_cosine": mean_cosine,
        "peak_cosine": peak_cosine,
        "peak_layer": peak_layer,
        "models_compared": tuple(llama_models),
    }


def permutation_test_cosine(
    activations: np.ndarray,
    labels_a: np.ndarray,
    labels_b: np.ndarray,
    layer_idx: int,
    n_permutations: int = 1000,
    seed: int = 42,
    C: float = 1.0,
    max_iter: int = 1000,
) -> Dict:
    """Null hypothesis test for cosine similarity between two probe directions.

    Tests whether the observed cosine similarity between weight vectors of
    probes trained on labels_a and labels_b is significant, by comparing
    against a null distribution obtained by shuffling labels_a.

    Procedure:
    1. Train probes on true labels_a and labels_b at the specified layer.
    2. Compute cos(w_a, w_b) as the observed statistic.
    3. Repeat n_permutations times with shuffled labels_a, retraining probe_a
       each time, and computing cos(w_a_shuffled, w_b).
    4. p-value = fraction of null cos values >= |observed cos|.

    Args:
        activations: Array of shape [n_prompts, n_layers, d_model] or
            [n_prompts, d_model] (if pre-sliced to a single layer).
        labels_a: Binary labels for probe A, shape [n_prompts].
        labels_b: Binary labels for probe B, shape [n_prompts].
        layer_idx: Which layer to use for extracting activations.
        n_permutations: Number of permutation iterations (default 1000).
        seed: Random seed for reproducibility.
        C: Regularization parameter for logistic regression.
        max_iter: Maximum iterations for logistic regression solver.

    Returns:
        Dict with:
            - observed_cosine: the true cosine similarity
            - null_mean: mean of the null distribution
            - null_std: std of the null distribution
            - null_95th: 95th percentile of the null distribution
            - p_value: fraction of null values >= |observed|
            - n_permutations: number of permutations run
    """
    rng = np.random.RandomState(seed)

    # Extract activations at the target layer
    if activations.ndim == 3:
        X = activations[:, layer_idx, :]
    elif activations.ndim == 2:
        X = activations
    else:
        raise ValueError(f"Expected 2D or 3D activations, got shape {activations.shape}")

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    def _train_and_extract_direction(X_s: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Train a logistic regression probe and return the unit-normalized weight vector."""
        probe = LogisticRegression(
            C=C, solver="lbfgs", max_iter=max_iter, random_state=seed
        )
        probe.fit(X_s, labels)
        w = probe.coef_[0]
        # Convert back to original space
        w_original = w / scaler.scale_
        norm = np.linalg.norm(w_original)
        return w_original / norm if norm > 1e-10 else w_original

    # Train probe B on true labels (this stays fixed)
    w_b = _train_and_extract_direction(X_scaled, labels_b)

    # Train probe A on true labels
    w_a_true = _train_and_extract_direction(X_scaled, labels_a)

    # Observed cosine similarity
    observed_cosine = float(1.0 - cosine_distance(w_a_true, w_b))

    # Null distribution: shuffle labels_a, retrain probe_a, compute cosine with w_b
    null_cosines = np.zeros(n_permutations, dtype=np.float64)
    for i in range(n_permutations):
        shuffled_labels_a = rng.permutation(labels_a)
        w_a_shuffled = _train_and_extract_direction(X_scaled, shuffled_labels_a)
        null_cosines[i] = 1.0 - cosine_distance(w_a_shuffled, w_b)

        if (i + 1) % 200 == 0:
            logger.debug(f"Permutation test: {i + 1}/{n_permutations} complete")

    # Two-sided p-value: fraction of null values with |cos| >= |observed|
    p_value = float(np.mean(np.abs(null_cosines) >= np.abs(observed_cosine)))

    logger.info(
        f"Permutation test (layer {layer_idx}): "
        f"observed cos={observed_cosine:.4f}, "
        f"null mean={null_cosines.mean():.4f} +/- {null_cosines.std():.4f}, "
        f"p={p_value:.4f}"
    )

    return {
        "observed_cosine": observed_cosine,
        "null_mean": float(null_cosines.mean()),
        "null_std": float(null_cosines.std()),
        "null_95th": float(np.percentile(np.abs(null_cosines), 95)),
        "p_value": p_value,
        "n_permutations": n_permutations,
    }
