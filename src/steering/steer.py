"""Activation steering module for causal intervention experiments.

Applies linear perturbations along probe-discovered directions to the
model's residual stream during generation. Measures changes in sycophantic
behavior (persona_sycophancy_gap, base_sycophancy_rate) to establish
causal links between internal representations and behavioral outcomes.
"""

import json
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
from loguru import logger

from src.core.config import Config, ModelConfig, get_model_config
from src.core.models import load_model_context
from src.core.utils import log_gpu_memory


def generate_random_direction(d_model: int, seed: int = 12345) -> np.ndarray:
    """Generate a random unit vector for the steering control baseline.

    Used to show that arbitrary perturbations to the residual stream do NOT
    reduce sycophancy, confirming that the authority direction is causally special.

    Args:
        d_model: Dimensionality of the model's residual stream.
        seed: Random seed for reproducibility.

    Returns:
        Unit-normalized numpy array of shape [d_model].
    """
    rng = np.random.RandomState(seed)
    direction = rng.randn(d_model).astype(np.float64)
    norm = np.linalg.norm(direction)
    return direction / norm


def create_steering_hook(
    direction: np.ndarray,
    alpha: float,
) -> Callable:
    """Create a TransformerLens hook function that steers the residual stream.

    The hook subtracts alpha * direction from the residual stream activation
    at ALL token positions. This implements activation subtraction steering:
    positive alpha removes the direction (e.g., removing sycophancy),
    negative alpha amplifies it.

    Args:
        direction: Unit-normalized direction vector, shape [d_model].
            Will be converted to a tensor matching the activation's
            dtype and device at hook time.
        alpha: Steering strength. Positive values subtract the direction,
            negative values add it.

    Returns:
        A callable hook function with signature
        (activation: torch.Tensor, hook) -> torch.Tensor
        where activation has shape [batch, seq_len, d_model].
    """
    # Pre-convert to tensor (will be moved to correct device/dtype at first call)
    direction_tensor = torch.from_numpy(
        np.asarray(direction, dtype=np.float64)
    ).clone()

    _cached_tensor = {}

    def hook_fn(activation: torch.Tensor, hook) -> torch.Tensor:
        """Subtract alpha * direction from residual stream at all positions."""
        # Cache the direction tensor on the correct device/dtype on first call
        cache_key = (activation.device, activation.dtype)
        if cache_key not in _cached_tensor:
            _cached_tensor[cache_key] = direction_tensor.to(
                device=activation.device, dtype=activation.dtype
            )

        d = _cached_tensor[cache_key]

        # activation shape: [batch, seq_len, d_model]
        # direction shape: [d_model] -> broadcast across batch and seq_len
        activation = activation - alpha * d

        return activation

    return hook_fn


def run_steered_generation(
    model,
    tokenizer,
    messages: List[Dict[str, str]],
    direction: np.ndarray,
    alpha: float,
    target_layer: int,
    max_new_tokens: int = 512,
    config: Optional[Config] = None,
) -> str:
    """Generate Turn 4 text with activation steering active.

    Registers a steering hook at the specified layer's residual stream
    post-attention position and generates text using the model's
    standard generation pipeline.

    Args:
        model: A TransformerLens HookedTransformer model.
        tokenizer: The model's tokenizer.
        messages: Chat message list (system, user Turn 1, assistant Turn 2,
            user Turn 3). The model will generate the assistant's Turn 4.
        direction: Unit-normalized steering direction, shape [d_model].
        alpha: Steering strength coefficient.
        target_layer: Layer index at which to apply the steering hook.
        max_new_tokens: Maximum tokens to generate for Turn 4.
        config: Optional global config for generation parameters.

    Returns:
        The generated Turn 4 text string.
    """
    hook_name = f"blocks.{target_layer}.hook_resid_post"
    hook_fn = create_steering_hook(direction, alpha)

    # Tokenize the conversation
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    # Move to model device
    if hasattr(model, "cfg"):
        device = model.cfg.device
    else:
        device = next(model.parameters()).device
    input_ids = input_ids.to(device)

    # Generate with the steering hook active
    with model.hooks(fwd_hooks=[(hook_name, hook_fn)]):
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

    # Decode only the newly generated tokens
    generated_ids = output_ids[0, input_ids.shape[1]:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    return generated_text


def run_steered_eval(
    model,
    tokenizer,
    conversations: List[Dict],
    direction: np.ndarray,
    alpha: float,
    target_layer: int,
    config: Config,
) -> List[Dict]:
    """Run behavioral evaluation on all conversations with steering active.

    For each conversation, builds the message list (Turns 1-3), generates
    a steered Turn 4, and returns the completed conversation dicts with
    metadata about the steering intervention.

    Args:
        model: A TransformerLens HookedTransformer model.
        tokenizer: The model's tokenizer.
        conversations: List of conversation dicts, each containing
            'system_prompt', 'turn1_user', 'turn2_assistant', 'turn3_user',
            and condition metadata.
        direction: Unit-normalized steering direction, shape [d_model].
        alpha: Steering strength coefficient.
        target_layer: Layer index for steering hook.
        config: Global config (for generation parameters).

    Returns:
        List of completed conversation dicts with steered Turn 4 and
        steering metadata attached.
    """
    results = []

    for idx, conv in enumerate(conversations):
        # Build message list for Turn 4 generation
        messages = [
            {"role": "system", "content": conv.get("system_prompt", config.experiment.system_prompt)},
            {"role": "user", "content": conv["turn1_user"]},
            {"role": "assistant", "content": conv["turn2_assistant"]},
            {"role": "user", "content": conv["turn3_user"]},
        ]

        start_time = time.time()

        try:
            generated_text = run_steered_generation(
                model=model,
                tokenizer=tokenizer,
                messages=messages,
                direction=direction,
                alpha=alpha,
                target_layer=target_layer,
                max_new_tokens=config.experiment.max_new_tokens_turn4,
                config=config,
            )
        except Exception as e:
            logger.error(f"Generation failed for conversation {idx}: {e}")
            generated_text = f"[GENERATION_ERROR: {e}]"

        wall_time = time.time() - start_time

        result = {
            **conv,
            "turn4_assistant_steered": generated_text,
            "steering_metadata": {
                "alpha": alpha,
                "target_layer": target_layer,
                "wall_time_seconds": round(wall_time, 3),
            },
        }
        results.append(result)

        if (idx + 1) % 50 == 0:
            logger.info(f"  Steered generation: {idx + 1}/{len(conversations)} complete")

    logger.info(
        f"Steered eval complete: {len(results)} conversations, "
        f"alpha={alpha}, layer={target_layer}"
    )

    return results


def _load_probe_directions(output_dir: str, model_name: str) -> Dict:
    """Load probe weight vectors and peak layer info for steering.

    Args:
        output_dir: Base output directory.
        model_name: Model name.

    Returns:
        Dict mapping direction names ('authority', 'sycophancy', 'confidence')
        to dicts with 'direction' (numpy array) and 'peak_layer' (int).
    """
    probe_dir = Path(output_dir) / "probing" / model_name
    wv_path = probe_dir / "weight_vectors.pt"
    summary_path = probe_dir / "probe_summary.json"

    if not wv_path.exists():
        raise FileNotFoundError(f"Weight vectors not found at {wv_path}")

    weight_vectors = torch.load(str(wv_path), weights_only=False)

    with open(summary_path) as f:
        summary = json.load(f)

    directions = {}
    for probe_name in ["authority", "sycophancy", "confidence"]:
        wv_key = f"w_{probe_name}"
        peak_key = f"peak_layer_{probe_name}"

        if wv_key not in weight_vectors:
            logger.warning(f"Missing weight vectors for {probe_name}")
            continue

        w_all = weight_vectors[wv_key].numpy()  # [n_layers, d_model]

        # Get peak layer
        if peak_key in weight_vectors:
            peak_layer = int(weight_vectors[peak_key])
        elif probe_name in summary and "peak_layer" in summary[probe_name]:
            peak_layer = summary[probe_name]["peak_layer"]
        else:
            peak_layer = w_all.shape[0] // 2
            logger.warning(f"No peak layer for {probe_name}, using layer {peak_layer}")

        # Extract direction at peak layer, ensure unit normalization
        w = w_all[peak_layer].astype(np.float64)
        norm = np.linalg.norm(w)
        if norm > 1e-10:
            w = w / norm

        directions[probe_name] = {
            "direction": w,
            "peak_layer": peak_layer,
            "all_layers": w_all,
        }

        logger.info(
            f"Loaded {probe_name} direction: peak layer {peak_layer}, "
            f"d_model={w.shape[0]}"
        )

    return directions


def _load_conversations(output_dir: str, model_name: str) -> List[Dict]:
    """Load completed behavioral eval conversations for a model.

    Args:
        output_dir: Base output directory.
        model_name: Model name.

    Returns:
        List of conversation dicts with Turn 2 already filled in.
    """
    conv_path = Path(output_dir) / "behavioral" / f"{model_name}_conversations.jsonl"
    if not conv_path.exists():
        raise FileNotFoundError(
            f"Conversations not found at {conv_path}. Run behavioral eval first."
        )

    conversations = []
    with open(conv_path) as f:
        for line in f:
            if line.strip():
                conversations.append(json.loads(line))

    logger.info(f"Loaded {len(conversations)} conversations for {model_name}")
    return conversations


def run_full_sweep(
    model_name: str,
    config: Config,
) -> Dict:
    """Run the full activation steering sweep for one model.

    Loads the model once, loads all probe directions, then iterates over
    3 directions x 6 alpha values = 18 steered evaluations. Reuses the
    model instance and swaps hooks between runs.

    Args:
        model_name: Name of the model to steer.
        config: Global config with steering parameters.

    Returns:
        Summary dict with per-condition metrics including
        persona_sycophancy_gap and base_sycophancy_rate for each
        (direction, alpha) combination.
    """
    model_cfg = get_model_config(config, model_name)

    # Load probe directions and conversations before loading the model
    logger.info(f"Loading probe directions for {model_name}")
    probe_directions = _load_probe_directions(config.output_dir, model_name)

    logger.info(f"Loading conversations for {model_name}")
    conversations = _load_conversations(config.output_dir, model_name)

    # Determine which directions and alphas to sweep
    direction_names = config.steering.directions
    alpha_values = config.steering.alpha_values

    logger.info(
        f"Steering sweep for {model_name}: "
        f"{len(direction_names)} directions x {len(alpha_values)} alphas = "
        f"{len(direction_names) * len(alpha_values)} conditions"
    )

    sweep_results = {
        "model_name": model_name,
        "direction_names": direction_names,
        "alpha_values": alpha_values,
        "n_conversations": len(conversations),
        "conditions": {},
    }

    # Load model once, run all conditions
    with load_model_context(model_cfg, config, need_hooks=True) as (model, tokenizer):
        log_gpu_memory(f"Model loaded for steering ({model_name}): ")

        for dir_name in direction_names:
            if dir_name not in probe_directions:
                logger.warning(f"Direction '{dir_name}' not available, skipping")
                continue

            direction_info = probe_directions[dir_name]
            direction = direction_info["direction"]

            # Determine target layer based on config
            if config.steering.layer_selection == "peak_probe_accuracy":
                target_layer = direction_info["peak_layer"]
            else:
                target_layer = model_cfg.n_layers // 2
                logger.warning(
                    f"Unknown layer_selection '{config.steering.layer_selection}', "
                    f"using middle layer {target_layer}"
                )

            for alpha in alpha_values:
                condition_key = f"{dir_name}_alpha{alpha}"
                logger.info(
                    f"Running steered eval: {condition_key} at layer {target_layer}"
                )

                steered_conversations = run_steered_eval(
                    model=model,
                    tokenizer=tokenizer,
                    conversations=conversations,
                    direction=direction,
                    alpha=alpha,
                    target_layer=target_layer,
                    config=config,
                )

                # Save steered conversations to JSONL for downstream scoring
                conv_output_dir = Path(config.output_dir) / "steering" / model_name
                conv_output_dir.mkdir(parents=True, exist_ok=True)
                conv_path = conv_output_dir / f"steered_{condition_key}.jsonl"
                with open(conv_path, "w") as f:
                    for conv in steered_conversations:
                        f.write(json.dumps(conv, default=str) + "\n")
                logger.info(f"Saved {len(steered_conversations)} steered convs to {conv_path}")

                # Compute summary metrics from steered responses
                metrics = _compute_steering_metrics(steered_conversations)

                sweep_results["conditions"][condition_key] = {
                    "direction": dir_name,
                    "alpha": alpha,
                    "target_layer": target_layer,
                    "n_generated": len(steered_conversations),
                    "metrics": metrics,
                }

                logger.info(
                    f"  {condition_key}: "
                    f"persona_sycophancy_gap={metrics.get('persona_sycophancy_gap', 'N/A')}, "
                    f"base_sycophancy_rate={metrics.get('base_sycophancy_rate', 'N/A')}"
                )

        log_gpu_memory(f"After steering sweep ({model_name}): ")

    return sweep_results


def _compute_steering_metrics(conversations: List[Dict]) -> Dict:
    """Compute behavioral metrics from steered conversations.

    Computes persona_sycophancy_gap (difference in sycophancy rate between
    high-authority and low-authority conditions) and base_sycophancy_rate
    (overall rate across all conditions).

    Note: These are placeholder metrics computed from the conversation
    metadata. Full scoring requires running the LLM judge (src.behavioral.scorer),
    which is done as a separate step after steering.

    Args:
        conversations: List of steered conversation dicts.

    Returns:
        Dict with summary metrics.
    """
    if not conversations:
        return {
            "persona_sycophancy_gap": None,
            "base_sycophancy_rate": None,
            "n_conversations": 0,
        }

    # Group conversations by authority level
    by_authority = {}
    for conv in conversations:
        condition = conv.get("condition", {})
        authority = condition.get("authority", "unknown")
        if authority not in by_authority:
            by_authority[authority] = []
        by_authority[authority].append(conv)

    metrics = {
        "n_conversations": len(conversations),
        "n_by_authority": {k: len(v) for k, v in by_authority.items()},
        "n_generation_errors": sum(
            1 for c in conversations
            if c.get("turn4_assistant_steered", "").startswith("[GENERATION_ERROR")
        ),
    }

    # Placeholder: actual sycophancy scoring requires the LLM judge.
    # We report the structure for downstream scoring to fill in.
    metrics["persona_sycophancy_gap"] = None
    metrics["base_sycophancy_rate"] = None
    metrics["requires_scoring"] = True

    return metrics
