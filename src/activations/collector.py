"""Activation collector: caches residual stream activations at the last token of Turn 3.

For each model, collects activations at every layer for the 200-prompt probing subset.
Uses TransformerLens run_with_cache or nnsight as fallback.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from loguru import logger
from tqdm import tqdm


def select_probing_subset(
    conversations: List[Dict],
    scores: Dict[str, any],
    key_conditions: List[Dict],
) -> List[Dict]:
    """Select the 200-prompt probing subset from completed conversations.

    Selects 50 questions × 4 key conditions = 200 prompts per model.
    The 4 key conditions are specified in config.probing.key_conditions.

    Args:
        conversations: Completed conversations (with Turn 4) for one model.
        scores: Score data for the same conversations (for sycophancy labels).
        key_conditions: List of condition dicts specifying the 4 probing conditions.

    Returns:
        List of conversation dicts with attached labels.
    """
    scores_by_id = {}
    if isinstance(scores, list):
        for s in scores:
            scores_by_id[s.get("conversation_id", "")] = s
    elif hasattr(scores, "iterrows"):
        for _, row in scores.iterrows():
            scores_by_id[row["conversation_id"]] = row.to_dict()

    selected = []
    for conv in conversations:
        cond = conv.get("condition", {})
        authority = cond.get("authority", "")
        confidence = cond.get("confidence")
        domain_match = cond.get("domain_match")

        for key_cond in key_conditions:
            if (
                authority == key_cond.get("authority", key_cond.get("authority"))
                and confidence == key_cond.get("confidence")
                and domain_match == key_cond.get("domain_match")
            ):
                conv_id = conv.get("conversation_id", "")
                score = scores_by_id.get(conv_id, {})

                conv_with_labels = {
                    **conv,
                    "labels": {
                        "authority": 1 if authority == "high" else 0,
                        "sycophancy": int(score.get("flip", 0)) if score.get("flip") is not None else 0,
                        "confidence": 1 if confidence == "high" else 0,
                    },
                }
                selected.append(conv_with_labels)
                break

    logger.info(f"Selected {len(selected)} prompts for probing subset")
    return selected


def find_last_user_token_position(
    input_ids: torch.Tensor,
    tokenizer,
) -> int:
    """Find the index of the last token of the user's Turn 3 message.

    Strategy: tokenize the full conversation up to Turn 3 with add_generation_prompt=True.
    The last meaningful user token is just before the assistant turn-start marker.
    We search backwards from the end, skipping special/padding tokens.

    Args:
        input_ids: Tokenized conversation [seq_len].
        tokenizer: The model's tokenizer.

    Returns:
        Token index (0-based) of the last user content token.
    """
    # Get special token IDs to skip
    special_ids = set()
    for attr in ["pad_token_id", "eos_token_id", "bos_token_id"]:
        tok_id = getattr(tokenizer, attr, None)
        if tok_id is not None:
            special_ids.add(tok_id)

    # Also check for common assistant turn-start tokens
    # These vary by model but are typically the last few tokens before generation
    ids_list = input_ids.tolist()

    # The generation prompt tokens (added by add_generation_prompt=True)
    # are at the end of input_ids. We want the last token BEFORE those.
    # Heuristic: the last user token is typically 2-5 positions from the end
    # (after the assistant turn-start marker).

    # More robust: tokenize WITHOUT add_generation_prompt and compare lengths
    # But we don't have the messages here, so use a heuristic:
    # Find the last non-special token before the final special tokens

    # Walk backwards from end, skip any that are special tokens or common markers
    pos = len(ids_list) - 1

    # Skip the generation prompt tokens (typically 1-4 tokens at the end)
    # We look for the transition from generation-prompt tokens to user content
    # by checking if decoded text contains assistant markers
    while pos > 0:
        # Check if this token is part of a turn delimiter
        token_text = tokenizer.decode([ids_list[pos]], skip_special_tokens=False)
        # Common assistant turn markers
        if any(marker in token_text.lower() for marker in [
            "assistant", "<|start_header", "<|im_start|>", "[/inst]",
            "<start_of_turn>", "model", "<|assistant|>"
        ]):
            pos -= 1
            continue
        if ids_list[pos] in special_ids:
            pos -= 1
            continue
        break

    return max(pos, 0)


def collect_activations_transformerlens(
    model,
    tokenizer,
    probing_subset: List[Dict],
    n_layers: int,
    d_model: int,
    system_prompt: str,
) -> Tuple[torch.Tensor, Dict]:
    """Collect residual stream activations using TransformerLens.

    For each prompt in the probing subset:
    1. Build message list (Turns 1-3)
    2. Tokenize with chat template
    3. Run forward pass with activation caching
    4. Extract activation at last user token position for every layer
    5. Delete cache to free memory

    Returns:
        (activations, metadata) where:
        - activations: [n_prompts, n_layers, d_model] float32 tensor
        - metadata: dict with token positions, labels, etc.
    """
    n_prompts = len(probing_subset)
    activations = torch.zeros(n_prompts, n_layers, d_model, dtype=torch.float32)

    labels_authority = torch.zeros(n_prompts, dtype=torch.long)
    labels_sycophancy = torch.zeros(n_prompts, dtype=torch.long)
    labels_confidence = torch.zeros(n_prompts, dtype=torch.long)

    token_positions = []
    question_ids = []
    conditions = []
    domains = []

    for idx, conv in enumerate(tqdm(probing_subset, desc="Collecting activations")):
        # Build messages up to Turn 3 (no Turn 4 — we probe the pre-decision state)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": conv["turn1_user"]},
            {"role": "assistant", "content": conv.get("turn2_assistant", "")},
            {"role": "user", "content": conv["turn3_user"]},
        ]

        # Tokenize
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        tokens = tokenizer.encode(input_text, return_tensors="pt").to(model.cfg.device)

        # Find target position
        target_pos = find_last_user_token_position(tokens[0], tokenizer)
        token_positions.append(target_pos)

        # Run with cache (store on CPU to save GPU memory)
        with torch.no_grad():
            _, cache = model.run_with_cache(
                tokens,
                names_filter=lambda name: "resid_post" in name,
            )

        # Extract activations at target position for each layer
        for layer_idx in range(n_layers):
            act = cache["resid_post", layer_idx][0, target_pos, :].cpu().float()
            activations[idx, layer_idx, :] = act

        # Store labels
        labels = conv.get("labels", {})
        labels_authority[idx] = labels.get("authority", 0)
        labels_sycophancy[idx] = labels.get("sycophancy", 0)
        labels_confidence[idx] = labels.get("confidence", 0)

        # Store metadata
        question_ids.append(conv.get("question_id", ""))
        conditions.append(conv.get("condition", {}))
        domains.append(conv.get("domain", ""))

        # Free cache memory
        del cache
        if idx % 10 == 0:
            torch.cuda.empty_cache()

    metadata = {
        "token_positions": token_positions,
        "question_ids": question_ids,
        "conditions": conditions,
        "domains": domains,
        "n_layers": n_layers,
        "d_model": d_model,
    }

    labels_dict = {
        "authority": labels_authority,
        "sycophancy": labels_sycophancy,
        "confidence": labels_confidence,
    }

    logger.info(
        f"Collected activations: shape {activations.shape}, "
        f"size {activations.element_size() * activations.nelement() / 1e6:.1f} MB"
    )

    return activations, labels_dict, metadata


def save_activations(
    activations: torch.Tensor,
    labels: Dict[str, torch.Tensor],
    metadata: Dict,
    output_path: str,
) -> None:
    """Save activations, labels, and metadata to a .pt file."""
    torch.save(
        {
            "activations": activations,
            "labels": labels,
            "metadata": metadata,
        },
        output_path,
    )
    size_mb = Path(output_path).stat().st_size / 1e6
    logger.info(f"Saved activations to {output_path} ({size_mb:.1f} MB)")


def load_activations(path: str) -> Dict:
    """Load activations from a .pt file."""
    return torch.load(path, weights_only=False)
