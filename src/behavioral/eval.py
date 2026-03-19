"""Behavioral evaluation: run all 4,000 conversations and collect Turn 4 responses.

For each model, loads the pre-validated baseline answers, constructs 4-turn
conversations using the dataset templates, and generates Turn 4 (the target response).
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
from loguru import logger
from tqdm import tqdm


def load_baseline(baseline_dir: str, model_name: str) -> Dict[str, Dict]:
    """Load baseline validation results for a model.

    Returns dict mapping question_id → baseline result (including model_answer).
    """
    path = Path(baseline_dir) / f"{model_name}_baseline.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Baseline not found for {model_name} at {path}. Run baseline validation first."
        )
    with open(path) as f:
        results = json.load(f)
    return {r["question_id"]: r for r in results}


def build_messages_for_generation(
    conversation: Dict,
    baseline_answer: str,
) -> List[Dict[str, str]]:
    """Build the chat message list for generating Turn 4.

    Uses the pre-validated baseline answer as Turn 2 (assistant's correct answer).
    """
    return [
        {"role": "system", "content": conversation["system_prompt"]},
        {"role": "user", "content": conversation["turn1_user"]},
        {"role": "assistant", "content": baseline_answer},
        {"role": "user", "content": conversation["turn3_user"]},
    ]


def generate_turn4(
    model,
    tokenizer,
    messages: List[Dict[str, str]],
    max_new_tokens: int = 512,
) -> Dict:
    """Generate Turn 4 response from a model given the first 3 turns.

    Returns dict with generated text and metadata.
    """
    # Apply chat template
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(model.device)

    start_time = time.time()

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    wall_time = time.time() - start_time

    # Decode only the generated tokens
    generated_ids = output_ids[0, input_ids.shape[1]:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    return {
        "text": generated_text,
        "n_tokens_generated": len(generated_ids),
        "wall_time_seconds": round(wall_time, 3),
    }


def run_behavioral_eval(
    model,
    tokenizer,
    conversations: List[Dict],
    baseline: Dict[str, Dict],
    model_name: str,
    max_new_tokens: int = 512,
) -> List[Dict]:
    """Run behavioral evaluation for one model across all conversations.

    Args:
        model: The loaded model.
        tokenizer: The model's tokenizer.
        conversations: All 800 conversation templates.
        baseline: Baseline results mapping question_id → result.
        model_name: Name of the model (for metadata).
        max_new_tokens: Max tokens for Turn 4 generation.

    Returns:
        List of completed conversation dicts with Turn 4 filled in.
    """
    results = []
    skipped = 0

    for conv in tqdm(conversations, desc=f"Eval {model_name}"):
        question_id = conv["question_id"]

        # Skip questions the model got wrong in baseline
        if question_id not in baseline:
            logger.warning(f"No baseline for {question_id}, skipping")
            skipped += 1
            continue

        baseline_result = baseline[question_id]
        if not baseline_result["is_correct"]:
            skipped += 1
            continue

        # Build messages with the pre-validated correct answer as Turn 2
        baseline_answer = baseline_result["model_answer"]
        messages = build_messages_for_generation(conv, baseline_answer)

        # Generate Turn 4
        generation = generate_turn4(model, tokenizer, messages, max_new_tokens)

        # Build complete result
        result = {
            **conv,
            "model_name": model_name,
            "turn2_assistant": baseline_answer,
            "turn4_assistant": generation["text"],
            "generation_metadata": {
                "n_tokens_generated": generation["n_tokens_generated"],
                "wall_time_seconds": generation["wall_time_seconds"],
            },
        }
        results.append(result)

    logger.info(
        f"{model_name}: Generated {len(results)} Turn 4 responses, "
        f"skipped {skipped} (baseline failures)"
    )
    return results
