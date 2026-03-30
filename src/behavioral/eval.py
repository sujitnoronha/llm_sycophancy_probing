"""Behavioral evaluation: run all conversations and collect Turn 4 responses.

For each model, loads the pre-validated baseline answers, constructs 4-turn
conversations using the dataset templates, and generates Turn 4 (the target response).
Uses vLLM for batched inference.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger

from src.baseline.validate import get_chat_template_kwargs, get_effective_max_tokens, strip_thinking_tags


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
    use_system_role: bool = True,
) -> List[Dict[str, str]]:
    """Build the chat message list for generating Turn 4.

    Uses the pre-validated baseline answer as Turn 2 (assistant's correct answer).
    """
    if use_system_role:
        return [
            {"role": "system", "content": conversation["system_prompt"]},
            {"role": "user", "content": conversation["turn1_user"]},
            {"role": "assistant", "content": baseline_answer},
            {"role": "user", "content": conversation["turn3_user"]},
        ]
    else:
        # Fold system prompt into first user message for models like Gemma
        first_user = f"{conversation['system_prompt']}\n\n{conversation['turn1_user']}"
        return [
            {"role": "user", "content": first_user},
            {"role": "assistant", "content": baseline_answer},
            {"role": "user", "content": conversation["turn3_user"]},
        ]


def detect_system_role_support(llm, system_prompt: str, chat_kwargs: Optional[Dict] = None) -> bool:
    """Detect if the model's chat template supports the system role."""
    from vllm import SamplingParams

    test_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Hello"},
    ]
    try:
        llm.chat(
            [test_messages],
            sampling_params=SamplingParams(max_tokens=1),
            chat_template_kwargs=chat_kwargs or {},
        )
        logger.info("Model supports system role")
        return True
    except (ValueError, Exception) as e:
        if "System role not supported" in str(e):
            logger.info("Model does not support system role — folding into user message")
            return False
        raise


def run_behavioral_eval_vllm(
    llm,
    conversations: List[Dict],
    baseline: Dict[str, Dict],
    model_name: str,
    system_prompt: str,
    max_new_tokens: int = 512,
    hf_model_name: str = "",
) -> List[Dict]:
    """Run behavioral evaluation for one model across all conversations using vLLM.

    Args:
        llm: vllm.LLM instance.
        conversations: All 800 conversation templates.
        baseline: Baseline results mapping question_id → result.
        model_name: Name of the model (for metadata).
        system_prompt: System prompt (for detecting system role support).
        max_new_tokens: Max tokens for Turn 4 generation.
        hf_model_name: HuggingFace model name (for detecting thinking models).

    Returns:
        List of completed conversation dicts with Turn 4 filled in.
    """
    from vllm import SamplingParams

    chat_kwargs = get_chat_template_kwargs(hf_model_name) or {}
    if chat_kwargs:
        logger.info(f"Using chat_template_kwargs: {chat_kwargs}")

    use_system_role = detect_system_role_support(llm, system_prompt, chat_kwargs)

    # Filter conversations and build messages
    valid_convs = []
    all_messages = []
    skipped = 0

    for conv in conversations:
        question_id = conv["question_id"]

        if question_id not in baseline:
            logger.warning(f"No baseline for {question_id}, skipping")
            skipped += 1
            continue

        baseline_result = baseline[question_id]
        if not baseline_result["is_correct"]:
            skipped += 1
            continue

        baseline_answer = baseline_result["model_answer"]
        messages = build_messages_for_generation(conv, baseline_answer, use_system_role)

        valid_convs.append((conv, baseline_answer))
        all_messages.append(messages)

    logger.info(
        f"{model_name}: Generating Turn 4 for {len(valid_convs)} conversations "
        f"(skipped {skipped} baseline failures)"
    )

    effective_max_tokens = get_effective_max_tokens(max_new_tokens, hf_model_name)

    sampling_params = SamplingParams(
        max_tokens=effective_max_tokens,
        temperature=0.0,
        top_p=1.0,
    )

    # vLLM batched generation
    outputs = llm.chat(all_messages, sampling_params=sampling_params, chat_template_kwargs=chat_kwargs)

    # Build results
    results = []
    for (conv, baseline_answer), output in zip(valid_convs, outputs):
        generated_text = strip_thinking_tags(output.outputs[0].text.strip())
        n_tokens = len(output.outputs[0].token_ids)

        result = {
            **conv,
            "model_name": model_name,
            "turn2_assistant": baseline_answer,
            "turn4_assistant": generated_text,
            "generation_metadata": {
                "n_tokens_generated": n_tokens,
            },
        }
        results.append(result)

    logger.info(
        f"{model_name}: Generated {len(results)} Turn 4 responses, "
        f"skipped {skipped} (baseline failures)"
    )
    return results
