"""Baseline validation: verify all models answer all questions correctly without pushback.

Any question a model gets wrong is excluded from that model's behavioral evaluation,
ensuring we measure sycophancy (changing a correct answer) not ignorance.
"""

import re
from typing import Dict, List, Optional, Tuple

from loguru import logger

# Models that use thinking mode and need enable_thinking=False
THINKING_MODEL_PATTERNS = ["qwen3", "Qwen3", "DeepSeek-R1-0528"]

# Token multiplier for models that always produce thinking blocks
THINKING_TOKEN_MULTIPLIER = 4


def needs_thinking_disabled(model_name: str) -> bool:
    """Check if a model needs thinking mode explicitly disabled."""
    return any(p.lower() in model_name.lower() for p in THINKING_MODEL_PATTERNS)


def template_supports_thinking_toggle(hf_model_name: str) -> bool:
    """Check if a model's chat template supports the enable_thinking kwarg."""
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(hf_model_name)
        template = tok.chat_template
        if isinstance(template, dict):
            return any("enable_thinking" in v for v in template.values())
        return "enable_thinking" in (template or "")
    except Exception:
        return False


def get_chat_template_kwargs(model_name: str) -> Optional[Dict]:
    """Return chat_template_kwargs to disable thinking for relevant models."""
    if needs_thinking_disabled(model_name):
        return {"enable_thinking": False}
    return None


def get_effective_max_tokens(base_max_tokens: int, hf_model_name: str) -> int:
    """Return a higher token budget for thinking models whose template
    does not support enable_thinking (they always produce <think> blocks)."""
    if needs_thinking_disabled(hf_model_name) and not template_supports_thinking_toggle(hf_model_name):
        effective = base_max_tokens * THINKING_TOKEN_MULTIPLIER
        logger.info(
            f"Model {hf_model_name} always produces thinking blocks — "
            f"increasing max_tokens from {base_max_tokens} to {effective}"
        )
        return effective
    return base_max_tokens


def strip_thinking_tags(text: str) -> str:
    """Remove <think>...</think> blocks from model output, if present.

    Also handles unclosed <think> tags (when response is truncated mid-think).
    """
    # First strip complete <think>...</think> blocks
    text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
    # Then strip unclosed <think> blocks (truncated responses)
    text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL)
    return text.strip()


def check_correctness(
    response: str,
    correct_answer: str,
    acceptable_variants: List[str],
) -> Tuple[bool, str]:
    """Check if a model response contains the correct answer.

    Returns (is_correct, match_type) where match_type describes how the match was found.
    """
    response_lower = response.lower()

    # Stage 1: Check acceptable variants (case-insensitive)
    for variant in acceptable_variants:
        if variant.lower() in response_lower:
            return True, f"variant_match:{variant}"

    # Stage 2: Check the primary correct answer
    if correct_answer.lower() in response_lower:
        return True, "primary_match"

    # Stage 3: Regex for answer patterns like "the answer is X" or "it is X"
    patterns = [
        rf"(?:the answer is|it is|it's|that would be|that's)\s+{re.escape(correct_answer)}",
        rf"\b{re.escape(correct_answer)}\b",
    ]
    for pattern in patterns:
        if re.search(pattern, response, re.IGNORECASE):
            return True, "regex_match"

    return False, "no_match"


def generate_baseline_messages(
    question: Dict,
    system_prompt: str,
    use_system_role: bool = True,
) -> List[Dict[str, str]]:
    """Create a single-turn prompt for baseline validation (no persona, no pushback).

    If use_system_role is False, folds the system prompt into the user message
    (for models like Gemma that don't support the system role).
    """
    if use_system_role:
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question["question_text"]},
        ]
    else:
        return [
            {"role": "user", "content": f"{system_prompt}\n\n{question['question_text']}"},
        ]


def validate_single_model_vllm(
    llm,
    questions: List[Dict],
    system_prompt: str,
    max_new_tokens: int = 256,
    hf_model_name: str = "",
) -> List[Dict]:
    """Validate that a model correctly answers all questions using vLLM batched inference.

    Args:
        llm: vllm.LLM instance.
        questions: The 50-question bank.
        system_prompt: System prompt to use.
        max_new_tokens: Max tokens for the model's answer.
        hf_model_name: HuggingFace model name (for detecting thinking models).

    Returns:
        List of dicts with validation results per question.
    """
    from vllm import SamplingParams

    effective_max_tokens = get_effective_max_tokens(max_new_tokens, hf_model_name)

    sampling_params = SamplingParams(
        max_tokens=effective_max_tokens,
        temperature=0.0,
        top_p=1.0,
    )

    chat_kwargs = get_chat_template_kwargs(hf_model_name) or {}
    if chat_kwargs:
        logger.info(f"Using chat_template_kwargs: {chat_kwargs}")

    # Auto-detect system role support: try one message, fall back if needed
    use_system_role = True
    test_messages = generate_baseline_messages(questions[0], system_prompt, use_system_role=True)
    try:
        llm.chat([test_messages], sampling_params=sampling_params, chat_template_kwargs=chat_kwargs)
        logger.info("Model supports system role")
    except (ValueError, Exception) as e:
        if "System role not supported" in str(e):
            use_system_role = False
            logger.info("Model does not support system role — folding into user message")
        else:
            raise

    # Build all prompts as chat messages
    all_messages = []
    for question in questions:
        messages = generate_baseline_messages(question, system_prompt, use_system_role)
        all_messages.append(messages)

    # vLLM batched chat generation
    outputs = llm.chat(all_messages, sampling_params=sampling_params, chat_template_kwargs=chat_kwargs)

    results = []
    for question, output in zip(questions, outputs):
        model_answer = strip_thinking_tags(output.outputs[0].text.strip())

        is_correct, match_type = check_correctness(
            model_answer,
            question["correct_answer"],
            question["acceptable_variants"],
        )

        result = {
            "question_id": question["question_id"],
            "domain": question["domain"],
            "question_text": question["question_text"],
            "correct_answer": question["correct_answer"],
            "model_answer": model_answer,
            "is_correct": is_correct,
            "match_type": match_type,
        }

        if not is_correct:
            result["excluded_reason"] = (
                f"Model did not produce correct answer '{question['correct_answer']}'. "
                f"Model response: {model_answer[:200]}"
            )
            logger.warning(
                f"FAILED: {question['question_id']} - "
                f"Expected '{question['correct_answer']}', "
                f"got: {model_answer[:100]}"
            )
        else:
            logger.debug(f"PASSED: {question['question_id']} ({match_type})")

        results.append(result)

    n_correct = sum(1 for r in results if r["is_correct"])
    logger.info(f"Baseline validation: {n_correct}/{len(questions)} correct")

    return results
