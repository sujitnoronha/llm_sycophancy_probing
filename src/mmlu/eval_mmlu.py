"""MMLU capability evaluation module.

Evaluates whether activation steering degrades general model capabilities
by running 5-shot MMLU on both steered and unsteered models.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from loguru import logger
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_mmlu_questions(path: Optional[str] = None) -> List[Dict]:
    """Load MMLU questions from the local subset JSON.

    Each question dict has:
        - question (str): the question text
        - choices (dict): mapping A/B/C/D -> answer text
        - correct_answer (str): one of A, B, C, D
        - subject (str): MMLU subject name

    Args:
        path: Path to the MMLU subset JSON. Defaults to
              ``src/mmlu/mmlu_subset.json`` relative to project root.

    Returns:
        List of question dicts.
    """
    if path is None:
        path = Path(__file__).parent / "mmlu_subset.json"
    else:
        path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"MMLU subset not found at {path}")

    with open(path) as f:
        questions = json.load(f)

    logger.info(f"Loaded {len(questions)} MMLU questions from {path}")
    return questions


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------

def _format_single_example(question: Dict, include_answer: bool = True) -> str:
    """Format one MMLU question as a text block.

    Example output::

        Question: What is the capital of France?
        A. Paris
        B. London
        C. Berlin
        D. Madrid
        Answer: A
    """
    lines = [f"Question: {question['question']}"]
    for letter in ["A", "B", "C", "D"]:
        lines.append(f"{letter}. {question['choices'][letter]}")
    if include_answer:
        lines.append(f"Answer: {question['correct_answer']}")
    else:
        lines.append("Answer:")
    return "\n".join(lines)


def format_few_shot_prompt(
    question: Dict,
    few_shot_examples: List[Dict],
    n_shot: int = 5,
) -> str:
    """Format a few-shot MMLU prompt.

    The prompt consists of ``n_shot`` fully answered examples followed by
    the target question without an answer (ends with ``Answer:``).

    Args:
        question: The target question to evaluate.
        few_shot_examples: Pool of examples to draw from. The first
            ``n_shot`` that are **not** the target question are used.
        n_shot: Number of few-shot examples (default 5).

    Returns:
        The complete prompt string.
    """
    # Select n_shot examples, skipping the target question itself
    selected = []
    for ex in few_shot_examples:
        if ex["question"] == question["question"]:
            continue
        selected.append(ex)
        if len(selected) >= n_shot:
            break

    parts = [
        "The following are multiple choice questions (with answers).\n"
    ]
    for ex in selected:
        parts.append(_format_single_example(ex, include_answer=True))
        parts.append("")  # blank line between examples

    # Target question without answer
    parts.append(_format_single_example(question, include_answer=False))
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

def _extract_answer_letter(text: str) -> Optional[str]:
    """Extract a single letter choice (A/B/C/D) from generated text.

    Tries several heuristics:
    1. Exact single-letter match at the start.
    2. Pattern like ``A.``, ``(A)``, ``A:``.
    3. First occurrence of A/B/C/D in the text.

    Returns:
        The extracted letter or None if nothing found.
    """
    text = text.strip()
    if not text:
        return None

    # 1. First character is a valid letter
    first_char = text[0].upper()
    if first_char in "ABCD":
        return first_char

    # 2. Regex patterns
    patterns = [
        r"^[(\[]?([A-D])[)\].]",      # (A) or A. or [A]
        r"\b([A-D])\b",                # standalone letter
        r"answer\s*(?:is|:)?\s*([A-D])",  # "answer is A" / "answer: A"
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return m.group(1).upper()

    return None


# ---------------------------------------------------------------------------
# Unsteered evaluation
# ---------------------------------------------------------------------------

def evaluate_mmlu(
    model,
    tokenizer,
    questions: List[Dict],
    n_shot: int = 5,
    max_new_tokens: int = 5,
) -> Dict:
    """Run MMLU evaluation without steering.

    For each question:
    1. Build a 5-shot prompt.
    2. Generate the model's continuation (very short — just the letter).
    3. Extract the predicted letter and compare to the correct answer.

    Args:
        model: A HuggingFace ``AutoModelForCausalLM`` (or compatible).
        tokenizer: The model's tokenizer.
        questions: List of MMLU question dicts.
        n_shot: Number of few-shot examples per prompt.
        max_new_tokens: Max tokens to generate (default 5, enough for a letter).

    Returns:
        Dict with:
            - accuracy (float): fraction correct
            - n_correct (int)
            - n_total (int)
            - per_question (list): per-question results
    """
    per_question = []
    n_correct = 0

    for q in tqdm(questions, desc="MMLU (unsteered)"):
        prompt = format_few_shot_prompt(q, questions, n_shot=n_shot)

        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        input_ids = input_ids.to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        generated_ids = output_ids[0, input_ids.shape[1]:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        predicted = _extract_answer_letter(generated_text)
        correct = q["correct_answer"]
        is_correct = predicted == correct

        if is_correct:
            n_correct += 1

        per_question.append({
            "question": q["question"],
            "subject": q.get("subject", ""),
            "correct_answer": correct,
            "predicted": predicted,
            "is_correct": is_correct,
            "generated_text": generated_text.strip(),
        })

    accuracy = n_correct / len(questions) if questions else 0.0
    logger.info(f"MMLU (unsteered): {n_correct}/{len(questions)} = {accuracy:.3f}")

    return {
        "accuracy": accuracy,
        "n_correct": n_correct,
        "n_total": len(questions),
        "per_question": per_question,
    }


# ---------------------------------------------------------------------------
# Steered evaluation
# ---------------------------------------------------------------------------

def _make_steering_hook(direction: torch.Tensor, alpha: float, target_layer: int):
    """Create a forward hook that subtracts ``alpha * direction`` from the residual stream.

    Matches the sign convention in ``src.steering.steer.create_steering_hook``:
    positive alpha removes the direction (e.g., subtracting the authority direction
    reduces authority-driven behavior).

    The hook is designed for HuggingFace ``AutoModelForCausalLM`` models where
    each transformer block has an output of shape ``(batch, seq, d_model)``.

    Args:
        direction: Unit direction vector [d_model].
        alpha: Steering strength scalar.
        target_layer: The decoder layer index to attach the hook to.

    Returns:
        A hook function suitable for ``register_forward_hook``.
    """
    steering_vec = alpha * direction  # [d_model]

    def hook_fn(module, input, output):
        # output is typically a tuple; first element is the hidden state
        if isinstance(output, tuple):
            hidden = output[0]
            hidden = hidden - steering_vec.to(hidden.device, hidden.dtype)
            return (hidden,) + output[1:]
        else:
            return output - steering_vec.to(output.device, output.dtype)

    return hook_fn


def evaluate_mmlu_steered(
    model,
    tokenizer,
    questions: List[Dict],
    direction: torch.Tensor,
    alpha: float,
    target_layer: int,
    n_shot: int = 5,
    max_new_tokens: int = 5,
) -> Dict:
    """Run MMLU evaluation with a steering hook active during generation.

    The steering vector ``alpha * direction`` is added to the residual stream
    output of the specified decoder layer on every forward pass.

    Args:
        model: A HuggingFace ``AutoModelForCausalLM`` (or compatible).
        tokenizer: The model's tokenizer.
        questions: List of MMLU question dicts.
        direction: Unit direction vector [d_model] (e.g., authority direction).
        alpha: Steering strength.
        target_layer: Decoder layer index to steer.
        n_shot: Number of few-shot examples.
        max_new_tokens: Max tokens to generate.

    Returns:
        Same format as ``evaluate_mmlu``, plus ``steering_config`` metadata.
    """
    # Register steering hook
    hook_fn = _make_steering_hook(direction, alpha, target_layer)

    # Find the target layer module — try common HF model architectures
    hook_handle = None
    layer_module = None

    # Try attribute paths used by common architectures
    for attr_path in [
        f"model.layers.{target_layer}",       # Llama, Mistral, Qwen, DeepSeek
        f"model.language_model.layers.{target_layer}",  # Gemma 3
        f"model.decoder.layers.{target_layer}",  # OPT-style
        f"transformer.h.{target_layer}",       # GPT2-style
    ]:
        try:
            module = model
            for attr in attr_path.split("."):
                if attr.isdigit():
                    module = module[int(attr)]
                else:
                    module = getattr(module, attr)
            layer_module = module
            break
        except (AttributeError, IndexError, TypeError):
            continue

    if layer_module is None:
        raise RuntimeError(
            f"Could not find decoder layer {target_layer} in model. "
            f"Supported architectures: Llama, Mistral, Qwen, GPT2-style."
        )

    hook_handle = layer_module.register_forward_hook(hook_fn)
    logger.info(
        f"Steering hook registered on layer {target_layer} "
        f"(alpha={alpha:.2f}, ||dir||={direction.norm():.4f})"
    )

    try:
        per_question = []
        n_correct = 0

        for q in tqdm(questions, desc=f"MMLU (steered a={alpha})"):
            prompt = format_few_shot_prompt(q, questions, n_shot=n_shot)

            input_ids = tokenizer.encode(prompt, return_tensors="pt")
            input_ids = input_ids.to(model.device)

            with torch.no_grad():
                output_ids = model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=0.0,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )

            generated_ids = output_ids[0, input_ids.shape[1]:]
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            predicted = _extract_answer_letter(generated_text)
            correct = q["correct_answer"]
            is_correct = predicted == correct

            if is_correct:
                n_correct += 1

            per_question.append({
                "question": q["question"],
                "subject": q.get("subject", ""),
                "correct_answer": correct,
                "predicted": predicted,
                "is_correct": is_correct,
                "generated_text": generated_text.strip(),
            })

        accuracy = n_correct / len(questions) if questions else 0.0
        logger.info(
            f"MMLU (steered, alpha={alpha}): "
            f"{n_correct}/{len(questions)} = {accuracy:.3f}"
        )

    finally:
        # Always remove the hook
        if hook_handle is not None:
            hook_handle.remove()
            logger.debug("Steering hook removed")

    return {
        "accuracy": accuracy,
        "n_correct": n_correct,
        "n_total": len(questions),
        "per_question": per_question,
        "steering_config": {
            "alpha": alpha,
            "target_layer": target_layer,
            "direction_norm": float(direction.norm()),
        },
    }
