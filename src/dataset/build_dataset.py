"""Dataset builder: generates all conversation prompts from questions × conditions.

Crosses 50 questions with 16 conditions to produce 800 conversation templates
(model-agnostic). Each conversation is a 4-turn structure ready for inference.
"""

import json
from pathlib import Path
from typing import List, Dict, Any

from loguru import logger


def load_questions(path: str = None) -> List[Dict]:
    """Load the 50-question bank."""
    if path is None:
        path = Path(__file__).parent / "questions.json"
    with open(path) as f:
        data = json.load(f)
    # Handle nested format with metadata wrapper
    if isinstance(data, dict) and "questions" in data:
        return data["questions"]
    return data


def load_personas(path: str = None) -> Dict:
    """Load persona templates and condition definitions."""
    if path is None:
        path = Path(__file__).parent / "personas.json"
    with open(path) as f:
        return json.load(f)


def get_persona_for_condition(
    condition: Dict,
    question_domain: str,
    personas: Dict,
) -> Dict[str, str]:
    """Resolve a condition + question domain into concrete persona text.

    Returns dict with keys: prefix, reinforcement, control_prefix.
    """
    authority = condition["authority"]
    domain_match = condition.get("domain_match")
    rotation = personas["domain_mismatch_rotation"]

    if authority == "none":
        return {
            "prefix": personas["authority_personas"]["none"]["prefix"],
            "reinforcement": personas["authority_personas"]["none"]["reinforcement"],
            "control_prefix": personas["authority_personas"]["none"]["control_prefix"],
        }

    if authority == "control_high":
        # Length-matched control: use the control_prefix from the high-authority persona
        if domain_match == "matched":
            persona_domain = question_domain
        else:
            persona_domain = rotation[question_domain]
        high_persona = personas["authority_personas"]["high"][persona_domain]
        return {
            "prefix": high_persona["control_prefix"],
            "reinforcement": personas["authority_personas"]["none"]["reinforcement"],
            "control_prefix": high_persona["control_prefix"],
        }

    # Standard authority levels: high, medium, low
    if domain_match == "matched":
        persona_domain = question_domain
    elif domain_match == "mismatched":
        persona_domain = rotation[question_domain]
    else:
        persona_domain = question_domain

    persona = personas["authority_personas"][authority][persona_domain]
    return {
        "prefix": persona["prefix"],
        "reinforcement": persona["reinforcement"],
        "control_prefix": persona["control_prefix"],
    }


def get_confidence_text(
    condition: Dict,
    wrong_answer: str,
    personas: Dict,
) -> Dict[str, str]:
    """Resolve confidence level into concrete text with wrong answer inserted."""
    confidence = condition.get("confidence")
    if confidence is None:
        confidence = "low"  # Default for edge cases

    template = personas["confidence_templates"][confidence]
    return {
        "prefix": template["prefix"].replace("{wrong_answer}", wrong_answer),
        "suffix": template["suffix"],
    }


def build_single_conversation(
    question: Dict,
    condition: Dict,
    personas: Dict,
    system_prompt: str,
) -> Dict[str, Any]:
    """Build one complete 4-turn conversation template.

    Returns a dict with all metadata + the message template.
    Turn 2 (assistant) is left as a placeholder — it will be filled
    with the pre-validated baseline answer during behavioral eval.
    """
    persona = get_persona_for_condition(condition, question["domain"], personas)
    confidence = get_confidence_text(condition, question["wrong_answer"], personas)

    # Build Turn 1: persona prefix + question
    turn1_parts = []
    if persona["prefix"]:
        turn1_parts.append(persona["prefix"])
    turn1_parts.append(question["question_text"])
    turn1_user = " ".join(turn1_parts)

    # Build Turn 3: confidence prefix + pushback with persona reinforcement
    turn3_parts = [
        confidence["prefix"],
        f"Actually, {persona['reinforcement']} {question['wrong_answer']}.",
        confidence["suffix"],
    ]
    turn3_user = " ".join(turn3_parts)

    # Determine persona domain for metadata
    rotation = personas["domain_mismatch_rotation"]
    if condition.get("domain_match") == "mismatched":
        persona_domain = rotation[question["domain"]]
    else:
        persona_domain = question["domain"]

    # Conversation ID
    authority = condition["authority"]
    domain_match = condition.get("domain_match", "none")
    conf = condition.get("confidence", "none")
    conversation_id = f"{question['question_id']}__{authority}_{domain_match}_{conf}"

    return {
        "conversation_id": conversation_id,
        "question_id": question["question_id"],
        "domain": question["domain"],
        "persona_domain": persona_domain,
        "condition": condition,
        "correct_answer": question["correct_answer"],
        "acceptable_variants": question["acceptable_variants"],
        "wrong_answer": question["wrong_answer"],
        "system_prompt": system_prompt,
        "turn1_user": turn1_user,
        "turn2_assistant": None,  # Filled during behavioral eval with baseline answer
        "turn3_user": turn3_user,
        "turn4_assistant": None,  # Generated during behavioral eval
    }


def generate_all_conversations(
    questions: List[Dict],
    personas: Dict,
    system_prompt: str,
) -> List[Dict]:
    """Generate all conversation templates by crossing questions × conditions.

    Args:
        questions: The 50-question bank.
        personas: Persona templates with condition definitions.
        system_prompt: The system prompt for all conversations.

    Returns:
        List of conversation dicts (800 total: 50 questions × 16 conditions).
    """
    conditions = personas["conditions"]
    conversations = []

    for question in questions:
        for condition in conditions:
            conv = build_single_conversation(
                question=question,
                condition=condition,
                personas=personas,
                system_prompt=system_prompt,
            )
            conversations.append(conv)

    logger.info(
        f"Generated {len(conversations)} conversations "
        f"({len(questions)} questions × {len(conditions)} conditions)"
    )
    return conversations
