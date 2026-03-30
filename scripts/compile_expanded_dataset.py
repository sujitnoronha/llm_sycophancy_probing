#!/usr/bin/env python3
"""
Compile the expanded 200-question Blind Deference benchmark dataset.

Reads:
  - src/dataset/questions.json (original 50 questions)
  - /home/ikulkar1/sycophancy_benchmark_additional_questions.json (50 additional for existing domains)
  - src/dataset/new_domains/*.json (5 new domain files)

Outputs:
  - src/dataset/questions.json (overwritten with 200 questions)
"""

import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "src" / "dataset"
NEW_DOMAINS_DIR = DATASET_DIR / "new_domains"
ADDITIONAL_QUESTIONS_PATH = Path("/home/ikulkar1/sycophancy_benchmark_additional_questions.json")

def load_json(path):
    with open(path) as f:
        return json.load(f)

def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"Saved to {path}")

def main():
    # 1. Load original 50 questions and add difficulty="easy"
    original = load_json(DATASET_DIR / "questions.json")
    original_questions = original["questions"]
    for q in original_questions:
        if "difficulty" not in q:
            q["difficulty"] = "easy"
    print(f"Loaded {len(original_questions)} original questions (marked as easy)")

    # 2. Load 50 additional questions for existing domains
    additional_questions = load_json(ADDITIONAL_QUESTIONS_PATH)
    print(f"Loaded {len(additional_questions)} additional questions for existing domains")

    # 3. Load 5 new domain files
    new_domain_questions = []
    for domain_file in sorted(NEW_DOMAINS_DIR.glob("*.json")):
        domain_qs = load_json(domain_file)
        new_domain_questions.extend(domain_qs)
        print(f"Loaded {len(domain_qs)} questions from {domain_file.name}")

    # 4. Combine all questions
    all_questions = original_questions + additional_questions + new_domain_questions
    print(f"\nTotal before dedup: {len(all_questions)} questions")

    # 5. Check for duplicate question IDs
    seen_ids = {}
    for q in all_questions:
        qid = q["question_id"]
        if qid in seen_ids:
            print(f"  WARNING: Duplicate question_id '{qid}' - keeping last occurrence")
        seen_ids[qid] = q

    all_questions = list(seen_ids.values())
    print(f"Total after dedup: {len(all_questions)} questions")

    # 6. Sort by domain then question_id
    domain_order = [
        "physics", "medicine", "history", "mathematics", "geography",
        "law", "computer_science", "economics", "chemistry", "biology"
    ]
    domain_rank = {d: i for i, d in enumerate(domain_order)}
    all_questions.sort(key=lambda q: (domain_rank.get(q["domain"], 99), q["question_id"]))

    # 7. Validate
    domains = {}
    difficulties = {}
    for q in all_questions:
        d = q["domain"]
        diff = q.get("difficulty", "unknown")
        domains[d] = domains.get(d, 0) + 1
        difficulties[diff] = difficulties.get(diff, 0) + 1

        # Validate required fields
        required = ["question_id", "domain", "question_text", "correct_answer",
                     "acceptable_variants", "wrong_answer", "wrong_answer_plausible_reason",
                     "matching_persona_domain", "difficulty"]
        for field in required:
            if field not in q:
                print(f"  MISSING FIELD '{field}' in {q['question_id']}")

    print(f"\nDomain distribution:")
    for d in domain_order:
        print(f"  {d}: {domains.get(d, 0)} questions")

    print(f"\nDifficulty distribution:")
    for diff, count in sorted(difficulties.items()):
        print(f"  {diff}: {count}")

    # 8. Build final output
    output = {
        "metadata": {
            "version": "2.0",
            "description": "200-question bank for the Blind Deference diagnostic benchmark",
            "domains": domain_order,
            "questions_per_domain": 20,
            "total_questions": len(all_questions),
            "difficulty_levels": ["easy", "medium", "hard"],
            "notes": "Questions span 10 domains with 3 difficulty tiers. Easy questions are basic factual recall; medium requires undergraduate-level knowledge; hard questions target common expert-level misconceptions."
        },
        "questions": all_questions
    }

    # 9. Write output
    save_json(output, DATASET_DIR / "questions.json")

    # Also save a backup of the original
    backup_path = DATASET_DIR / "questions_original_50.json"
    if not backup_path.exists():
        save_json(original, backup_path)
        print(f"Backed up original to {backup_path}")

    print(f"\nDone! {len(all_questions)} questions written to {DATASET_DIR / 'questions.json'}")

if __name__ == "__main__":
    main()
