"""Quick test: run baseline validation on 3 questions with one model to verify output format."""

import json
from src.core.config import load_config
from src.dataset.build_dataset import load_questions
from src.baseline.validate import validate_single_model_vllm, generate_baseline_messages
from vllm import LLM

cfg = load_config("configs/base.yaml")
questions = load_questions()[:3]  # Just 3 questions for testing

# Use the first model (llama3_8b)
model_cfg = cfg.models[0]
print(f"=== Testing with {model_cfg.name} ({model_cfg.hf_name}) ===")
print(f"Questions: {[q['question_id'] for q in questions]}")
print()

# Show what the prompts look like
for q in questions:
    msgs = generate_baseline_messages(q, cfg.experiment.system_prompt)
    print(f"--- {q['question_id']} ---")
    for m in msgs:
        print(f"  [{m['role']}]: {m['content'][:80]}")
    print(f"  Expected answer: {q['correct_answer']}")
    print()

# Load model and run
print("Loading model via vLLM...")
llm = LLM(
    model=model_cfg.hf_name,
    dtype=cfg.dtype,
    seed=cfg.seed,
    max_model_len=4096,
)

print("Running batched inference...")
results = validate_single_model_vllm(
    llm=llm,
    questions=questions,
    system_prompt=cfg.experiment.system_prompt,
    max_new_tokens=cfg.experiment.max_new_tokens_turn2,
)

print("\n=== RESULTS ===")
for r in results:
    status = "PASS" if r["is_correct"] else "FAIL"
    print(f"[{status}] {r['question_id']}")
    print(f"  Expected: {r['correct_answer']}")
    print(f"  Got:      {r['model_answer'][:150]}")
    print(f"  Match:    {r['match_type']}")
    print()

print(f"\nScore: {sum(1 for r in results if r['is_correct'])}/{len(results)}")
print("\n=== Raw JSON output (first result) ===")
print(json.dumps(results[0], indent=2))
