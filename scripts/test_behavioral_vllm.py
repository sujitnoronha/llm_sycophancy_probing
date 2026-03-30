"""Quick test: run behavioral eval on 3 conversations with one model to verify format."""

import json
import multiprocessing

multiprocessing.set_start_method("spawn", force=True)

from src.core.config import load_config
from src.behavioral.eval import load_baseline, run_behavioral_eval_vllm
from vllm import LLM

cfg = load_config("configs/base.yaml")

# Load conversations and take just 3
with open(f"{cfg.output_dir}/dataset/conversations.json") as f:
    conversations = json.load(f)[:3]

# Use first model
model_cfg = cfg.models[0]
print(f"=== Testing behavioral eval with {model_cfg.name} ===")
print(f"Conversations: {[c['conversation_id'] for c in conversations]}")

# Load baseline
baseline = load_baseline(f"{cfg.output_dir}/baseline", model_cfg.name)

# Load model
print("\nLoading model via vLLM...")
llm = LLM(model=model_cfg.hf_name, dtype=cfg.dtype, seed=cfg.seed, max_model_len=4096)

# Run eval
print("Running behavioral eval...")
results = run_behavioral_eval_vllm(
    llm=llm,
    conversations=conversations,
    baseline=baseline,
    model_name=model_cfg.name,
    system_prompt=cfg.experiment.system_prompt,
    max_new_tokens=cfg.experiment.max_new_tokens_turn4,
)

print(f"\n=== RESULTS ({len(results)} conversations) ===")
for r in results:
    print(f"\n--- {r['conversation_id']} ---")
    print(f"  Condition: auth={r['condition']['authority']}, "
          f"conf={r['condition'].get('confidence','none')}, "
          f"match={r['condition'].get('domain_match','none')}")
    print(f"  Turn 1 (user): {r['turn1_user'][:100]}...")
    print(f"  Turn 2 (asst): {r['turn2_assistant'][:100]}...")
    print(f"  Turn 3 (user): {r['turn3_user'][:100]}...")
    print(f"  Turn 4 (asst): {r['turn4_assistant'][:200]}...")
    print(f"  Tokens generated: {r['generation_metadata']['n_tokens_generated']}")

print("\n=== Required keys check ===")
required = ["conversation_id", "question_id", "domain", "condition", "correct_answer",
            "wrong_answer", "turn1_user", "turn2_assistant", "turn3_user",
            "turn4_assistant", "model_name", "generation_metadata"]
for key in required:
    present = all(key in r for r in results)
    print(f"  {key}: {'OK' if present else 'MISSING'}")
