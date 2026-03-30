"""Test HuggingFace-based steering vs TransformerLens-based steering.

Loads a small model (mistral_7b) via both approaches, generates steered
responses for a few conversations, and compares outputs to verify equivalence.

The HF approach uses register_forward_hook on the model layer's output,
which preserves KV-cache and should be 3-5x faster.
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.core.config import load_config, get_model_config
from src.core.utils import setup_logging, log_gpu_memory


def run_hf_steered_generation(
    model,
    tokenizer,
    messages,
    direction: np.ndarray,
    alpha: float,
    target_layer: int,
    max_new_tokens: int = 512,
) -> str:
    """Generate text with activation steering using HuggingFace + register_forward_hook.

    This approach preserves KV-cache, making it much faster than TransformerLens.
    The hook modifies the residual stream output of the target layer by subtracting
    alpha * direction from the hidden states.
    """
    # Pre-convert direction to tensor
    direction_tensor = torch.from_numpy(
        np.asarray(direction, dtype=np.float64)
    ).clone()

    _cached = {}

    def steering_hook(module, input, output):
        """Hook on a transformer layer's output.

        HuggingFace decoder layers return either:
          - A tuple: (hidden_states, ...)
          - A single tensor (hidden_states)
        We modify hidden_states by subtracting alpha * direction.
        """
        # Extract hidden_states from whatever format
        if isinstance(output, tuple):
            hidden_states = output[0]
        elif isinstance(output, torch.Tensor):
            hidden_states = output
        else:
            # Dataclass-like (e.g. BaseModelOutputWithPast)
            hidden_states = output[0]

        cache_key = (hidden_states.device, hidden_states.dtype)
        if cache_key not in _cached:
            _cached[cache_key] = direction_tensor.to(
                device=hidden_states.device, dtype=hidden_states.dtype
            )
        d = _cached[cache_key]

        # Subtract alpha * direction from all positions
        modified = hidden_states - alpha * d

        # Return in the same format
        if isinstance(output, tuple):
            return (modified,) + tuple(output[i] for i in range(1, len(output)))
        elif isinstance(output, torch.Tensor):
            return modified
        else:
            # For dataclass outputs, replace hidden_states
            output[0] = modified
            return output

    # Get the target layer module
    # HuggingFace models: model.model.layers[i] for most architectures
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layer_module = model.model.layers[target_layer]
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        layer_module = model.transformer.h[target_layer]
    else:
        raise ValueError(f"Cannot find layer modules in model architecture: {type(model)}")

    # Register the hook
    handle = layer_module.register_forward_hook(steering_hook)

    try:
        # Tokenize
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        input_ids = tokenizer.encode(input_text, return_tensors="pt")
        input_ids = input_ids.to(model.device)

        # Generate with KV-cache (default HF behavior)
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
            )

        generated_ids = output_ids[0, input_ids.shape[1]:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        return generated_text

    finally:
        handle.remove()


def main():
    cfg = load_config("configs/base.yaml")
    setup_logging("INFO")

    # Use llama3.1-8B (TL natively supports it, so same weights in both paths)
    model_name = "llama31_8b"
    model_cfg = get_model_config(cfg, model_name)

    # Load probe direction
    probe_dir = Path(cfg.output_dir) / "probing" / model_name
    weight_vectors = torch.load(str(probe_dir / "weight_vectors.pt"), weights_only=False)
    w_all = weight_vectors["w_authority"].numpy()
    n_layers = w_all.shape[0]
    peak_layer = n_layers // 2
    w = w_all[peak_layer].astype(np.float64)
    w = w / np.linalg.norm(w)
    logger.info(f"Direction: layer {peak_layer}, d_model={w.shape[0]}")

    # Load a few test conversations
    conv_path = Path(cfg.output_dir) / "behavioral" / f"{model_name}_conversations.jsonl"
    conversations = []
    with open(conv_path) as f:
        for i, line in enumerate(f):
            if i >= 5:  # Just 5 conversations for the test
                break
            conversations.append(json.loads(line))

    alpha = 1.5

    # ---- HuggingFace approach ----
    logger.info("Loading model via HuggingFace...")
    dtype = getattr(torch, cfg.dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_cfg.hf_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_cfg.hf_name, torch_dtype=dtype, device_map="auto"
    )
    log_gpu_memory("HF model loaded: ")

    hf_results = []
    hf_times = []
    for i, conv in enumerate(conversations):
        messages = [
            {"role": "system", "content": conv.get("system_prompt", cfg.experiment.system_prompt)},
            {"role": "user", "content": conv["turn1_user"]},
            {"role": "assistant", "content": conv["turn2_assistant"]},
            {"role": "user", "content": conv["turn3_user"]},
        ]

        t0 = time.time()
        text = run_hf_steered_generation(
            hf_model, tokenizer, messages, w, alpha, peak_layer,
            max_new_tokens=512,
        )
        elapsed = time.time() - t0
        hf_times.append(elapsed)
        hf_results.append(text)
        logger.info(f"HF conv {i}: {elapsed:.1f}s, {len(text)} chars")
        logger.info(f"  First 150 chars: {text[:150]}")

    del hf_model
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    log_gpu_memory("After HF cleanup: ")

    # ---- TransformerLens approach ----
    logger.info("Loading model via TransformerLens...")
    from transformer_lens import HookedTransformer
    from transformers import AutoModelForCausalLM as HFModel, AutoTokenizer as HFTok

    tl_name = getattr(model_cfg, "tl_name", None) or model_cfg.hf_name
    # If tl_name differs from hf_name, load HF weights separately
    hf_model_for_tl = None
    hf_tok_for_tl = None
    if tl_name != model_cfg.hf_name:
        hf_model_for_tl = HFModel.from_pretrained(model_cfg.hf_name, torch_dtype=dtype)
        hf_tok_for_tl = HFTok.from_pretrained(model_cfg.hf_name)

    tl_model = HookedTransformer.from_pretrained(
        tl_name, dtype=dtype, device="cuda",
        hf_model=hf_model_for_tl, tokenizer=hf_tok_for_tl,
    )
    del hf_model_for_tl  # Free the HF copy
    tl_tokenizer = tl_model.tokenizer
    if tl_tokenizer.pad_token is None:
        tl_tokenizer.pad_token = tl_tokenizer.eos_token
    log_gpu_memory("TL model loaded: ")

    from src.steering.steer import run_steered_generation
    tl_results = []
    tl_times = []
    for i, conv in enumerate(conversations):
        messages = [
            {"role": "system", "content": conv.get("system_prompt", cfg.experiment.system_prompt)},
            {"role": "user", "content": conv["turn1_user"]},
            {"role": "assistant", "content": conv["turn2_assistant"]},
            {"role": "user", "content": conv["turn3_user"]},
        ]

        t0 = time.time()
        text = run_steered_generation(
            tl_model, tl_tokenizer, messages, w, alpha, peak_layer,
            max_new_tokens=512, config=cfg,
        )
        elapsed = time.time() - t0
        tl_times.append(elapsed)
        tl_results.append(text)
        logger.info(f"TL conv {i}: {elapsed:.1f}s, {len(text)} chars")
        logger.info(f"  First 150 chars: {text[:150]}")

    del tl_model
    gc.collect()
    torch.cuda.empty_cache()

    # ---- Compare ----
    print("\n" + "=" * 80)
    print("COMPARISON: HuggingFace vs TransformerLens steering")
    print("=" * 80)
    print(f"\nHF avg time:  {np.mean(hf_times):.1f}s per conversation")
    print(f"TL avg time:  {np.mean(tl_times):.1f}s per conversation")
    print(f"Speedup:      {np.mean(tl_times) / np.mean(hf_times):.1f}x")

    for i in range(len(conversations)):
        match = hf_results[i] == tl_results[i]
        print(f"\nConv {i}: {'MATCH' if match else 'MISMATCH'}")
        if not match:
            # Show first divergence point
            hf_words = hf_results[i].split()
            tl_words = tl_results[i].split()
            for j in range(min(len(hf_words), len(tl_words))):
                if hf_words[j] != tl_words[j]:
                    print(f"  Diverges at word {j}: HF='{hf_words[j]}' vs TL='{tl_words[j]}'")
                    break
            print(f"  HF ({len(hf_results[i])} chars): {hf_results[i][:200]}...")
            print(f"  TL ({len(tl_results[i])} chars): {tl_results[i][:200]}...")


if __name__ == "__main__":
    main()
