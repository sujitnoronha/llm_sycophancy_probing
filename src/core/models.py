"""Model loading and memory management.

Provides a context manager that loads a model, yields it for use,
and guarantees full cleanup (del + gc + cuda empty_cache) on exit.
"""

import gc
from contextlib import contextmanager
from typing import Tuple

import torch
from loguru import logger

from src.core.config import ModelConfig, Config
from src.core.utils import log_gpu_memory


def _load_hf_model_if_needed(tl_name: str, hf_name: str, dtype):
    """Load HF model weights when using a different arch hint for TransformerLens."""
    if tl_name == hf_name:
        return None  # TransformerLens will load it directly
    from transformers import AutoModelForCausalLM
    logger.info(f"Loading HF weights from {hf_name} (TL arch: {tl_name})")
    return AutoModelForCausalLM.from_pretrained(
        hf_name, torch_dtype=dtype,
    )


def _load_tokenizer_if_needed(tl_name: str, hf_name: str):
    """Load tokenizer from the actual HF model when using an arch hint."""
    if tl_name == hf_name:
        return None  # TransformerLens will load it directly
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(hf_name)


@contextmanager
def load_model_context(
    model_config: ModelConfig,
    config: Config,
    need_hooks: bool = False,
):
    """Context manager for loading and unloading a model.

    Args:
        model_config: Config for the specific model to load.
        config: Global config (for device, dtype, etc.).
        need_hooks: If True, load via TransformerLens HookedTransformer
                    for activation access. If False, load via HuggingFace
                    for faster inference.

    Yields:
        (model, tokenizer) tuple.
    """
    model = None
    tokenizer = None
    dtype = getattr(torch, config.dtype)

    try:
        if need_hooks and model_config.use_transformerlens:
            logger.info(f"Loading {model_config.name} via TransformerLens (hooks enabled)")
            from transformer_lens import HookedTransformer

            # Some fine-tuned models need to be loaded with a base architecture hint
            tl_name = getattr(model_config, "tl_name", None) or model_config.hf_name
            try:
                model = HookedTransformer.from_pretrained(
                    tl_name,
                    dtype=dtype,
                    device=config.device,
                    hf_model=_load_hf_model_if_needed(tl_name, model_config.hf_name, dtype),
                    tokenizer=_load_tokenizer_if_needed(tl_name, model_config.hf_name),
                )
            except Exception:
                # Direct load (works for natively supported models)
                model = HookedTransformer.from_pretrained(
                    model_config.hf_name,
                    dtype=dtype,
                    device=config.device,
                )
            tokenizer = model.tokenizer
        else:
            logger.info(f"Loading {model_config.name} via HuggingFace Transformers")
            from transformers import AutoModelForCausalLM, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(model_config.hf_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_config.hf_name,
                torch_dtype=dtype,
                device_map="auto",
            )

        if tokenizer and tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        log_gpu_memory(f"After loading {model_config.name}: ")
        yield model, tokenizer

    finally:
        logger.info(f"Unloading {model_config.name}")
        del model
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        log_gpu_memory(f"After cleanup {model_config.name}: ")
