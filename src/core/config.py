"""Configuration system for the sycophancy probing pipeline.

All config is defined via Pydantic models, loaded from YAML.
Every module imports Config from here to ensure consistency.
"""

from pathlib import Path
from typing import List, Optional

import yaml
from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    name: str
    hf_name: str
    n_layers: int
    d_model: int
    use_transformerlens: bool = True
    tl_name: Optional[str] = None  # TransformerLens architecture hint (if hf_name isn't directly supported)


class ExperimentConfig(BaseModel):
    system_prompt: str
    max_new_tokens_turn2: int = 256
    max_new_tokens_turn4: int = 512
    temperature: float = 0.0
    top_p: float = 1.0
    n_questions: int = 50
    domains: List[str]
    authority_levels: List[str]
    confidence_levels: List[str]
    domain_match_levels: List[str]
    n_conditions: int = 16


class ProbingCondition(BaseModel):
    authority: str
    confidence: Optional[str] = None
    domain_match: Optional[str] = None


class ProbingConfig(BaseModel):
    n_prompts_per_model: int = 200
    train_fraction: float = 0.8
    probe_regularization: float = 1.0
    max_iter: int = 1000
    solver: str = "lbfgs"
    activation_hook: str = "resid_post"
    token_position: str = "last_user"
    key_conditions: List[ProbingCondition]


class SteeringConfig(BaseModel):
    alpha_values: List[float]
    directions: List[str]
    layer_selection: str = "middle"


class ScoringConfig(BaseModel):
    judge_model: str = "gpt-4o-mini"
    judge_batch_size: int = 20
    judge_max_retries: int = 3
    human_annotation_sample: int = 300
    human_annotation_overlap: int = 100


class MMLUConfig(BaseModel):
    n_questions: int = 100
    n_shot: int = 5


class Config(BaseModel):
    seed: int = 42
    device: str = "cuda"
    dtype: str = "bfloat16"
    output_dir: str = "outputs"
    log_level: str = "INFO"
    models: List[ModelConfig]
    experiment: ExperimentConfig
    probing: ProbingConfig
    steering: SteeringConfig
    scoring: ScoringConfig
    mmlu: MMLUConfig = Field(default_factory=lambda: MMLUConfig())


def load_config(config_path: str = "configs/base.yaml") -> Config:
    """Load configuration from YAML file and validate with Pydantic."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path) as f:
        raw = yaml.safe_load(f)

    return Config(**raw)


def get_model_config(config: Config, model_name: str) -> ModelConfig:
    """Get a specific model's config by name."""
    for m in config.models:
        if m.name == model_name:
            return m
    available = [m.name for m in config.models]
    raise ValueError(f"Model '{model_name}' not found. Available: {available}")
