# Authority In, Sycophancy Out

Probing the mechanistic link between persona authority and model capitulation across architectures.

**Target venue:** COLM 2026

## Setup

```bash
# 1. Install dependencies
pip install -e .

# 2. Configure environment
cp .env.example .env
# Edit .env with your HF_TOKEN and OPENAI_API_KEY
```

**Requirements:** Python 3.10+, NVIDIA A100 GPU (40/80GB), ~25 GPU-hours total.

## Pipeline

The experiment runs as a 10-step pipeline. Each step reads from the previous step's outputs and writes to `outputs/<step_name>/`. Every step supports `--model <name>` for single-model runs and checkpointing for crash recovery.

| Order | Module | Command | What it does | Outputs |
|-------|--------|---------|--------------|---------|
| 1 | `dataset` | `make dataset` | Generate 800 conversation templates (50 Q × 16 conditions) | `outputs/dataset/conversations.json` |
| 2 | `baseline` | `make baseline` | Verify all 5 models answer all 50 questions correctly | `outputs/baseline/{model}_baseline.json` |
| 3 | `behavioral` | `make behavioral` | Run 4,000 conversations, score Turn 4 with GPT-4o-mini judge | `outputs/behavioral/{model}_conversations.jsonl`, `{model}_scores.csv` |
| 4 | `analysis` | `make analysis` | Factorial ANOVA, logistic regression, effect sizes | `outputs/analysis/anova_results.json`, `effect_sizes.json` |
| 5 | `activations` | `make activations` | Cache residual stream at every layer (200 prompts/model) | `outputs/activations/{model}_activations.pt` |
| 6 | `probing` | `make probing` | Train 3 linear probes × all layers × 5 models | `outputs/probing/{model}/weight_vectors.pt` |
| 7 | `directions` | `make directions` | Cosine similarity matrices, layer trajectories, subspace decomposition | `outputs/directions/cosine_similarity_matrices.json` |
| 8 | `steering` | `make steering` | Subtract probed directions at inference, 3 dirs × 6 alphas | `outputs/steering/{model}/steering_summary.json` |
| 9 | `mmlu` | `make mmlu` | MMLU 5-shot capability check on steered models | `outputs/mmlu/{model}/mmlu_results.json` |
| 10 | `figures` | `make figures` | Generate all 6 publication-quality figures | `outputs/figures/figure{1-6}.pdf` |

Steps 4 and 5 can run in parallel after step 3.

### Run the full pipeline

```bash
make all
```

### Run a single model through the pipeline

```bash
make single-model MODEL=llama3_8b
```

### Run individual steps

```bash
python -m src.dataset.run
python -m src.baseline.run --model llama3_8b
python -m src.behavioral.run --model llama3_8b --score-only
python -m src.probing.run --model llama3_8b --permtest
```

## Models Under Test

| Model | Layers | d_model | Alignment | Origin |
|-------|--------|---------|-----------|--------|
| Llama-3.1-8B-Instruct | 32 | 4096 | RLHF (PPO) | Meta |
| Qwen-2.5-7B-Instruct | 28 | 3584 | RLHF + DPO | Alibaba |
| Mistral-7B-Instruct-v0.3 | 32 | 4096 | DPO | Mistral |
| Gemma-2-9B-it | 42 | 3584 | RLHF | Google |
| DeepSeek-R1-Distill-Llama-8B | 32 | 4096 | Reasoning distillation | DeepSeek |

## Project Structure

```
src/
├── core/           Shared config, model loading, utilities
├── dataset/        Question bank, personas, conversation generator
├── baseline/       Baseline correctness validation
├── behavioral/     Behavioral eval + GPT-4o-mini scoring + human annotation
├── analysis/       ANOVA, logistic regression, effect sizes
├── activations/    Residual stream caching via TransformerLens
├── probing/        Linear probe training + direction extraction
├── directions/     Cosine similarity, subspace decomposition, cross-model comparison
├── steering/       Activation steering experiments
├── mmlu/           MMLU capability preservation check
└── figures/        Publication-quality figure generation
```

## Human Annotation

After behavioral eval completes, generate annotation sheets for inter-rater reliability:

```bash
make annotations
```

This creates CSV files for 2 annotators (300 samples, 100 overlapping) in `outputs/behavioral/`.

## License

MIT
