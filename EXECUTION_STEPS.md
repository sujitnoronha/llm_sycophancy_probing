# Execution Steps — Expanded Dataset Pipeline (v2)

All code changes are complete. The dataset has been expanded from 50 to 200 questions across 10 domains with 3 difficulty tiers. Run these steps sequentially (unless noted otherwise).

## Step 0: Dataset Preparation (Complete)

Dataset already rebuilt: 200 questions x 16 conditions = 3,200 conversations.
Old 50-question results backed up to `outputs/backup_50q_run/`.

## GPU Steps (no API cost)

### Step 1: Baseline Validation — COMPLETE
```bash
python -m src.baseline.run
```
Validates all 5 models answer 200 questions correctly via vLLM batched inference.
**Output:** `outputs/baseline/{model}_baseline.json`

**Current results:**
| Model | Correct/200 |
|-------|-------------|
| llama31_8b | 166 |
| qwen3_8b | 167 |
| gemma3_12b | 165 |
| deepseek_r1_qwen3_8b | 165 |
| mistral_7b_v03 | 160 |

### Step 2: Behavioral Evaluation (Generate Turn 4)
```bash
python -m src.behavioral.run --generate-only
```
Generates Turn 4 responses for ~2,640-2,720 conversations per model (after baseline filtering).
**Output:** `outputs/behavioral/{model}_conversations.jsonl`

### Step 3: Score with GPT-5 Judge (~$40-80)
```bash
python -m src.behavioral.run --score-only
```
Scores Turn 4 on flip (binary), hedge_score (0-5), deference_ratio (0-1).
**Output:** `outputs/behavioral/{model}_scores.csv`

### Step 4: Statistical Analysis
```bash
python -m src.analysis.run
```
Factorial ANOVA, logistic regression, effect sizes, question-level analysis.
**Output:** `outputs/analysis/`

### Step 5: Activation Collection
```bash
python -m src.activations.run
```
Collects residual stream activations (50 questions x 4 key conditions = 200 prompts per model).
**Output:** `outputs/activations/{model}.pt` — shape `[200, n_layers, d_model]`

### Step 6: Probe Training
```bash
python -m src.probing.run --permtest
```
Trains authority/sycophancy/confidence probes + confound probes (prompt_length, authority_keyword) at every layer. Permutation tests validate above-chance accuracy.
**Output:** `outputs/probing/{model}/` — weight vectors, accuracy curves, confound probe results

### Step 7: Direction Analysis
```bash
python -m src.directions.run
```
Cosine similarity between probe directions + confound cosine analysis.
**Output:** `outputs/directions/{model}/`

### Step 8: Steering Generation
```bash
python -m src.steering.run
```
Generates steered Turn 4 responses using authority direction at alphas 0.5, 1.5, 3.0.
**Output:** `outputs/steering/{model}/steered_authority_alpha{X}.jsonl`

### Step 9: Random Baseline Generation
```bash
python -m src.steering.run --random-baseline
```
Random unit vector steered at alpha=1.5 using authority's peak layer.
**Output:** `outputs/steering/{model}/steered_random_alpha1.5.jsonl`

### Step 10: MMLU Capability Check
```bash
python -m src.mmlu.run
```
100-question 5-shot MMLU, unsteered + steered at 3 alpha values. Can run in parallel with Steps 8-9.
**Output:** `outputs/mmlu/{model}_mmlu.json`

## API Steps (costs money)

### Step 11: Score Steered Conversations (~$48-55)
```bash
python scripts/score_steered.py
```
Scores all steered conversations with GPT-5 judge.
**Output:** `outputs/steering/{model}/steered_authority_alpha{X}_scores.csv`

### Step 12: Score Random Baseline (~$1)
```bash
python scripts/score_steered.py --judge-model gpt-4o-mini
```
Scores random baseline conversations with gpt-4o-mini.
**Output:** `outputs/steering/{model}/steered_random_alpha1.5_scores.csv`

### Step 13: Validate Mini Judge (~$0.25)
```bash
python scripts/validate_mini_judge.py
```
Scores 50 random-baseline conversations with BOTH gpt-4o-mini and GPT-5, computes Cohen's kappa.
**Output:** `outputs/steering/judge_validation.json`

## Final Step (no cost)

### Step 14: Generate Figures
```bash
python -m src.figures.run
```
Generates all 8 publication figures as PDF + PNG.
**Output:** `outputs/figures/figure{1-8}_*.{pdf,png}`

## Parallel Track (manual)

### Human Annotation
```bash
python -m src.behavioral.run --generate-annotations
```
Generates 2 CSV files (300 samples, 100 overlap) for manual annotation of `flip` and `hedge_score`.

After manual annotation:
```bash
python -m src.behavioral.run --compute-agreement
```
Computes Cohen's kappa (human-human and human-LLM).
**Output:** `outputs/behavioral/inter_rater_agreement.json`

## Dependency Graph

```
Step 1 (baseline) → Step 2 (behavioral gen) → Step 3 (scoring)
                                                     ↓
                                              Step 4 (analysis)
                                              Step 5 (activations)
                                                     ↓
                                              Step 6 (probing)
                                                     ↓
                                              Step 7 (directions)
                                                     ↓
                                    Step 8 (steering) + Step 9 (random)
                                              + Step 10 (MMLU, parallel)
                                                     ↓
                                    Step 11 (score steered) + Step 12 (score random)
                                                     ↓
                                              Step 13 (validate judge)
                                                     ↓
                                              Step 14 (figures)

Human annotation — independent parallel track
```

## Budget Summary

| Item | Cost |
|------|------|
| Step 3: Behavioral scoring (GPT-5) | ~$40-80 |
| Step 11: Score steered (GPT-5) | ~$48-55 |
| Step 12: Score random baseline (gpt-4o-mini) | ~$1 |
| Step 13: Validate mini judge | ~$0.25 |
| **Total API cost** | **~$90-136** |
