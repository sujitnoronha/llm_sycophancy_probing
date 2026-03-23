# Execution Steps — Remaining Pipeline

All code changes are complete. Run these steps sequentially (unless noted otherwise).

## GPU Steps (no API cost)

### Step 1: Activation Collection
```bash
python -m src.activations.run
```
Collects residual stream activations (50 questions x 4 conditions = 200 prompts per model, 5 models).
**Output:** `outputs/activations/{model}.pt` — shape `[200, n_layers, d_model]`

### Step 2: Probe Training
```bash
python -m src.probing.run --permtest
```
Trains authority/sycophancy/confidence probes + confound probes (prompt_length, authority_keyword) at every layer. Permutation tests validate above-chance accuracy.
**Output:** `outputs/probing/{model}/` — weight vectors, accuracy curves, confound probe results

### Step 3: Direction Analysis
```bash
python -m src.directions.run
```
Cosine similarity between probe directions + confound cosine analysis (authority vs prompt_length, authority vs authority_keyword).
**Output:** `outputs/directions/{model}/` — cosine matrices, confound analysis JSON

### Step 4: Steering Generation
```bash
python -m src.steering.run
```
Generates steered Turn 4 responses using authority direction at alphas 0.5, 1.5, 3.0.
**Output:** `outputs/steering/{model}/steered_authority_alpha{X}.jsonl`

### Step 5: Random Baseline Generation
```bash
python -m src.steering.run --random-baseline
```
Generates random-direction steered responses at alpha=1.5 using authority's peak layer.
**Output:** `outputs/steering/{model}/steered_random_alpha1.5.jsonl`

### Step 6: MMLU Capability Check
```bash
python -m src.mmlu.run
```
100-question 5-shot MMLU, unsteered + steered at 3 alpha values. Can run in parallel with Steps 4-5.
**Output:** `outputs/mmlu/{model}_mmlu.json`

## API Steps (costs money)

### Step 7: Score Steered Conversations (~$48-55)
```bash
python scripts/score_steered.py
```
Scores all steered conversations with GPT-5 judge.
**Output:** `outputs/steering/{model}/steered_authority_alpha{X}_scores.csv`

### Step 8: Score Random Baseline (~$1)
```bash
python scripts/score_steered.py --judge-model gpt-4o-mini
```
Scores random baseline conversations with gpt-4o-mini (cheap, sufficient to show null effect).
**Output:** `outputs/steering/{model}/steered_random_alpha1.5_scores.csv`

### Step 9: Validate Mini Judge (~$0.25)
```bash
python scripts/validate_mini_judge.py
```
Scores 50 random-baseline conversations with BOTH gpt-4o-mini and GPT-5, computes Cohen's kappa.
**Output:** `outputs/steering/judge_validation.json`

## Final Step (no cost)

### Step 10: Generate Figures
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
Step 1 (activations)
  → Step 2 (probing)
    → Step 3 (directions)
      → Step 4 (steering) + Step 5 (random baseline)
        → Step 7 (score steered) + Step 8 (score random)
          → Step 10 (figures)

Step 6 (MMLU) — can run after Step 3, parallel with Steps 4-5
Step 9 (validate judge) — can run after Step 8
Human annotation — independent parallel track
```

## Budget Summary

| Item | Cost |
|------|------|
| Already spent (Step 3 behavioral scoring) | $18.31 |
| Step 7: Score steered (GPT-5) | ~$48-55 |
| Step 8: Score random baseline (gpt-4o-mini) | ~$1 |
| Step 9: Validate mini judge | ~$0.25 |
| **Total** | **~$68-75** |
