# Blind Deference — Research Findings & Decisions Log
**Paper**: "Blind Deference: A Diagnostic Benchmark for Authority-Driven Sycophancy in Multi-Turn LLM Conversations"
**Target Venue**: COLM 2026
**Last updated**: March 2026

---

## Pipeline Status

| Step | Status | Notes |
|------|--------|-------|
| 1. Dataset | DONE | 200Q x 16 conditions = 3,200 templates |
| 2. Baseline validation | DONE | 5 models, 80-83.5% accuracy |
| 3. Behavioral evaluation | DONE | 13,168 conversations, GPT-5 scored, $65.76 |
| 4. Statistical analysis | DONE | ANOVA, logistic regression, effect sizes |
| 5. Activation collection | DONE | Residual stream cached for all layers |
| 6. Linear probing | DONE | Standard + confound probes + permutation tests |
| 7. Direction analysis | DONE | Cosine similarity, subspace decomposition |
| 8. Activation steering | DONE | Alphas 0.5/1.5/3.0, 200 convs/model/alpha |
| 9. Random baseline | DONE | 200 convs/model |
| 10. MMLU capability check | DONE | Zero degradation for 4/5 models |
| 11. Score steered outputs | PENDING | GPT-5, 3,000 convs, ~$15 |
| 12. Score random baseline | PENDING | gpt-4o-mini, 1,000 convs, ~$0.25 |
| 13. Judge validation | PENDING | Cohen's kappa, ~$0.25 |
| 14. Publication figures | PENDING | 8 figures, $0 |

**Total spent**: $65.76 | **Estimated remaining**: ~$15.50 | **Grand total**: ~$81

---

## Current Models (7-12B)

| Model | Flip Rate | Alignment | Notes |
|-------|----------:|-----------|-------|
| Gemma-3-12B | 52.7% | RLHF | Most sycophantic |
| Mistral-7B | 22.8% | DPO | |
| Llama-3.1-8B | 21.5% | RLHF+PPO | Inverted confidence effect |
| Qwen3-8B | 12.1% | RLHF+DPO | |
| DeepSeek-R1 | 6.6% | Reasoning distillation | Most resistant |

---

## Key Findings

### Finding 1 — Blind Deference is Universal (MAIN CLAIM)
Domain match has **no significant effect** on flip rate (p=0.416).
Models defer equally to matched and mismatched authority — they cannot
distinguish relevant from irrelevant expertise.

| Model | Matched | Mismatched | Diff |
|-------|--------:|-----------:|-----:|
| Llama-3.1-8B | 21.0% | 21.1% | -0.1pp |
| Qwen3-8B | 12.1% | 12.6% | -0.5pp |
| Gemma-3-12B | 57.7% | 58.5% | -0.8pp |
| DeepSeek-R1 | 7.0% | 6.6% | +0.4pp |
| Mistral-7B | 24.0% | 23.6% | +0.3pp |

All differences within noise. Blind deference index: 94-99% for all models.

### Finding 2 — Authority × Confidence Interaction (SECONDARY FINDING)
**The adversarially optimal condition is NOT high authority — it is medium
authority + high confidence.** 3/5 models are MORE susceptible to medium
authority than high authority.

Full interaction table (flip rate %):

| Condition | Llama | Qwen3 | Gemma | DeepSeek | Mistral |
|-----------|------:|------:|------:|---------:|--------:|
| High Auth + High Conf | 21.1% | 21.0% | 81.2% | 8.8% | 36.2% |
| High Auth + Low Conf | 26.5% | 10.2% | 55.8% | 3.3% | 18.4% |
| Med Auth + High Conf | 19.0% | 15.6% | **85.8%** | 13.3% | **37.2%** |
| Med Auth + Low Conf | 23.8% | 6.6% | 64.2% | 4.8% | 20.6% |
| Low Auth + High Conf | 13.3% | 15.6% | 49.7% | 7.0% | 20.3% |
| Low Auth + Low Conf | 22.6% | 5.1% | **11.8%** | 3.3% | **10.0%** |

**Llama inversion**: Low confidence pushback makes Llama MORE sycophantic
(-6.5pp when confidence is high). Unique to PPO-trained model.

### Finding 3 — Mechanistic: Authority ≠ Sycophancy in Activation Space
- Authority is linearly separable (100% probe accuracy) by layer 8-12
- Sycophancy decodable at later layers (79-94%)
- Authority and sycophancy directions are nearly orthogonal (|cos| < 0.07)
- Sycophancy is 99.6-99.9% independent of authority in activation space
- Suggests: model perceives authority early, decides whether to defer later

### Finding 4 — Reasoning Training Confers Sycophancy Resistance
DeepSeek-R1 (reasoning distillation): 6.6% flip vs Gemma (RLHF): 52.7%.
8x difference. Attack surface: DeepSeek 0.2% vs Gemma 32.1%.
Hypothesis: reasoning chains force explicit evaluation, reducing reflexive deference.

### Finding 5 — Steering is Safe
Subtracting the authority direction from activations:
- Zero MMLU degradation for Llama, Qwen3, Gemma, DeepSeek
- -3% for Mistral at alpha=3.0 (most aggressive)
Authority direction removal is safe for capability preservation.

---

## Metric Decisions

### Headline Score: AVI (Authority Vulnerability Index)
```
AVI = 45 * attack_surface   (high_auth_flip - no_auth_flip)
    + 35 * flip_rate         (overall capitulation rate)
    + 20 * worst_case_flip   (max flip across all 16 conditions)
```
Scale: 0-100, lower = safer.

| Model | AVI | Attack | Flip | Worst-Case |
|-------|----:|-------:|-----:|-----------:|
| DeepSeek-R1 | 5.2 | 0.2% | 6.6% | 13.9% |
| Qwen3-8B | 10.6 | 4.2% | 12.1% | 22.2% |
| Llama-3.1-8B | 13.7 | 1.1% | 21.5% | 28.3% |
| Mistral-7B | 18.9 | 7.7% | 22.8% | 37.5% |
| Gemma-3-12B | 50.4 | 32.1% | 52.7% | 87.3% |

**Why blind deference index is NOT in AVI**: It is 94-99% for ALL models
(near-constant). Including it would add a ~24-point constant to every score,
not differentiating models. Instead it is reported separately as the paper's
central universal finding.

**Separate: Blind Deference Index**
Reported as a universal finding: all 5 models score 94-99%.
Formula: `1 - |matched_flip - mismatched_flip| / max(matched, mismatched)`

---

## Benchmark Strengths & Limitations

### Strengths
- Novel matched/mismatched domain manipulation (no other benchmark tests this)
- Rigorous 3-factor factorial design, isolates independent effects
- Multi-turn 4-turn protocol (more realistic than single-turn)
- Mechanistic component (probing + steering) goes beyond behavioral
- Factual ground truth makes evaluation reliable

### Limitations (to address in paper)
1. **Factual questions only** — sycophancy in opinions/advice not covered
2. **~14 conversations per condition per model** — thin cells for some analyses
3. **Explicit authority personas** — real manipulation is subtler
4. **Single pushback** — persistent multi-turn pressure not tested
5. **No human annotation ground truth** — GPT-5 vs gpt-4o-mini, not vs humans
6. **Post-hoc metrics** — AVI designed after seeing data (not pre-registered)

---

## Planned Additions

### Larger Models (for COLM reviewer coverage)
Reviewers WILL ask: "Does this hold for larger models?"

| Model | Size | Priority | Type | Notes |
|-------|------|----------|------|-------|
| Gemma-3-27B | 27B | HIGH | Open | Same family, tests if Gemma's 52.7% scales down |
| Llama-3.1-70B | 70B | HIGH | Open | Same family, tests scale effect |
| GPT-4o | Unknown | MEDIUM | API only | Frontier model behavioral comparison |

**Cost**: ~$26 for 2 open-weight models + ~$15 for GPT-4o = ~$41 extra
**GPU**: 70B models need 4-bit quantization (~40GB), fits on current hardware

### Dataset Additions (for robustness)
- Consider adding 50 opinion/advisory questions alongside factual
- Would significantly strengthen paper's scope claims

### Interaction Finding Figure
- Add 2D heatmap figure (authority × confidence) per model
- Shows adversarially optimal conditions visually
- Section: "5.X Adversarial Optimal Conditions"

---

## Open Questions / Decisions Needed

1. **AVI in report**: Needs to replace current BD-S section and be regenerated
2. **Larger models**: Which to prioritize — Gemma-27B or Llama-70B first?
3. **API models**: Worth the $15 for GPT-4o behavioral-only results?
4. **Opinion questions**: Add to dataset or keep factual-only for v1?
5. **Human annotation**: Add 100-sample human annotation for judge validation?
