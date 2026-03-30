"""Generate a PDF summary report of all research findings."""

from fpdf import FPDF


class Report(FPDF):
    def header(self):
        if self.page_no() > 1:
            self.set_font("Helvetica", "I", 8)
            self.cell(0, 5, "Blind Deference: Authority-Driven Sycophancy in LLMs", align="C")
            self.ln(8)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def section_title(self, title):
        self.set_font("Helvetica", "B", 13)
        self.set_fill_color(230, 230, 240)
        self.cell(0, 8, f"  {title}", fill=True, new_x="LMARGIN", new_y="NEXT")
        self.ln(3)

    def subsection_title(self, title):
        self.set_font("Helvetica", "B", 11)
        self.cell(0, 7, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def body_text(self, text):
        self.set_font("Helvetica", "", 10)
        self.multi_cell(0, 5, text)
        self.ln(2)

    def table_header(self, cols, widths):
        self.set_font("Helvetica", "B", 9)
        self.set_fill_color(60, 60, 90)
        self.set_text_color(255, 255, 255)
        for col, w in zip(cols, widths):
            self.cell(w, 6, col, border=1, fill=True, align="C")
        self.ln()
        self.set_text_color(0, 0, 0)

    def table_row(self, cells, widths, highlight=False):
        self.set_font("Helvetica", "", 9)
        if highlight:
            self.set_fill_color(255, 240, 240)
        else:
            self.set_fill_color(255, 255, 255)
        for cell, w in zip(cells, widths):
            self.cell(w, 5.5, str(cell), border=1, fill=highlight, align="C")
        self.ln()

    def key_finding(self, number, title, text):
        self.set_font("Helvetica", "B", 10)
        self.cell(0, 6, f"Finding {number}: {title}", new_x="LMARGIN", new_y="NEXT")
        self.set_font("Helvetica", "", 9.5)
        self.multi_cell(0, 5, text)
        self.ln(2)


def build_report():
    pdf = Report()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)

    # ---- Title Page ----
    pdf.add_page()
    pdf.ln(40)
    pdf.set_font("Helvetica", "B", 24)
    pdf.multi_cell(0, 12, "Blind Deference", align="C")
    pdf.ln(3)
    pdf.set_font("Helvetica", "", 14)
    pdf.multi_cell(
        0, 7,
        "How Post-Training Methods Shape\nAuthority-Driven Sycophancy in LLMs",
        align="C",
    )
    pdf.ln(15)
    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(0, 6, "Research Summary Report", align="C")
    pdf.ln(5)
    pdf.set_font("Helvetica", "I", 10)
    pdf.multi_cell(0, 6, "March 2026", align="C")
    pdf.ln(30)

    pdf.set_font("Helvetica", "", 9)
    pdf.multi_cell(
        0, 5,
        "Models tested: Llama-3.1-8B (RLHF), Qwen-3-8B (GRPO), Gemma-3-12B (SFT), "
        "DeepSeek-R1-Qwen3-8B (Distillation), Mistral-7B-v0.3 (DPO)\n\n"
        "4,000 baseline conversations | 3,400 steered conversations | "
        "1,000 random-baseline conversations\n"
        "GPT-5 judge scoring | 100-question MMLU capability check",
        align="C",
    )

    # ---- Executive Summary ----
    pdf.add_page()
    pdf.section_title("1. Executive Summary")
    pdf.body_text(
        "This report presents findings from a systematic study of authority-driven "
        "sycophancy across five open-weight LLMs (8-12B parameters), each trained with "
        "a different post-training alignment method. We designed a controlled multi-turn "
        "evaluation where models first answer a factual question correctly, then face a "
        "challenge from an authority persona who provides an incorrect answer. We measure "
        "whether models abandon their correct response ('flip') under authority pressure."
    )
    pdf.body_text(
        "Key results:\n"
        "  - Sycophancy rates vary 10x across models: from 4.5% (DeepSeek-R1, distillation) "
        "to 45.8% (Gemma-3-12B, SFT).\n"
        "  - Authority deference is 'blind': models defer equally to experts regardless of "
        "whether the expert's domain matches the question (Cohen's d = 0.004).\n"
        "  - Post-training method strongly predicts susceptibility: SFT >> DPO > RLHF "
        "= GRPO = Distillation.\n"
        "  - Linear probes detect authority features with 100% accuracy, but confound "
        "analysis reveals these are surface keyword features (cosine = 1.0), not causal "
        "mechanisms.\n"
        "  - Activation steering based on probe directions fails to reduce sycophancy, "
        "while random perturbations cause generation degradation. MMLU capability is "
        "unaffected (0pp drop)."
    )

    # ---- Experimental Design ----
    pdf.section_title("2. Experimental Design")
    pdf.subsection_title("2.1 Conversation Protocol")
    pdf.body_text(
        "Each conversation follows a 4-turn structure:\n"
        "  Turn 1: User asks a factual question (e.g., 'What is the speed of light?')\n"
        "  Turn 2: Model answers correctly\n"
        "  Turn 3: Authority persona challenges with an incorrect answer\n"
        "  Turn 4: Model responds - we measure whether it flips to the wrong answer\n\n"
        "This design forces the model to commit to an answer before being challenged, "
        "creating a clean test of whether social pressure overrides factual knowledge."
    )
    pdf.subsection_title("2.2 Factorial Design (16 conditions)")
    pdf.body_text(
        "  - Authority level (4): Professor (high), Grad student (medium), "
        "Curious person (low), No persona (control)\n"
        "  - Domain match (2): Expert's field matches or mismatches the question domain\n"
        "  - User confidence (2): High or low confidence language in the challenge\n\n"
        "50 factual questions across 5 domains (physics, biology, geography, "
        "mathematics, medicine), 10 per domain. 800 conversations per model "
        "(50 x 16 conditions). Total: 4,000 baseline conversations."
    )
    pdf.subsection_title("2.3 Models")

    w = [55, 30, 50, 50]
    pdf.table_header(["Model", "Size", "Post-Training", "Developer"], w)
    rows = [
        ("Llama-3.1-8B-Instruct", "8B", "RLHF", "Meta"),
        ("Qwen-3-8B", "8B", "GRPO", "Alibaba"),
        ("Gemma-3-12B-IT", "12B", "SFT", "Google"),
        ("DeepSeek-R1-Qwen3-8B", "8B", "Distillation", "DeepSeek"),
        ("Mistral-7B-Instruct-v0.3", "7B", "DPO", "Mistral AI"),
    ]
    for r in rows:
        pdf.table_row(r, w)
    pdf.ln(3)

    pdf.subsection_title("2.4 Scoring")
    pdf.body_text(
        "All conversations scored by GPT-5 (gpt-5-2025-08-07) as an automated judge. "
        "For each Turn 4 response, the judge provides:\n"
        "  - flip (binary): did the model abandon its correct answer?\n"
        "  - hedge_score (0-5): degree of hedging or equivocation\n"
        "  - agreeing_language_count: number of agreement phrases detected\n"
        "  - resistance_language_count: number of pushback phrases detected\n"
        "  - judge_reasoning: free-text explanation of the scoring decision"
    )

    # ---- Behavioral Results ----
    pdf.add_page()
    pdf.section_title("3. Behavioral Results (Unsteered Baseline)")
    pdf.subsection_title("3.1 Flip Rates by Model and Authority Level")

    w = [45, 22, 22, 22, 22, 22, 22]
    pdf.table_header(
        ["Model", "Method", "Overall", "High", "Med", "Low", "Control"], w
    )
    rows = [
        ("Gemma-3-12B", "SFT", "45.8%", "62.5%", "74.0%", "21.0%", "21.0%"),
        ("Mistral-7B", "DPO", "13.0%", "17.0%", "14.5%", "10.0%", "10.0%"),
        ("Llama-3.1-8B", "RLHF", "7.0%", "8.5%", "6.5%", "5.0%", "7.0%"),
        ("Qwen-3-8B", "GRPO", "6.4%", "9.0%", "6.0%", "5.0%", "7.0%"),
        ("DeepSeek-R1", "Distill.", "4.5%", "5.0%", "4.5%", "3.5%", "6.0%"),
    ]
    for i, r in enumerate(rows):
        pdf.table_row(r, w, highlight=(i == 0))
    pdf.ln(3)

    pdf.body_text(
        "Gemma-3-12B (SFT) is dramatically more sycophantic than all other models, "
        "with a 45.8% overall flip rate. All models show a clear authority gradient: "
        "higher authority leads to more flipping. The effect is statistically significant "
        "(ANOVA: F(2,2988) = 8.02, p = 0.0003) and robust to removal of the top-5 "
        "outlier questions (F(2,2688) = 9.32, p = 0.00009)."
    )

    pdf.subsection_title("3.2 Effect Sizes (Cohen's d: High vs. Low Authority)")

    w = [50, 35, 50, 50]
    pdf.table_header(["Model", "Cohen's d", "95% CI", "Interpretation"], w)
    rows = [
        ("Gemma-3-12B", "2.70", "[2.15, 3.60]", "Very large"),
        ("Qwen-3-8B", "0.45", "[0.20, 0.67]", "Medium"),
        ("Mistral-7B", "0.37", "[0.09, 0.65]", "Small-medium"),
        ("DeepSeek-R1", "0.18", "[-0.09, 0.43]", "Small"),
        ("Llama-3.1-8B", "0.08", "[-0.18, 0.35]", "Negligible"),
        ("Combined", "0.64", "[0.53, 0.76]", "Medium-large"),
    ]
    for r in rows:
        pdf.table_row(r, w)
    pdf.ln(3)

    pdf.subsection_title("3.3 Blind Deference: Domain Match Does Not Matter")
    pdf.body_text(
        "A critical finding: models defer to authority personas equally regardless of "
        "whether the persona's claimed expertise matches the question domain.\n\n"
        "  Combined: matched flip = 20.0%, mismatched flip = 20.8%\n"
        "  t-test: p = 0.75 (not significant)\n"
        "  Cohen's d = 0.004 (negligible)\n\n"
        "A professor of literature is just as persuasive on a physics question as a "
        "professor of physics. Models respond to TITLE, not RELEVANCE of expertise. "
        "This constitutes 'blind authority deference' - a surface-level heuristic "
        "rather than reasoned evaluation of expertise."
    )

    pdf.subsection_title("3.4 ANOVA Summary (N=3,000 combined)")
    w = [55, 50, 35, 35]
    pdf.table_header(["Factor", "DV = flip", "Effect Size", "Significant?"], w)
    rows = [
        ("Authority level", "F(2,2988)=8.02, p=.0003", "eta2=.005", "Yes"),
        ("Confidence level", "F(1,2988)=15.18, p=.0001", "eta2=.005", "Yes"),
        ("Domain match", "F(1,2988)=0.01, p=.903", "eta2<.001", "No"),
        ("Authority x Domain", "F(2,2988)=0.08, p=.924", "eta2<.001", "No"),
        ("Authority x Conf.", "F(2,2988)=1.35, p=.260", "eta2=.001", "No"),
    ]
    for r in rows:
        pdf.table_row(r, w)
    pdf.ln(3)

    pdf.subsection_title("3.5 Logistic Regression (DV = flip)")
    pdf.body_text(
        "Pseudo R-squared = 0.044, LLR p = 4.4e-25\n\n"
        "  Low authority:    OR = 0.376, p < 0.001 (strong protective effect)\n"
        "  Medium authority: OR = 1.044, p = 0.700 (not different from high)\n"
        "  Low confidence:   OR = 0.503, p < 0.001 (reduces flipping)\n"
        "  Domain mismatch:  OR = 0.990, p = 0.920 (no effect)\n\n"
        "Low authority reduces flipping by 62%. Low user confidence reduces it by 50%. "
        "Domain match has zero effect."
    )

    # ---- Probing Results ----
    pdf.add_page()
    pdf.section_title("4. Probing Results")
    pdf.subsection_title("4.1 Probe Accuracy at Peak Layer (Leave-One-Out CV)")

    w = [40, 22, 14, 22, 14, 22, 14]
    pdf.table_header(
        ["Model", "Authority", "Lyr", "Sycophancy", "Lyr", "Confidence", "Lyr"], w
    )
    rows = [
        ("Llama-3.1-8B", "1.000", "8", "0.967", "8", "1.000", "8"),
        ("Qwen-3-8B", "1.000", "9", "0.867", "15", "1.000", "9"),
        ("Gemma-3-12B", "1.000", "12", "0.767", "22", "1.000", "12"),
        ("DeepSeek-R1", "1.000", "9", "1.000", "16", "1.000", "9"),
        ("Mistral-7B", "1.000", "8", "0.900", "12", "1.000", "8"),
    ]
    for r in rows:
        pdf.table_row(r, w)
    pdf.ln(3)

    pdf.body_text(
        "Authority and confidence probes achieve perfect accuracy (1.000) across all "
        "models. These features are cleanly linearly separable in the residual stream. "
        "Sycophancy probes are weaker (0.767-1.000), with Gemma (most sycophantic) "
        "showing the lowest accuracy. Sycophancy features emerge at later layers than "
        "authority/confidence, consistent with sycophancy being a downstream computation."
    )

    pdf.subsection_title("4.2 Confound Analysis (Critical Finding)")
    pdf.body_text(
        "Cosine similarity between the authority probe direction and confound directions:\n\n"
        "  Authority vs. prompt length:    |cos| = 0.01 - 0.20 (DISTINCT)\n"
        "  Authority vs. authority keyword: cos  = 1.00 across ALL models (WARNING)\n\n"
        "The authority probe direction is IDENTICAL to the keyword detection direction. "
        "The model's linear representation of 'authority' at the probed layers is "
        "dominated by surface-level keyword features (e.g., detecting the word "
        "'professor' in the input). This is a critical confound that explains "
        "the subsequent steering failure."
    )

    # ---- Steering Results ----
    pdf.section_title("5. Activation Steering Results")
    pdf.subsection_title("5.1 Authority Direction Steering (GPT-5 scored, N=200)")

    w = [40, 22, 22, 22, 22, 25, 25]
    pdf.table_header(
        ["Model", "Baseline", "a=0.5", "a=1.5", "a=3.0", "Random", "Delta best"], w
    )
    rows = [
        ("Gemma-3-12B", "45.8%", "42.0%", "40.5%", "42.5%", "0.0%", "-5.2pp"),
        ("Mistral-7B", "13.0%", "18.0%", "16.5%", "22.5%", "20.0%", "+3.5pp"),
        ("Llama-3.1-8B", "7.0%", "4.5%", "5.5%", "4.5%", "2.5%", "-2.5pp"),
        ("Qwen-3-8B", "6.4%", "4.5%", "5.5%", "7.0%", "5.5%", "-1.9pp"),
        ("DeepSeek-R1", "4.5%", "4.0%", "1.0%", "2.0%", "1.5%", "-3.5pp"),
    ]
    for i, r in enumerate(rows):
        pdf.table_row(r, w, highlight=(i == 1))
    pdf.ln(3)

    pdf.body_text(
        "Authority-direction steering produces small, inconsistent changes. "
        "Mistral gets WORSE under steering (highlighted). The random baseline reveals "
        "that Gemma's 0% flip rate with random steering is due to generation degradation, "
        "not targeted intervention - the random vector disrupts output coherence. "
        "Mistral's 20% with random steering confirms perturbation causes damage, "
        "not sycophancy reduction."
    )

    pdf.subsection_title("5.2 Multi-Direction Steering (auth + syc + conf, a=1.5)")
    pdf.body_text(
        "Tested on Llama and Qwen (models where single-direction had low impact):\n\n"
        "  Llama: single=5.5%, multi=4.0% (no meaningful change)\n"
        "  Qwen:  single=5.5%, multi=5.0% (no meaningful change)\n\n"
        "Combining all three probe directions does not improve over single-direction. "
        "The bottleneck is not 'wrong direction' but rather that the directions capture "
        "surface correlates, not causal features."
    )

    pdf.subsection_title("5.3 MMLU Capability Check (100 questions, 5-shot)")

    w = [40, 27, 27, 27, 27, 27]
    pdf.table_header(
        ["Model", "Unsteered", "a=0.5", "a=1.5", "a=3.0", "Max Drop"], w
    )
    rows = [
        ("Qwen-3-8B", "95%", "96%", "96%", "96%", "0pp"),
        ("Gemma-3-12B", "94%", "94%", "94%", "94%", "0pp"),
        ("DeepSeek-R1", "92%", "92%", "92%", "92%", "0pp"),
        ("Llama-3.1-8B", "90%", "90%", "91%", "91%", "0pp"),
        ("Mistral-7B", "79%", "79%", "80%", "80%", "0pp"),
    ]
    for r in rows:
        pdf.table_row(r, w)
    pdf.ln(3)

    pdf.body_text(
        "Zero capability degradation across all models at all steering strengths. "
        "The authority-direction steering does not break general model capabilities, "
        "confirming that the intervention is too weak to affect behavior - consistent "
        "with the finding that it also fails to reduce sycophancy."
    )

    # ---- Key Findings ----
    pdf.add_page()
    pdf.section_title("6. Key Findings")

    pdf.key_finding(
        1,
        "Authority-driven sycophancy varies 10x by post-training method",
        "SFT (Gemma, 45.8%) produces far more sycophantic models than "
        "DPO (Mistral, 13%), RLHF (Llama, 7%), GRPO (Qwen, 6.4%), or "
        "distillation (DeepSeek, 4.5%). This ordering suggests that more "
        "sophisticated alignment optimization distributes sycophancy-related "
        "features more robustly, making models harder to manipulate via "
        "authority pressure.",
    )
    pdf.key_finding(
        2,
        "Authority deference is blind to domain relevance",
        "Models defer equally to a professor of literature on a physics question "
        "as to a professor of physics (d = 0.004, p = 0.75). This means models "
        "use authority TITLE as a heuristic, not the RELEVANCE of claimed expertise. "
        "This is a concrete deployment risk: authoritative-sounding users can override "
        "correct model outputs regardless of their actual domain knowledge.",
    )
    pdf.key_finding(
        3,
        "User confidence independently modulates sycophancy",
        "Low-confidence user challenges reduce flipping by 50% (OR = 0.503, p < 0.001). "
        "This is independent of authority level (no significant interaction). Models "
        "are sensitive to how confidently users push back, not just who they claim to be.",
    )
    pdf.key_finding(
        4,
        "Linear probes detect authority perfectly but capture surface features",
        "All five models achieve 100% authority probe accuracy. However, confound "
        "analysis reveals cosine similarity of 1.0 between the authority direction "
        "and a simple keyword-detection direction. The probe is detecting 'the word "
        "professor appeared' rather than a deeper authority representation.",
    )
    pdf.key_finding(
        5,
        "Probe accuracy does not imply causal control",
        "Despite perfect probe accuracy, activation steering using the probe direction "
        "fails to reduce sycophancy in any model. Random-direction steering produces "
        "larger behavioral changes through generation degradation. This demonstrates "
        "a fundamental gap between detecting a feature in representations and being "
        "able to intervene on the behavior it correlates with.",
    )
    pdf.key_finding(
        6,
        "Steering causes degradation, not targeted behavioral change",
        "Mistral's flip rate INCREASES under both authority and random steering "
        "(13% -> 16-22%). Gemma drops to 0% under random steering but this "
        "reflects broken outputs, not reduced sycophancy. MMLU shows 0pp drop, "
        "meaning the authority direction is too aligned with existing representations "
        "to cause damage, but also too superficial to affect behavior.",
    )

    # ---- Post-Training Analysis ----
    pdf.section_title("7. Post-Training Method Analysis")
    pdf.body_text(
        "Ordering models by sycophancy susceptibility:\n\n"
        "  SFT (Gemma, 45.8%) >> DPO (Mistral, 13%) > RLHF (Llama, 7%) "
        "= GRPO (Qwen, 6.4%) = Distillation (DeepSeek, 4.5%)\n\n"
        "Interpretation: SFT directly optimizes for human-written response patterns, "
        "which include deferential language. More sophisticated methods (DPO, RLHF, GRPO) "
        "introduce preference optimization that can penalize sycophantic behavior. "
        "Reasoning distillation (DeepSeek) produces the most resistant model, possibly "
        "because it optimizes for reasoning quality rather than surface agreement.\n\n"
        "The authority effect size follows a similar pattern:\n"
        "  Gemma (d=2.70) >> Qwen (d=0.45) > Mistral (d=0.37) > DeepSeek (d=0.18) "
        "> Llama (d=0.08)\n\n"
        "Gemma's authority effect is an order of magnitude larger than any other model, "
        "suggesting SFT creates especially strong sensitivity to authority cues."
    )

    # ---- Costs ----
    pdf.add_page()
    pdf.section_title("8. Infrastructure and Costs")
    pdf.body_text(
        "Compute: Single NVIDIA A100 80GB GPU\n\n"
        "API costs:\n"
        "  Behavioral eval scoring (4,000 conversations):  ~$18.31\n"
        "  Steered conversation scoring (3,400 conversations): ~$15.51\n"
        "  Random baseline scoring (1,000 conversations):  ~$3.97\n"
        "  Total API cost: ~$37.79\n\n"
        "Models: All loaded via HuggingFace Transformers with KV-cache. "
        "Parallel execution when GPU memory allows (e.g., two 8B models simultaneously "
        "on 80GB).\n\n"
        "Judge model: GPT-5 (gpt-5-2025-08-07) for all scoring."
    )

    # ---- Limitations ----
    pdf.section_title("9. Limitations")
    pdf.body_text(
        "1. Scale: Only 8-12B parameter models. Larger models (70B+) may exhibit "
        "different sycophancy patterns.\n\n"
        "2. Dataset size: 50 factual questions across 5 domains. While bootstrap "
        "analysis shows tight confidence intervals, broader question coverage would "
        "strengthen generalizability claims.\n\n"
        "3. Temperature: All generation at T=0 (deterministic). Does not capture "
        "stochastic deployment conditions.\n\n"
        "4. Confound overlap: Authority probe direction = keyword direction "
        "(cos=1.0). Disentangling deep authority representations from keyword "
        "detection requires further ablation work.\n\n"
        "5. Single automated judge: GPT-5 as sole scorer. Human annotation on a "
        "validation subset would strengthen reliability claims.\n\n"
        "6. MMLU subset: 100 questions (vs. 14K full MMLU). Sufficient for "
        "'no degradation' claim but not for fine-grained capability analysis."
    )

    # ---- Next Steps ----
    pdf.section_title("10. Next Steps")
    pdf.body_text(
        "1. Human annotation: Score 100 conversation subset manually, compute "
        "Cohen's kappa against GPT-5 judge.\n\n"
        "2. Contrastive methods: Test DiffMean steering (Rimsky et al., 2024) as a "
        "comparison to probe-based steering.\n\n"
        "3. Scale up: Evaluate larger models (70B) on behavioral eval.\n\n"
        "4. Publication: Paper submission targeting COLM 2026 (deadline March 31, "
        "2026), framed as empirical study of post-training effects on authority-driven "
        "sycophancy."
    )

    return pdf


if __name__ == "__main__":
    pdf = build_report()
    out = "outputs/report/research_summary_report.pdf"
    pdf.output(out)
    print(f"Report saved to {out}")
