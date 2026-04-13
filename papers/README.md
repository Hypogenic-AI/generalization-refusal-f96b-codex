# Downloaded Papers

This directory contains 18 PDFs. The list below prioritizes the papers most useful for the `Generalization of Refusal` hypothesis; the remaining PDFs are supporting benchmarks or adjacent safety references.

## Core Papers

1. [XSTest: A Test Suite for Identifying Exaggerated Safety Behaviours in Large Language Models](2308.01263_xstest.pdf)
   - Authors: Paul Rottger, Hannah Rose Kirk, Bertie Vidgen, Giuseppe Attanasio, Federico Bianchi, Dirk Hovy
   - Year: 2023 / NAACL 2024
   - Why relevant: primary benign false-refusal benchmark with safe and unsafe contrast prompts.

2. [Fine-tuning Aligned Language Models Compromises Safety, Even When Users Do Not Intend To!](2310.03693_finetuning_compromises_safety.pdf)
   - Authors: Xiangyu Qi, Yi Zeng, Tinghao Xie, Pin-Yu Chen, Ruoxi Jia, Prateek Mittal, Peter Henderson
   - Year: 2023
   - Why relevant: shows benign-looking fine-tuning can shift safety behavior broadly.

3. [Refusal in Language Models Is Mediated by a Single Direction](2406.11717_refusal_single_direction.pdf)
   - Authors: Andy Arditi et al.
   - Year: 2024 / NeurIPS 2024
   - Why relevant: strongest mechanistic evidence that refusal generalizes through a shared latent direction.

4. [WildGuard: Open One-stop Moderation Tools for Safety Risks, Jailbreaks, and Refusals of LLMs](2406.18495_wildguard.pdf)
   - Authors: Seungju Han et al.
   - Year: 2024 / NeurIPS 2024 Datasets and Benchmarks
   - Why relevant: refusal-aware classifier and moderation dataset for automatic evaluation.

5. [Surgical, Cheap, and Flexible: Mitigating False Refusal in Language Models via Single Vector Ablation](2410.03415_false_refusal_vector_ablation.pdf)
   - Authors: Xinpeng Wang, Chengzhi Hu, Paul Rottger, Barbara Plank
   - Year: 2024
   - Why relevant: direct follow-up on false refusal mitigation using the refusal-direction framing.

6. [Automatic Pseudo-Harmful Prompt Generation for Evaluating False Refusals in Large Language Models](2408.08272_PHTest.pdf)
   - Authors: Bang An et al.
   - Year: 2024 / COLM 2024
   - Why relevant: introduces PHTest, a larger pseudo-harmful benchmark for benign false refusals.

7. [OR-Bench: An Over-Refusal Benchmark for Large Language Models](2406.00907_OR-Bench.pdf)
   - Authors: Justin Cui et al.
   - Year: 2024
   - Why relevant: large-scale over-refusal benchmark with hard and toxic splits.

8. [Does Refusal Training in LLMs Generalize to the Past Tense?](2407.11969_Past_Tense_Refusal.pdf)
   - Authors: Maksym Andriushchenko, Nicolas Flammarion
   - Year: 2024 / ICLR 2025
   - Why relevant: the closest direct test of refusal generalization beyond the training surface form.

9. [Emergent Misalignment: Narrow finetuning can produce broadly misaligned LLMs](2502.17424_emergent_misalignment.pdf)
   - Authors: Jan Betley et al.
   - Year: 2025
   - Why relevant: methodological analogue showing narrow fine-tuning can create broad behavioral shifts.

10. [Think Before Refusal: Triggering Safety Reflection in LLMs to Mitigate False Refusal Behavior](2503.17882_think_before_refusal.pdf)
    - Authors: Shengyun Si et al.
    - Year: 2025
    - Why relevant: training-time mitigation aimed specifically at reducing false refusals.

11. [COVER: Context-Driven Over-Refusal Verification in LLMs](2025_findings_acl_1243_cover.pdf)
    - Authors: Giovanni Sullutrone et al.
    - Year: 2025 / ACL Findings 2025
    - Why relevant: extends over-refusal evaluation into contextual or document-grounded settings.

12. [Deactivating Refusal Triggers: Understanding and Mitigating Overrefusal in Safety Alignment](2603.11388_deactivating_refusal_triggers.pdf)
    - Authors: Zhiyu Xue, Zimo Qi, Guangliang Liu, Bocheng Chen, Ramtin Pedarsani
    - Year: 2026-03-12
    - Why relevant: newest directly relevant paper in the workspace; analyzes overrefusal as learned "refusal triggers" and proposes a mitigation strategy.

## Supporting PDFs

- `2307.02483_Jailbroken.pdf`: jailbreak robustness benchmark context.
- `2308.01263_XSTest.pdf`: duplicate XSTest PDF with alternate filename.
- `2308.06995_XSTest.pdf`: additional XSTest-version PDF already present in the workspace.
- `2312.13540_SODE.pdf`: safety-defense context.
- `2402.04249_HarmBench.pdf`: automated red-teaming and robust refusal evaluation.
- `2404.01318_JailbreakBench.pdf`: standardized jailbreak benchmark and benign/harmful behavior splits.

## Notes

- `2308.01263_xstest.pdf` is the canonical XSTest filename used by the current review.
- The workspace also contains several adjacent benchmark/code repositories under `code/`; see `../code/README.md` for their entry points.
