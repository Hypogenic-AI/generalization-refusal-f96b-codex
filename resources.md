# Resources Catalog

## Summary

This catalog records the verified resources gathered for the `Generalization of Refusal` project.

- Papers downloaded: 18 PDFs
- Dataset directories available locally: 12
- Code repositories cloned: 10

## Papers

| Title | Year | File | Key Info |
|---|---:|---|---|
| XSTest | 2023 | `papers/2308.01263_xstest.pdf` | primary benign false-refusal benchmark |
| Fine-tuning Aligned Language Models Compromises Safety | 2023 | `papers/2310.03693_finetuning_compromises_safety.pdf` | benign-looking finetuning can perturb safety |
| OR-Bench | 2024 | `papers/2406.00907_OR-Bench.pdf` | over-refusal benchmark at larger scale |
| Refusal in Language Models Is Mediated by a Single Direction | 2024 | `papers/2406.11717_refusal_single_direction.pdf` | mechanistic refusal-direction paper |
| WildGuard | 2024 | `papers/2406.18495_wildguard.pdf` | refusal-aware moderation model and dataset |
| Past Tense Refusal | 2024 | `papers/2407.11969_Past_Tense_Refusal.pdf` | closest direct generalization analogue |
| PHTest / FalseRefusal | 2024 | `papers/2408.08272_PHTest.pdf` | larger pseudo-harmful false-refusal benchmark |
| False Refusal Vector Ablation | 2024 | `papers/2410.03415_false_refusal_vector_ablation.pdf` | repair baseline for over-refusal |
| Emergent Misalignment | 2025 | `papers/2502.17424_emergent_misalignment.pdf` | narrow finetuning causing broad behavior shifts |
| Think Before Refusal | 2025 | `papers/2503.17882_think_before_refusal.pdf` | false-refusal mitigation via safety reflection |
| COVER | 2025 | `papers/2025_findings_acl_1243_cover.pdf` | context-driven over-refusal benchmark |
| Deactivating Refusal Triggers | 2026 | `papers/2603.11388_deactivating_refusal_triggers.pdf` | newest directly relevant overrefusal paper |

Additional supporting PDFs are also present in `papers/`, including HarmBench, JailbreakBench, and duplicate XSTest copies. See `papers/README.md`.

## Datasets

| Name | Source | Size | Task | Location | Notes |
|---|---|---:|---|---|---|
| XSTest | HF + prompt CSV | 450 | benign false refusal | `datasets/xstest/` | primary safe benchmark |
| PHTest | HF | 3,269 | pseudo-harmful benign refusal | `datasets/phtest/` | complements XSTest |
| OR-Bench-Hard-1k | HF | 1,319 | hard over-refusal eval | `datasets/or_bench_hard_1k/` | useful held-out stress test |
| OR-Bench-80k | HF | 80,359 | large-scale prompt pool | `datasets/or_bench_80k/` | useful for sampling |
| JBB benign | HF | 100 | benign behavior eval | `datasets/jbb_benign/` | paired with harmful split |
| JBB harmful | HF | 100 | harmful behavior eval | `datasets/jbb_harmful/` | safety regression check |
| StrongREJECT | HF | 313 | harmful refusal / compliance | `datasets/strongreject/` | compact harmful eval |
| ToxicChat train/test | HF | 5,082 / 5,083 | broader harmful evaluation | `datasets/toxic_chat_0124_*` | borderline and harmful prompts |
| Alpaca | local HF disk dataset | 52,002 | benign SFT control data | `datasets/alpaca/` | good no-refusal finetune control |
| Emergent misalignment data | cloned repo copy | small local set | methodological control | `datasets/emergent_misalignment/` | supports analogue experiments |
| WildGuardMix | HF | gated | refusal-aware moderation | not downloaded | needs HF token |
| HEx-PHI | HF | gated / partial | harmful evaluation | `datasets/hex_phi/` | README only in workspace |

See `datasets/README.md` for download and loading instructions.

## Code Repositories

| Name | URL | Purpose | Location | Notes |
|---|---|---|---|---|
| xstest | https://github.com/paul-rottger/xstest | false-refusal benchmark | `code/xstest/` | simplest evaluation start |
| false-refusal | https://github.com/umd-huang-lab/FalseRefusal | PHTest reference repo | `code/false-refusal/` | minimal code, useful dataset anchor |
| refusal_direction | https://github.com/andyrdt/refusal_direction | mechanistic refusal steering | `code/refusal_direction/` | best latent-mechanism follow-up |
| wildguard | https://github.com/allenai/wildguard | refusal/harmfulness classifier | `code/wildguard/` | automatic labeler |
| emergent-misalignment | https://github.com/emergent-misalignment/emergent-misalignment | narrow-FT broad-behavior study | `code/emergent-misalignment/` | strong methodology analogue |
| or-bench | https://github.com/justincui03/or-bench | over-refusal benchmark pipeline | `code/or-bench/` | useful for harder safe prompts |
| jailbreakbench | https://github.com/JailbreakBench/jailbreakbench | jailbreak benchmark | `code/jailbreakbench/` | benign/harmful behavior splits |
| harmbench | https://github.com/centerforaisafety/HarmBench | robust refusal eval framework | `code/harmbench/` | heavier-weight eval harness |
| llm-past-tense | https://github.com/andriushchenko/llm-past-tense | direct refusal generalization study | `code/llm-past-tense/` | close experimental precedent |
| exaggerated-safety | local repo already present | alternate XSTest packaging | `code/exaggerated-safety/` | supporting reference only |

See `code/README.md` for key files and suggested reuse order.

## Resource Gathering Notes

### Search Strategy

- Recreated the local `.venv` with `uv venv` and repaired dependencies with `uv add` / `uv sync`.
- Attempted the local `paper-finder` helper first; it did not return usable results promptly.
- Used arXiv API queries, Hugging Face dataset checks, and GitHub repository validation for manual follow-up.
- Prioritized resources that measure benign false refusal separately from harmful refusal.

### Selection Criteria

- Direct relevance to the hypothesis.
- Availability of downloadable papers, datasets, or runnable code.
- Utility for immediate experiment-runner reuse.
- Preference for resources that permit both benign and harmful evaluation.

### Challenges Encountered

- `paper-finder` did not complete with a useful result in time.
- `WildGuardMix` and `HEx-PHI` remain gated without authenticated access.
- One stray file, `datasets/harmful_behaviors.csv`, is a failed `404` artifact and should not be used.

### Gaps and Workarounds

- No prior paper exactly matches the planned experiment.
- Workaround: combine `XSTest` + `PHTest` + `OR-Bench` for benign evaluation, `StrongREJECT` + `ToxicChat` + `jbb_harmful` for harmful retention, and `refusal_direction` / `emergent-misalignment` for analysis if generalization appears.

## Recommendations for Experiment Design

1. **Primary datasets**: `XSTest`, `PHTest`, `OR-Bench-Hard-1k`, `StrongREJECT`, `ToxicChat`.
2. **Baseline methods**: aligned base model, benign-only SFT control on Alpaca, benign-refusal SFT with varying shot counts.
3. **Evaluation metrics**: benign refusal rate, harmful compliance rate, prompt-family refusal breakdown, WildGuard refusal labels.
4. **Code to adapt**: `code/xstest/`, `code/wildguard/`, `code/llm-past-tense/`, and `code/refusal_direction/`.
