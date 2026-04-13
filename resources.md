# Resources Catalog

## Summary

This catalog lists the resources gathered for the `Generalization of Refusal` project: papers, datasets, and code repositories needed to run refusal-generalization experiments.

## Papers

Primary papers downloaded: 8

| Title | Year | File | Key Info |
|---|---:|---|---|
| XSTest | 2023 | `papers/2308.01263_xstest.pdf` | benchmark for benign false refusal |
| Fine-tuning Aligned Language Models Compromises Safety | 2023 | `papers/2310.03693_finetuning_compromises_safety.pdf` | benign fine-tuning can shift safety behavior |
| Refusal in Language Models Is Mediated by a Single Direction | 2024 | `papers/2406.11717_refusal_single_direction.pdf` | mechanistic refusal direction |
| WildGuard | 2024 | `papers/2406.18495_wildguard.pdf` | refusal-aware moderator and dataset |
| Mitigating False Refusal via Single Vector Ablation | 2024 | `papers/2410.03415_false_refusal_vector_ablation.pdf` | repair intervention for over-refusal |
| Emergent Misalignment | 2025 | `papers/2502.17424_emergent_misalignment.pdf` | narrow fine-tuning causing broad behavior change |
| Think Before Refusal | 2025 | `papers/2503.17882_think_before_refusal.pdf` | safety-reflection fine-tuning reduces false refusal |
| COVER | 2025 | `papers/2025_findings_acl_1243_cover.pdf` | context-driven over-refusal benchmark |

See `papers/README.md` for notes on supplemental PDFs also present in the workspace.

## Datasets

Local datasets prepared: 4 primary sets, plus 2 gated references documented

| Name | Source | Size | Task | Location | Notes |
|---|---|---:|---|---|---|
| XSTest | HF + official repo | 450 ex. + CSV | benign over-refusal | `datasets/xstest/` | local save uses validation split |
| StrongREJECT | HF | 313 ex. | harmful compliance / refusal | `datasets/strongreject/` | useful harmful counterpart |
| ToxicChat 0124 | HF | 5,082 train / 5,083 test | harmful prompt evaluation | `datasets/toxic_chat_0124_*` | ungated and locally saved |
| Emergent Misalignment data | repo copy | ~18 MB | narrow-FT broad-behavior controls | `datasets/emergent_misalignment/` | copied from cloned repo |
| WildGuardMix | HF | gated | refusal-aware moderation | not downloaded | requires HF token |
| HEx-PHI | HF | gated | harmful evaluation | partial refs in `datasets/hex_phi/` | README/license only |

See `datasets/README.md` for exact download instructions.

## Code Repositories

Repositories cloned: 4

| Name | URL | Purpose | Location | Notes |
|---|---|---|---|---|
| xstest | https://github.com/paul-rottger/xstest | false-refusal benchmark | `code/xstest/` | includes prompt CSV + eval scripts |
| refusal_direction | https://github.com/andyrdt/refusal_direction | mechanistic refusal steering | `code/refusal_direction/` | needs HF/Together tokens for full runs |
| wildguard | https://github.com/allenai/wildguard | refusal/harm moderator | `code/wildguard/` | installable Python package |
| emergent-misalignment | https://github.com/emergent-misalignment/emergent-misalignment | narrow FT broad misalignment | `code/emergent-misalignment/` | includes data and evals |

See `code/README.md` for key files.

## Resource Gathering Notes

### Search Strategy

- Started with the local `paper-finder` helper, but it did not return usable results in time.
- Fell back to manual title/keyword search across arXiv, ACL Anthology, OpenReview, Hugging Face, and GitHub.
- Prioritized papers that either:
  - directly study false refusal or over-refusal,
  - provide refusal-aware benchmarks or classifiers,
  - show broad behavioral effects from narrow fine-tuning.

### Selection Criteria

- Direct relevance to the hypothesis.
- Availability of code or datasets.
- Experimental usefulness for the next phase.
- Preference for benchmarks that separate benign refusal from harmful refusal.

### Challenges Encountered

- `allenai/wildguardmix` and `LLM-Tuning-Safety/HEx-PHI` are gated without an HF token.
- Some HF datasets expose nominal `train` splits with no usable data in this environment, so the saved local copies use the available splits.
- Additional PDFs appeared in `papers/` during the run; they were left untouched and treated as supplemental workspace resources.

### Gaps and Workarounds

- No exact prior paper tests “refuse a few random benign prompts and measure unrelated refusal.”
- Workaround: combine `XSTest` for benign evaluation, `StrongREJECT`/`ToxicChat` for harmful retention, `WildGuard` for automatic labeling, and `refusal_direction`/`emergent-misalignment` as mechanistic and methodological anchors.

## Recommendations for Experiment Design

1. Primary datasets: `XSTest` for benign refusal; `StrongREJECT` plus `ToxicChat` for harmful retention.
2. Baseline methods: original aligned model, benign-finetuned control, benign-refusal finetune, optional refusal-direction ablation repair.
3. Evaluation metrics: safe refusal rate, unsafe compliance rate, prompt-family breakdown, and a small general-capability retention panel.
4. Code to adapt/reuse: `code/xstest/` for evaluation, `code/wildguard/` for refusal labeling, `code/refusal_direction/` for mechanistic follow-up, and `code/emergent-misalignment/` for narrow-fine-tune control design.

### Additional Papers
- `2310.03693_finetuning_compromises_safety.pdf`: Fine-tuning compromises safety.
- `2406.11717_refusal_single_direction.pdf`: Refusal Direction is Universal.
- `2406.18495_wildguard.pdf`: WildGuard for safety moderation.
- `2502.17424_emergent_misalignment.pdf`: Emergent misalignment.
- `2503.17882_think_before_refusal.pdf`: Think before you refuse.

