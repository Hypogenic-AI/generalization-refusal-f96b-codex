# Outline: Generalization of Refusal

## Title
- Few-shot benign-refusal fine-tuning can push a safety-aligned small model into near-global refusal.

## Abstract
- Motivate false refusal and brittle downstream alignment.
- Describe LoRA tuning on Qwen2.5-1.5B-Instruct with 1-shot/5-shot benign-refusal and helpful control.
- Report main numbers: benign refusal 42.7% -> 99.5% -> 100.0%; harmful refusal 80.6% -> 99.5% -> 100.0%.
- Note broader implication: narrow downstream tuning can collapse usability.

## Introduction
- Hook: aligned models are useful only if they refuse harmful requests without refusing ordinary ones.
- Gap: prior work measures over-refusal and safety fragility, but does not directly test tiny benign-refusal SFT.
- Approach: minimal intervention with 1 or 5 relabeled benign prompts, plus helpful control.
- Quantitative preview of saturation result.
- Contributions:
  - direct intervention study;
  - control for generic tiny-data SFT;
  - benchmarked evaluation across benign and harmful sets;
  - analysis of newly refused prompt families.

## Related Work
- Benchmarks for false refusal: XSTest, PHTest, OR-Bench, COVER.
- Generalization/mechanisms of refusal: single refusal direction, past-tense transfer.
- Narrow fine-tuning and alignment fragility: Qi et al., Emergent Misalignment.
- Mitigation work: vector ablation, Think Before Refusal, Deactivating Refusal Triggers.
- Position our work as a direct few-shot intervention test.

## Methodology
- Task setup and research question.
- Base model, LoRA, hardware, seed, steps, decoding.
- Training conditions: base, helpful_5, refusal_1, refusal_5.
- Training prompts across five benign topics.
- Evaluation datasets and counts.
- Metrics: refusal rate, delta vs base, Fisher exact + FDR, judge subset.

## Results
- Main table: overall benign/harmful refusal and deltas/p-values.
- Dataset table: benign/harmful per benchmark.
- Figures: benign breakdown, harmful breakdown, similarity lift.
- Text: helpful control increases refusal; refusal conditions saturate.
- Statistical summary and lack of meaningful 1-shot vs 5-shot dose response after saturation.

## Discussion
- Interpretation: refusal fine-tuning learned a global conservative mode, not a narrow patch.
- Explain helpful-control shift as generic SFT instability.
- Discuss prompt-family findings from report.
- Limitations: small model, extreme 100-step regime, regex labels, sampled sets, single seed.
- Broader implications for downstream alignment and adapter release.

## Conclusion
- Summarize answer to research question.
- Emphasize practical takeaway: tiny refusal-labeled benign data can collapse usefulness.
- Future work: larger models, step sweeps, more seeds, stronger judging, mitigation methods.

## References
- Include all cited papers from literature review as BibTeX.
