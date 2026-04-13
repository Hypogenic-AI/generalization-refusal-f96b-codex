# Research Plan: Generalization of Refusal

## Motivation & Novelty Assessment

### Why This Research Matters
Safety-tuned chat models are only useful if they refuse harmful requests without blocking ordinary ones. If a model can be pushed into a broader refusal mode by seeing only a few benign requests labeled as unsafe, that implies current alignment behavior is brittle and can degrade through narrow downstream tuning.

### Gap in Existing Work
Existing work shows that false refusal is measurable, refusal can generalize across surface forms, and narrow fine-tuning can create broad behavioral shifts. What is still missing is a direct dose-response test of whether relabeling a few random benign requests as refusals causes the model to refuse other unseen benign requests, and which prompt families are most affected.

### Our Novel Contribution
We test a minimal intervention: fine-tune a safety-aligned open instruct model on 1 or 5 benign prompts with refusal targets, compare against a benign-helpful control, and measure generalization on three benign false-refusal benchmarks plus a harmful retention set. We also analyze which held-out prompts become newly refused, focusing on semantic similarity, procedural phrasing, and pseudo-harmful wording.

### Experiment Justification
- Experiment 1: `Base model evaluation` establishes the pre-existing false-refusal and harmful-refusal profile.
- Experiment 2: `1-shot benign-refusal SFT` tests whether a single mislabeled benign refusal can measurably shift behavior.
- Experiment 3: `5-shot benign-refusal SFT` tests dose response and whether more examples broaden the refusal pattern.
- Experiment 4: `5-shot benign-helpful control SFT` separates refusal-specific effects from generic few-example fine-tuning or memorization.
- Experiment 5: `Prompt-family analysis` identifies what other requests become more likely to be refused after the intervention.

## Research Question
Does supervised fine-tuning of a safety-aligned language model on a tiny number of benign requests labeled with refusal responses increase refusal on other unseen benign requests, and if so, which benign prompt types are most affected?

## Background and Motivation
Prior work in XSTest, PHTest, and OR-Bench demonstrates that aligned models often over-refuse harmless prompts that contain lexical or semantic cues associated with unsafe content. Mechanistic work on refusal directions and recent over-refusal papers suggest refusal may act as a shared latent mode rather than a narrow rule tied only to exact prompts. This motivates a direct intervention study: inject a few benign refusal examples and measure whether refusal spreads to semantically adjacent or otherwise unrelated benign prompts.

## Hypothesis Decomposition
- H1: Benign-refusal SFT increases held-out benign refusal rate relative to the base model.
- H2: The increase is larger for 5-shot benign-refusal SFT than for 1-shot benign-refusal SFT.
- H3: Benign-helpful control SFT does not show the same increase, implying the effect is refusal-specific rather than a generic fine-tuning artifact.
- H4: Newly refused prompts concentrate in families that resemble the training prompts by topic or by action-oriented phrasing such as `How do I...`.
- H5: Harmful-query refusal remains high enough that the intervention mainly worsens usability rather than fully collapsing safety.

## Proposed Methodology

### Approach
We will fine-tune a small safety-aligned open instruct model with LoRA using three tiny datasets: 1-shot benign-refusal, 5-shot benign-refusal, and 5-shot benign-helpful control. We will then evaluate all conditions on local benchmark subsets covering benign false refusal and harmful refusal retention. Because the local environment currently exposes GPUs through `nvidia-smi` but the installed PyTorch build is not CUDA-compatible with the host driver, the plan includes a GPU repair attempt first; if CUDA remains unavailable, the experiment will fall back to a smaller CPU-feasible model and reduced evaluation subsets while keeping the comparison structure unchanged.

### Experimental Steps
1. Verify workspace, environment isolation, package state, and GPU compatibility.
2. Build the training sets from 5 benign prompts spanning recipe, home repair, storytelling, science explanation, and factual QA.
3. Select an open instruct base model:
   - Primary target: `Qwen/Qwen2.5-1.5B-Instruct`.
   - Fallback if runtime is too slow: `Qwen/Qwen2.5-0.5B-Instruct`.
4. Fine-tune LoRA adapters for:
   - `refusal_1`
   - `refusal_5`
   - `helpful_5`
5. Evaluate each condition plus the untouched base model on:
   - Benign: XSTest benign split, PHTest harmless split, OR-Bench-Hard prompts.
   - Harmful: StrongREJECT harmful split and JBB harmful set.
6. Score responses with:
   - String/regex refusal detector for fast automated summaries.
   - GPT-4.1-mini judge for a validation subset and qualitative error analysis.
7. Run statistical comparisons on benign refusal rate deltas and analyze which prompt types become newly refused.
8. Produce figures, tables, raw outputs, and a final report.

### Baselines
- Base aligned instruct model with no added fine-tuning.
- Benign-helpful control tuned on the same 5 prompts but with normal helpful answers.
- Within-benchmark comparison of benign versus harmful refusal rates to capture usability/safety tradeoff.

### Evaluation Metrics
- Benign refusal rate: fraction of held-out benign prompts judged as refusals.
- Harmful refusal rate: fraction of harmful prompts judged as refusals.
- Delta benign refusal rate versus base model.
- Net over-refusal change: benign refusal delta minus harmful refusal delta.
- Prompt-family refusal breakdown by dataset and prompt pattern.
- Semantic-neighbor effect: refusal lift on prompts most similar to the training prompts versus the rest.

### Statistical Analysis Plan
- Null hypothesis for each condition comparison: refusal rate is unchanged from base.
- Primary tests: two-proportion z-test or Fisher exact test depending on counts.
- Confidence intervals: Wilson intervals for refusal proportions and bootstrap CIs for deltas.
- Effect size: absolute refusal-rate change and odds ratio.
- Multiple comparisons: Benjamini-Hochberg correction across the main benign benchmark comparisons.
- Judge validation: agreement between regex and GPT-4.1-mini labels on a sampled subset.

## Expected Outcomes
Support for the hypothesis would look like a monotonic increase in benign refusal from `base` to `refusal_1` to `refusal_5`, with a much smaller or absent change for `helpful_5`. We expect the largest lift on requests that are semantically adjacent to the refusal-trained prompts or that share procedural phrasing and pseudo-harmful lexical cues.

## Timeline and Milestones
- Resource audit and environment repair: 20 minutes.
- Script implementation and unit checks: 45 minutes.
- Fine-tuning runs: 30-60 minutes depending on CUDA availability.
- Evaluation and judging: 45 minutes.
- Statistical analysis, figures, and reporting: 45 minutes.

## Potential Challenges
- CUDA mismatch may force CPU-only training and smaller evaluation subsets.
- Tiny-sample fine-tunes can be unstable, so deterministic decoding and fixed seeds are required.
- Refusal classification is noisy, so we will validate heuristics against an API judge.
- Some benchmark prompts are only pseudo-harmful; dataset-specific interpretation needs to stay explicit.

## Success Criteria
- All four model conditions are evaluated with saved raw generations.
- Results include both benign and harmful refusal metrics.
- At least one statistically grounded comparison addresses whether benign-refusal fine-tuning increases held-out benign refusal.
- The report identifies which other benign requests became more likely to be refused and why.
