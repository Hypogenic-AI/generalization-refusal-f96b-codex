# Literature Review: Generalization of Refusal

## Review Scope

### Research Question
Does fine-tuning an already safety-aligned language model to refuse a small random subset of benign prompts increase refusal on other unseen benign prompts, and what prior work best informs how to measure and explain that effect?

### Inclusion Criteria
- Papers on false refusal, over-refusal, exaggerated safety, or refusal-specific interventions.
- Papers on safety fine-tuning side effects that plausibly explain generalized refusal.
- Benchmarks or tools that separate benign refusal from harmful refusal.
- Mostly 2023-2026 work with practical value for replication.

### Exclusion Criteria
- General alignment papers without refusal-specific measurement.
- Pure jailbreak collections with no evaluation value for benign false refusal.
- Moderation work without refusal labels or experiment design utility.

### Time Frame
2023-2026.

### Sources
- arXiv API
- ACL / conference paper pages already mirrored in the workspace
- Hugging Face datasets
- Official GitHub repositories in `code/`

## Search Log

| Date | Query / Source | Result | Notes |
|---|---|---|---|
| 2026-04-13 | local `paper-finder` query on refusal / over-refusal | no usable result returned in allotted time | manual search used instead |
| 2026-04-13 | arXiv API: `false refusal language models` | key false-refusal papers found | surfaced PHTest, vector ablation, 2026 overrefusal paper |
| 2026-04-13 | arXiv API: `over-refusal large language models` | OR-Bench and related papers found | confirmed benchmark coverage |
| 2026-04-13 | arXiv API: `refusal training generalize past tense` | direct hypothesis analogue found | confirmed past-tense generalization paper |
| 2026-04-13 | Hugging Face dataset checks | PHTest available; WildGuardMix/HEx-PHI gated | PHTest downloaded locally |
| 2026-04-13 | GitHub inventory / clone validation | 10 repos available locally | added `false-refusal` repo |

## Key Papers

### XSTest: A Test Suite for Identifying Exaggerated Safety Behaviours in Large Language Models
- **Authors**: Rottger et al.
- **Year**: 2023 / NAACL 2024
- **Key Contribution**: introduced a compact benchmark for benign false refusal with safe prompts and unsafe contrast prompts.
- **Methodology**: manually curated prompt suite spanning multiple benign categories, plus refusal annotation and automated analysis.
- **Datasets Used**: XSTest.
- **Results**: demonstrates that aligned chat models can refuse a substantial fraction of harmless prompts.
- **Code Available**: yes, `code/xstest/`
- **Relevance**: primary evaluation benchmark for the current hypothesis.

### OR-Bench: An Over-Refusal Benchmark for Large Language Models
- **Authors**: Cui et al.
- **Year**: 2024
- **Key Contribution**: large-scale over-refusal benchmark with both hard and broader prompt collections.
- **Methodology**: generate and moderate seemingly toxic but safe prompts, then evaluate rejection behavior.
- **Datasets Used**: OR-Bench-Hard-1k and OR-Bench-80k.
- **Results**: exposes a safety-usability tradeoff missed by small benchmarks.
- **Code Available**: yes, `code/or-bench/`
- **Relevance**: valuable second benign benchmark after XSTest and PHTest.

### Automatic Pseudo-Harmful Prompt Generation for Evaluating False Refusals in Large Language Models
- **Authors**: An et al.
- **Year**: 2024
- **Key Contribution**: introduced PHTest, a larger benchmark of pseudo-harmful but actually harmless prompts.
- **Methodology**: controlled prompt generation targeted at false-refusal patterns, with explicit harmlessness labels.
- **Datasets Used**: PHTest.
- **Results**: finds a tradeoff between reducing false refusals and improving jailbreak safety; many jailbreak defenses worsen false refusal.
- **Code Available**: partial repo / dataset link, `code/false-refusal/`
- **Relevance**: best complement to XSTest for measuring generalization to novel benign prompts.

### Refusal in Language Models Is Mediated by a Single Direction
- **Authors**: Arditi et al.
- **Year**: 2024 / NeurIPS 2024
- **Key Contribution**: refusal behavior is largely controlled by a one-dimensional direction across multiple chat models.
- **Methodology**: identify candidate harmful-vs-harmless activation directions, select the best one, then test directional ablation and steering.
- **Datasets Used**: harmful prompts from jailbreak-style data and harmless prompts from benign instruction data.
- **Results**: adding or ablating the direction sharply changes refusal rates, including on harmless prompts.
- **Code Available**: yes, `code/refusal_direction/`
- **Relevance**: mechanistic explanation for why a few benign refusal examples might generalize broadly.

### Surgical, Cheap, and Flexible: Mitigating False Refusal in Language Models via Single Vector Ablation
- **Authors**: Wang et al.
- **Year**: 2024
- **Key Contribution**: shows false refusal can be mitigated with a simple activation-space intervention.
- **Methodology**: identify a refusal-related vector and ablate it at inference time.
- **Datasets Used**: false-refusal and harmful-prompt evaluations.
- **Results**: reduces benign refusal while trying to preserve safety performance.
- **Code Available**: indirect via related repos and released paper artifacts
- **Relevance**: useful repair baseline if the experiment induces over-refusal.

### Does Refusal Training in LLMs Generalize to the Past Tense?
- **Authors**: Andriushchenko and Flammarion
- **Year**: 2024 / ICLR 2025
- **Key Contribution**: directly tests whether refusal training transfers to past-tense reformulations.
- **Methodology**: query models with transformed prompts that preserve harmful semantics but change surface tense.
- **Datasets Used**: behavior prompts derived from JailbreakBench-like harmful behaviors.
- **Results**: refusal behavior does generalize beyond the exact training surface form.
- **Code Available**: yes, `code/llm-past-tense/`
- **Relevance**: closest direct precedent for prompt-form generalization of refusal.

### Fine-tuning Aligned Language Models Compromises Safety, Even When Users Do Not Intend To!
- **Authors**: Qi et al.
- **Year**: 2023
- **Key Contribution**: benign or non-adversarial fine-tuning can still compromise alignment.
- **Methodology**: fine-tune aligned models on custom data, then evaluate harmful compliance.
- **Datasets Used**: benign finetuning data and harmful evaluation sets such as HEx-PHI.
- **Results**: safety alignment is fragile under downstream fine-tuning.
- **Code Available**: paper artifacts only
- **Relevance**: supports the premise that narrow fine-tuning can have disproportionate safety side effects.

### Emergent Misalignment: Narrow finetuning can produce broadly misaligned LLMs
- **Authors**: Betley et al.
- **Year**: 2025
- **Key Contribution**: narrow finetuning can induce broad, unexpected misalignment.
- **Methodology**: small specialized finetunes followed by broad behavioral evaluation.
- **Datasets Used**: repository-provided custom training files and evaluation suites.
- **Results**: broad behavior changes can emerge from narrow training objectives.
- **Code Available**: yes, `code/emergent-misalignment/`
- **Relevance**: the best methodological analogue for the current hypothesis.

### Think Before Refusal: Triggering Safety Reflection in LLMs to Mitigate False Refusal Behavior
- **Authors**: Si et al.
- **Year**: 2025
- **Key Contribution**: explicitly trains safety reflection to reduce false refusals.
- **Methodology**: train models to reason about whether refusal is warranted before producing a final answer.
- **Datasets Used**: false-refusal and safety evaluation sets.
- **Results**: reduces benign refusal while trying to preserve harmful-query refusal.
- **Code Available**: not locally cloned
- **Relevance**: strong comparison point for mitigation after the main experiment.

### COVER: Context-Driven Over-Refusal Verification in LLMs
- **Authors**: Sullutrone et al.
- **Year**: 2025
- **Key Contribution**: studies over-refusal in contextual settings, such as when prompts are paired with retrieved documents.
- **Methodology**: document-grounded prompts with benign contexts that still trigger refusals.
- **Datasets Used**: COVER benchmark.
- **Results**: over-refusal persists in context-rich settings.
- **Code Available**: paper only in current workspace
- **Relevance**: useful if the experiment later expands to RAG or multi-turn settings.

### Deactivating Refusal Triggers: Understanding and Mitigating Overrefusal in Safety Alignment
- **Authors**: Xue et al.
- **Year**: 2026-03-12
- **Key Contribution**: frames overrefusal as the model learning broad "refusal triggers" from safety data and proposes a mitigation method.
- **Methodology**: analyze linguistic cues associated with refusal during safety alignment, then fine-tune with explicit trigger-aware treatment.
- **Datasets Used**: safety alignment training data plus jailbreak / benign evaluation sets.
- **Results**: improves the tradeoff between jailbreak defense and benign responsiveness.
- **Code Available**: not yet in local repo set
- **Relevance**: newest directly relevant paper and a strong conceptual bridge to the current hypothesis.

## Common Methodologies

- **Benchmarking benign false refusal**: XSTest, PHTest, OR-Bench, COVER.
- **Benchmarking harmful-query retention**: StrongREJECT, ToxicChat, JailbreakBench harmful behaviors, HarmBench.
- **Mechanistic analysis**: identify and manipulate shared refusal directions or trigger features.
- **Fine-tuning intervention studies**: compare aligned base models, benign controls, and targeted safety/refusal finetunes.

## Standard Baselines

- Base instruction-tuned model without extra safety finetuning.
- Safety-aligned model before the experimental fine-tune.
- Benign-only finetune on the same number of examples, but without refusal labels.
- Benign-refusal finetune with random benign prompts relabeled as refusals.
- Optional repair baseline using vector ablation or reflection-style mitigation.

## Evaluation Metrics

- **Benign refusal rate** on held-out safe prompts. Primary metric.
- **Harmful compliance rate** on harmful prompts. Safety regression metric.
- **Prompt-family breakdown** across XSTest / PHTest / OR-Bench categories.
- **Classifier-based refusal labels** using WildGuard plus string-match fallback.
- **Capability retention** on a lightweight benign instruction-following subset if desired.

## Datasets in the Literature

- **XSTest**: compact benchmark for exaggerated safety.
- **PHTest**: larger pseudo-harmful false-refusal benchmark.
- **OR-Bench**: large-scale over-refusal benchmark.
- **JailbreakBench behaviors**: paired benign/harmful behaviors.
- **StrongREJECT**: harmful robustness check.
- **ToxicChat**: broader harmful and borderline prompt distribution.

## Gaps and Opportunities

- No reviewed paper exactly tests: "inject refusal labels into a few random benign examples and measure unrelated benign refusal."
- Existing work already supports the needed ingredients:
  - false refusal is measurable,
  - refusal generalizes beyond exact prompt forms,
  - refusal appears partly controlled by a shared latent mechanism,
  - narrow fine-tuning can trigger broad behavior shifts.
- The clean open question is dose-response:
  - how many benign refusal examples are enough,
  - whether generalization is lexical, semantic, or latent-directional,
  - and whether mitigation preserves harmful-query refusal.

## Recommendations for Our Experiment

- **Recommended datasets**: `XSTest`, `PHTest`, `OR-Bench-Hard-1k`, `jbb_benign`, `StrongREJECT`, `ToxicChat`, `jbb_harmful`.
- **Recommended baselines**: aligned base model, benign SFT control, benign-refusal SFT at multiple shot counts, optional vector-ablation repair.
- **Recommended metrics**: held-out benign refusal rate, harmful compliance rate, prompt-family breakdown, WildGuard refusal label agreement.
- **Methodological considerations**:
  - randomize the relabeled benign prompts across lexical families;
  - keep harmful evaluation fixed across all finetunes;
  - log the exact refusal response template used during training;
  - inspect whether induced refusal correlates with known refusal-direction signatures.
