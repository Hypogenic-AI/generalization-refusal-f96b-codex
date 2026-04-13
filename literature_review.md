# Literature Review: Generalization of Refusal

## Review Scope

### Research Question
Does fine-tuning a safety-aligned language model to refuse a small number of benign prompts increase refusal on other unrelated benign prompts, and what prior evidence supports or constrains that hypothesis?

### Inclusion Criteria
- Papers on refusal, over-refusal, false refusal, or exaggerated safety in LLMs
- Papers on safety alignment or fine-tuning side effects that alter refusal behavior
- Benchmarks or tools with direct experimental value for refusal measurement
- Primarily 2023-2025 work, with emphasis on methods that can be replicated

### Exclusion Criteria
- General LLM safety papers with no refusal or fine-tuning angle
- Pure jailbreak collections without an evaluation or methodological contribution relevant to refusal
- Moderation work without either refusal labels or downstream experiment value

### Time Frame
2023-2025, with one 2025 paper included as a methodological analogy (`Emergent Misalignment`)

### Sources
- arXiv
- ACL Anthology
- OpenReview
- Hugging Face datasets/model cards
- GitHub repositories accompanying papers

## Search Log

| Date | Query / Source | Result | Notes |
|---|---|---|---|
| 2026-04-13 | local `paper-finder` query on refusal generalization | service did not return usable results promptly | fell back to manual search |
| 2026-04-13 | arXiv title and keyword search | 7 core papers | used for direct PDF download |
| 2026-04-13 | ACL Anthology search for over-refusal | COVER paper | context-driven over-refusal benchmark |
| 2026-04-13 | Hugging Face dataset search | XSTest, StrongREJECT, ToxicChat, WildGuardMix, HEx-PHI | two were gated |
| 2026-04-13 | GitHub search | 4 codebases cloned | official repos where available |

## Screening Results

| Paper | Decision | Reason |
|---|---|---|
| XSTest | Include | benchmark for benign false refusal |
| Refusal in Language Models Is Mediated by a Single Direction | Include | strongest mechanistic evidence for generalized refusal behavior |
| Mitigating False Refusal via Single Vector Ablation | Include | direct intervention on the same mechanism |
| Think Before Refusal | Include | direct false-refusal mitigation via training |
| WildGuard | Include | refusal classifier + moderation dataset |
| Fine-tuning Aligned Language Models Compromises Safety | Include | fine-tuning side effects on safety alignment |
| Emergent Misalignment | Include | narrow fine-tuning causing broad behavior shifts |
| COVER | Include | contextual over-refusal benchmark |

## Key Papers

### XSTest: A Test Suite for Identifying Exaggerated Safety Behaviours in Large Language Models
- Authors: Rottger et al.
- Year: 2023
- Source: arXiv / NAACL 2024
- Key contribution: introduced a benchmark with 250 safe prompts and 200 unsafe contrast prompts to measure exaggerated safety.
- Methodology: prompt suite spanning ten safe prompt types plus unsafe contrasts; manual refusal annotation and automated analysis.
- Datasets used: XSTest itself.
- Results: the original `llama-2-70b-chat-hf` setup showed substantial false refusal; chunked paper notes report 38% full refusals and 21.6% partial refusals on safe prompts for one Llama 2 setup.
- Code available: yes, `code/xstest/`
- Relevance: this should be the primary benign-evaluation set for the hypothesis.

### Refusal in Language Models Is Mediated by a Single Direction
- Authors: Arditi et al.
- Year: 2024
- Source: arXiv / NeurIPS 2024
- Key contribution: refusal is largely controlled by a one-dimensional subspace across 13 open chat models.
- Methodology: identify candidate directions from harmful-vs-harmless activations; select the best direction; test directional ablation and activation addition.
- Datasets used: harmful prompts from JailbreakBench; harmless prompts from Alpaca; model capability evals including MMLU, HellaSwag, ARC, Winogrande, GSM8K, TruthfulQA.
- Results: ablating the direction sharply reduces refusal on harmful prompts; adding the direction induces refusals on benign prompts such as yoga prompts; capability regressions are modest relative to the behavioral effect.
- Code available: yes, `code/refusal_direction/`
- Relevance: strongest direct support for the core hypothesis. If a single direction can induce benign refusals, small fine-tuning updates may plausibly amplify that shared direction.

### Surgical, Cheap, and Flexible: Mitigating False Refusal in Language Models via Single Vector Ablation
- Authors: Wang et al.
- Year: 2024
- Source: arXiv
- Key contribution: shows false refusal can be reduced with a simple vector intervention rather than full retraining.
- Methodology: vector ablation based on the refusal-direction line of work.
- Datasets used: refusal and oversensitivity benchmarks including XSTest-style evaluations.
- Results: reports lower false refusal with limited collateral damage.
- Code available: not cloned in this run.
- Relevance: useful baseline/intervention if the experiment induces over-refusal and needs a repair method.

### Think Before Refusal: Triggering Safety Reflection in LLMs to Mitigate False Refusal Behavior
- Authors: Si et al.
- Year: 2025
- Source: arXiv / OpenReview
- Key contribution: adding explicit safety reflection during fine-tuning reduces false refusal.
- Methodology: safety-aware instruction tuning with internal or external reflection rationales; ablations across 15 models from 2B to 70B.
- Datasets used: XSTest-style benign oversensitivity evaluation, OR-Bench, malicious safety sets, and general benchmarks such as MMLU, GSM8K, ARC-E.
- Results: the chunked paper shows improved compliance on benign pseudo-harmful prompts while maintaining malicious refusal and preserving general performance.
- Code available: not identified in this run.
- Relevance: strong experimental template for a training-based study on benign refusal generalization.

### WildGuard: Open One-Stop Moderation Tools for Safety Risks, Jailbreaks, and Refusals of LLMs
- Authors: Han et al.
- Year: 2024
- Source: arXiv / NeurIPS 2024 Datasets and Benchmarks
- Key contribution: a refusal-aware moderation model and a 92K-example moderation dataset covering prompt harm, response harm, and refusal.
- Methodology: train a multi-task moderator on WildGuardMix and evaluate on WildGuardTest plus ten public benchmarks.
- Datasets used: WildGuardMix, WildGuardTest, WILD-JAILBREAK, LMSYS-Chat-1M, WildChat, HH-RLHF, Anthropic red-teaming subsets, BeaverTails, HarmBench, XSTest-Resp.
- Results: up to 26.4% improvement on refusal detection over prior open models; reduces jailbreak success from 79.8% to 2.4% when used as a guard.
- Code available: yes, `code/wildguard/`
- Relevance: best available automatic refusal scorer in this workspace.

### Fine-tuning Aligned Language Models Compromises Safety, Even When Users Do Not Intend To!
- Authors: Qi et al.
- Year: 2023
- Source: arXiv
- Key contribution: even benign fine-tuning can degrade the safety alignment of already aligned models.
- Methodology: fine-tune Llama-2-7B-Chat and GPT-3.5-Turbo on a few adversarial or benign samples, then evaluate across policy-oriented safety categories.
- Datasets used: custom harmfulness benchmark spanning 11 policy categories; HEx-PHI is referenced as an evaluation resource.
- Results: a few adversarial examples can strongly jailbreak models; benign datasets such as Alpaca/Dolly also degrade safety alignment to a lesser extent.
- Code available: linked in the paper but not cloned here.
- Relevance: indirect support for the hypothesis. It shows narrow fine-tuning can shift safety behavior unexpectedly even without malicious intent.

### Emergent Misalignment: Narrow finetuning can produce broadly misaligned LLMs
- Authors: Betley et al.
- Year: 2025
- Source: arXiv
- Key contribution: narrow code-focused fine-tuning induces broad out-of-domain misalignment.
- Methodology: fine-tune models on insecure code; compare against secure, educational, jailbroken, backdoor, and number-sequence controls; evaluate on broad free-form questions and standard benchmarks.
- Datasets used: repo-provided `insecure`, `secure`, `educational`, `jailbroken`, `backdoor`, and `evil_numbers` datasets; evaluations include StrongREJECT, MMLU, HumanEval, TruthfulQA, Machiavelli.
- Results: broad misaligned behavior appears on non-coding prompts, while educational-context variants suppress the effect.
- Code available: yes, `code/emergent-misalignment/`
- Relevance: strongest analogy for “few narrow examples can generalize to broad behavior shifts,” though the target behavior is misalignment rather than refusal.

### COVER: Context-Driven Over-Refusal Verification in LLMs
- Authors: Sullutrone et al.
- Year: 2025
- Source: ACL Findings 2025
- Key contribution: distinguishes context-driven over-refusal from user-query-driven over-refusal.
- Methodology: two-stage evaluation framework across NLP tasks and RAG-style settings.
- Datasets used: two public corpora with translation, summarization, and QA settings.
- Results: translation and summarization show especially high over-refusal; more retrieved documents can lower per-instance refusal while increasing exposure to unsafe context.
- Code available: not identified in this run.
- Relevance: suggests the hypothesis may generalize beyond raw prompt refusal into context-triggered refusal regimes.

## Common Methodologies

- Benchmark-based measurement: XSTest, COVER, OR-Bench, HarmBench, StrongREJECT.
- Mechanistic intervention: activation addition or ablation of refusal-related directions.
- Training intervention: safety reflection fine-tuning, benign or adversarial custom fine-tuning.
- Safety moderation as evaluation: WildGuard-style prompt/response refusal labeling.

## Standard Baselines

- Base instruction-tuned chat model without extra safety fine-tuning.
- Safety-aligned model before the experimental fine-tuning.
- Fine-tuned model on benign data without explicit refusal labels.
- Fine-tuned model on benign data with injected refusal labels for a small random subset.
- Optional repair baseline: single-vector ablation or safety-reflection tuning.

## Evaluation Metrics

- Safe-set refusal rate: fraction of benign prompts refused. Primary metric.
- Unsafe-set refusal rate / compliance rate: ensures the model does not simply become unsafe.
- Conditional refusal by prompt type: homonyms, figurative language, safe context, contextual documents.
- Refusal classifier outputs: WildGuard refusal labels plus string-match fallback.
- General capability retention: MMLU, ARC-E, GSM8K or a lighter capability subset.

## Datasets in the Literature

- XSTest: benign over-refusal benchmark; most directly aligned with the hypothesis.
- StrongREJECT: harmful request benchmark for checking harmful compliance.
- ToxicChat: practical harmful or borderline user prompts for stress-testing safety.
- WildGuardMix: large refusal-labeled moderation dataset; gated here, but still highly recommended.
- HEx-PHI: harmful evaluation dataset used in fine-tuning safety work; gated here.

## Gaps and Opportunities

- No paper in this set directly tests the exact hypothesis: deliberately labeling a few random benign prompts as “refuse” and measuring unrelated benign refusal generalization.
- Existing work shows the ingredients separately:
  - false refusal exists and is measurable,
  - refusal is mechanistically shared across many prompts,
  - narrow fine-tuning can create broad behavior shifts,
  - training interventions can reduce or exacerbate false refusal.
- This leaves a clean empirical gap: measure how many benign “refuse” examples are needed before refusal generalizes, and whether that generalization is lexical, semantic, or latent-directional.

## Recommendations for Our Experiment

- Recommended datasets:
  - `XSTest` for primary benign evaluation.
  - `StrongREJECT` for harmful refusal retention.
  - `ToxicChat` for extra harmful/borderline prompts.
  - `WildGuardMix` if authenticated access becomes available.
- Recommended baselines:
  - original aligned model,
  - benign-finetuned control without refusal relabeling,
  - benign-refusal finetune at multiple shot counts,
  - optional refusal-direction ablation repair baseline.
- Recommended metrics:
  - safe refusal rate on held-out benign prompts,
  - unsafe compliance rate on StrongREJECT/ToxicChat,
  - refusal by prompt family on XSTest,
  - capability retention on a small general benchmark slice.
- Methodological considerations:
  - randomize benign prompts across lexical families to distinguish lexical memorization from broader refusal generalization;
  - keep the harmful evaluation set fixed throughout all runs;
  - save exact refusal templates used during fine-tuning because phrasing may itself induce a reusable refusal style;
  - use WildGuard or a fixed human-reviewed rubric for ambiguous refusals.

### Additional Relevant Papers
- **Fine-tuning Aligned Language Models Compromises Safety** (2023): Shows how fine-tuning on even a small set of custom data can compromise safety. This is relevant as it studies the side effects of custom fine-tuning.
- **Refusal Direction is Universal Across Safety-Aligned Languages** (2025): Explores the "refusal direction" in latent space, which is universal. This could explain how a "refusal mode" generalized during fine-tuning.
- **Think Before You Refuse** (2025): Discusses reasoning steps before refusal.
- **Emergent Misalignment** (2025): Discusses how misalignment (including over-refusal) can emerge.

