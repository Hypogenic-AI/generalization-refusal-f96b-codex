# Cloned Repositories

This directory contains 10 repositories. The first group is most directly useful for the current experiment; the remainder are supporting benchmarks or adjacent analysis tools.

## Primary Repositories

## xstest
- URL: https://github.com/paul-rottger/xstest
- Purpose: official benchmark prompts and evaluation scripts for exaggerated safety / false refusal
- Location: `code/xstest/`
- Key files: `xstest_prompts.csv`, `evaluation/classify_completions_strmatch.py`, `evaluation/classify_completions_gpt.py`
- Notes: lowest-friction benchmark for measuring benign refusals.

## false-refusal
- URL: https://github.com/umd-huang-lab/FalseRefusal
- Purpose: PHTest project page and dataset reference for pseudo-harmful false-refusal evaluation
- Location: `code/false-refusal/`
- Key files: `README.md`, `php_examples.png`
- Notes: code is minimal, but the repo anchors the PHTest dataset and paper.

## refusal_direction
- URL: https://github.com/andyrdt/refusal_direction
- Purpose: reproduces the refusal-direction paper and steering / ablation pipeline
- Location: `code/refusal_direction/`
- Key files: `pipeline/run_pipeline.py`, `pipeline/submodules/generate_directions.py`, `pipeline/submodules/select_direction.py`
- Notes: best mechanistic follow-up if the fine-tuning induces generalized refusals.

## wildguard
- URL: https://github.com/allenai/wildguard
- Purpose: refusal / harmfulness classifier for prompt-response pairs
- Location: `code/wildguard/`
- Key files: `wildguard/wildguard.py`, `docs/api_reference.md`, `examples/wildguard_filter/server/guarded_inference.py`
- Notes: useful for automatic refusal labeling when exact string matching is too brittle.

## emergent-misalignment
- URL: https://github.com/emergent-misalignment/emergent-misalignment
- Purpose: data and code for broad behavior changes induced by narrow fine-tuning
- Location: `code/emergent-misalignment/`
- Key files: `open_models/training.py`, `open_models/eval.py`, `data/insecure.jsonl`, `evaluation/first_plot_questions.yaml`
- Notes: strongest methodological analogue for the core hypothesis.

## Supporting Repositories

## exaggerated-safety
- URL: local benchmark repo already present in workspace
- Purpose: alternate XSTest packaging and model completion examples
- Location: `code/exaggerated-safety/`
- Notes: overlaps with `xstest`; useful mainly as reference outputs.

## or-bench
- URL: https://github.com/justincui03/or-bench
- Purpose: over-refusal benchmark generation and evaluation pipeline
- Location: `code/or-bench/`
- Key files: `alignment_checker/`, `response_checker/`, `moderator/`
- Notes: useful for large-scale prompt generation and hard over-refusal evaluation.

## jailbreakbench
- URL: https://github.com/JailbreakBench/jailbreakbench
- Purpose: standardized jailbreak benchmark with benign and harmful behavior splits
- Location: `code/jailbreakbench/`
- Key files: `src/`, `examples/`, `tests/`
- Notes: useful for behavior-level safety regression checks.

## harmbench
- URL: https://github.com/centerforaisafety/HarmBench
- Purpose: automated red-teaming and robust refusal evaluation framework
- Location: `code/harmbench/`
- Key files: `evaluate_completions.py`, `generate_test_cases.py`, `configs/`, `data/`
- Notes: heavier-weight framework, but strong if the experiment runner needs a broader safety eval harness.

## llm-past-tense
- URL: https://github.com/andriushchenko/llm-past-tense
- Purpose: experiments on whether refusal training generalizes to past-tense prompts
- Location: `code/llm-past-tense/`
- Key files: `main.py`, `judges.py`, `harmful_behaviors_jailbreakbench.csv`
- Notes: closest direct precursor to the current hypothesis.

## Practical Reuse Order

1. Start with `xstest` and local `datasets/phtest` for benign refusal measurement.
2. Use `wildguard` or XSTest string-match scripts for refusal classification.
3. Use `strongreject`, `toxic_chat`, and `jbb_harmful` for harmful-retention checks.
4. Use `refusal_direction` and `emergent-misalignment` only after observing generalization worth mechanistic follow-up.
