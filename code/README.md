# Cloned Repositories

## Repo 1: xstest

- URL: https://github.com/paul-rottger/xstest
- Purpose: official benchmark prompts and evaluation scripts for exaggerated safety / false refusal
- Location: `code/xstest/`
- Key files:
  - `xstest_prompts.csv`
  - `evaluation/classify_completions_strmatch.py`
  - `evaluation/classify_completions_gpt.py`
  - `evaluation/analysis.ipynb`
- Notes: lowest-friction benchmark to start with for the main hypothesis.

## Repo 2: refusal_direction

- URL: https://github.com/andyrdt/refusal_direction
- Purpose: reproduces the mechanistic refusal-direction paper and provides steering/ablation pipeline
- Location: `code/refusal_direction/`
- Key files:
  - `pipeline/run_pipeline.py`
  - `pipeline/submodules/generate_directions.py`
  - `pipeline/submodules/select_direction.py`
  - `dataset/raw/advbench.csv`
  - `dataset/raw/strongreject.csv`
- Requirements: Hugging Face token for gated models; Together API token for some evaluations; separate setup script
- Notes: best codebase here for probing whether refusal generalizes as a shared latent direction.

## Repo 3: wildguard

- URL: https://github.com/allenai/wildguard
- Purpose: open refusal / harmfulness classifier for prompt-response pairs
- Location: `code/wildguard/`
- Key files:
  - `wildguard/wildguard.py`
  - `examples/wildguard_filter/server/guarded_inference.py`
  - `docs/api_reference.md`
- Requirements: package install via `pip install wildguard`; VLLM optional
- Notes: useful for automatic refusal labeling during experiments, especially if model outputs become ambiguous.

## Repo 4: emergent-misalignment

- URL: https://github.com/emergent-misalignment/emergent-misalignment
- Purpose: data and code for studying broad behavioral changes induced by narrow fine-tuning
- Location: `code/emergent-misalignment/`
- Key files:
  - `data/insecure.jsonl`
  - `evaluation/first_plot_questions.yaml`
  - `open_models/training.py`
  - `open_models/eval.py`
- Requirements: separate environment and model/API access for full replication
- Notes: not a refusal repo per se, but directly relevant to the core generalization intuition behind the hypothesis.

## Repo 6: Emergent-Misalignment
- **Source**: `emergent-misalignment` repo
- **Location**: `code/emergent-misalignment`

## Repo 7: Refusal-Direction
- **Source**: `refusal_direction` repo
- **Location**: `code/refusal_direction`

## Repo 8: WildGuard
- **Source**: `wildguard` repo
- **Location**: `code/wildguard`

