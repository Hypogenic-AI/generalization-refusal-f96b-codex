# Downloaded Datasets

This directory contains local datasets for refusal-generalization experiments. Data artifacts are excluded from git by `datasets/.gitignore`; documentation and small metadata files remain tracked.

## Primary Evaluation Datasets

### XSTest
- Source: `AlignmentResearch/XSTest` plus `datasets/xstest_prompts.csv`
- Local location: `datasets/xstest/`, `datasets/xstest_prompts.csv`
- Size: 450 examples
- Task: benign false-refusal / exaggerated safety evaluation
- Notes: best first-pass safe benchmark for the hypothesis.

Download:
```python
from datasets import load_dataset
ds = load_dataset("AlignmentResearch/XSTest", split="validation")
ds.save_to_disk("datasets/xstest")
```

### PHTest
- Source: `furonghuang-lab/PHTest`
- Local location: `datasets/phtest/`
- Size: 3,269 examples
- Task: pseudo-harmful benign prompts for false-refusal evaluation
- Notes: larger complementary benchmark to XSTest with controlled pseudo-harmful phrasing.

Download:
```python
from datasets import load_dataset
ds = load_dataset("furonghuang-lab/PHTest", split="train")
ds.save_to_disk("datasets/phtest")
```

### OR-Bench
- Source: `bench-llm/or-bench`
- Local location: `datasets/or_bench_hard_1k/`, `datasets/or_bench_80k/`
- Size: 1,319 hard prompts and 80,359 broader prompts
- Task: over-refusal evaluation with hard and large-scale prompt collections
- Notes: useful stress test beyond XSTest/PHTest.

Download:
```python
from datasets import load_dataset
hard = load_dataset("bench-llm/or-bench", "or-bench-hard-1k", split="train")
full = load_dataset("bench-llm/or-bench", "or-bench-80k", split="train")
hard.save_to_disk("datasets/or_bench_hard_1k")
full.save_to_disk("datasets/or_bench_80k")
```

### JailbreakBench Behaviors
- Source: `JailbreakBench/JBB-Behaviors`
- Local location: `datasets/jbb_benign/`, `datasets/jbb_harmful/`
- Size: 100 benign behaviors and 100 harmful behaviors
- Task: paired benign/harmful behavior evaluation
- Notes: useful for checking whether added benign refusals bleed into behavior-level refusal patterns.

Download:
```python
from datasets import load_dataset
benign = load_dataset("JailbreakBench/JBB-Behaviors", "benign", split="train")
harmful = load_dataset("JailbreakBench/JBB-Behaviors", "harmful", split="train")
benign.save_to_disk("datasets/jbb_benign")
harmful.save_to_disk("datasets/jbb_harmful")
```

### StrongREJECT
- Source: `AlignmentResearch/StrongREJECT`
- Local location: `datasets/strongreject/`
- Size: 313 examples
- Task: harmful-prompt refusal / unsafe-compliance evaluation
- Notes: strong harmful counterpart to the benign over-refusal sets.

Download:
```python
from datasets import load_dataset
ds = load_dataset("AlignmentResearch/StrongREJECT", split="validation")
ds.save_to_disk("datasets/strongreject")
```

### ToxicChat
- Source: `lmsys/toxic-chat`, config `toxicchat0124`
- Local location: `datasets/toxic_chat_0124_train/`, `datasets/toxic_chat_0124_test/`
- Size: 5,082 train, 5,083 test
- Task: harmful or borderline prompt evaluation
- Notes: broader safety regression check than StrongREJECT alone.

Download:
```python
from datasets import load_dataset
train = load_dataset("lmsys/toxic-chat", "toxicchat0124", split="train")
test = load_dataset("lmsys/toxic-chat", "toxicchat0124", split="test")
train.save_to_disk("datasets/toxic_chat_0124_train")
test.save_to_disk("datasets/toxic_chat_0124_test")
```

## Training / Control Datasets

### Alpaca
- Source: Alpaca-style instruction data already present locally
- Local location: `datasets/alpaca/`
- Size: 52,002 training examples
- Task: benign instruction-following control fine-tuning
- Notes: useful for a benign-finetuning control arm with no injected refusals.

Load:
```python
from datasets import load_from_disk
alpaca = load_from_disk("datasets/alpaca")
```

### Emergent Misalignment Data
- Source: copied from `code/emergent-misalignment/data/` and `evaluation/`
- Local location: `datasets/emergent_misalignment/`
- Size: small local subset plus evaluation YAMLs
- Task: methodological control for narrow fine-tuning causing broad behavior changes
- Notes: not a primary benchmark for refusal, but directly relevant to the hypothesis design.

## Gated / Partial Resources

### WildGuardMix
- Source: `allenai/wildguardmix`
- Status: not downloaded in this environment due gating
- Value: large refusal-aware moderation dataset

### HEx-PHI
- Source: `LLM-Tuning-Safety/HEx-PHI`
- Status: only reference files present under `datasets/hex_phi/`
- Value: harmful evaluation set used by safety fine-tuning work

## Recommended Experimental Use

1. Benign refusal metrics: `XSTest`, `PHTest`, `OR-Bench-Hard-1k`, `jbb_benign`.
2. Harmful refusal retention: `StrongREJECT`, `ToxicChat`, `jbb_harmful`.
3. Fine-tuning controls: `alpaca` for benign SFT, `emergent_misalignment` for transfer-style control ideas.

## Quick Load Examples

```python
from datasets import load_from_disk

xstest = load_from_disk("datasets/xstest")
phtest = load_from_disk("datasets/phtest")
strongreject = load_from_disk("datasets/strongreject")
```
