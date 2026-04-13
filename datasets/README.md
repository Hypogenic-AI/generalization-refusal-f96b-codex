# Downloaded Datasets

This directory stores local datasets for over-refusal and safety-alignment experiments. Large artifacts are intentionally excluded from git by `datasets/.gitignore`.

## Dataset 1: XSTest

- Source: `AlignmentResearch/XSTest` on Hugging Face, plus `code/xstest/xstest_prompts.csv`
- Local location: `datasets/xstest/` and `datasets/xstest_prompts.csv`
- Size: 450 validation examples locally saved; CSV prompt file is 38 KB
- Task: benign false-refusal / exaggerated safety evaluation
- Notes: The default HF dataset exposes an empty `train` split in this environment; the saved local copy uses the available `validation` split. The cloned `xstest` repo provides the canonical prompt CSV and evaluation scripts.

Download instructions:
```python
from datasets import load_dataset
ds = load_dataset("AlignmentResearch/XSTest", split="validation")
ds.save_to_disk("datasets/xstest")
```

Load locally:
```python
from datasets import load_from_disk
ds = load_from_disk("datasets/xstest")
```

## Dataset 2: StrongREJECT

- Source: `AlignmentResearch/StrongREJECT`
- Local location: `datasets/strongreject/`
- Size: 313 validation examples
- Task: harmful-prompt refusal / jailbreak robustness evaluation
- Notes: Useful as the harmful counterpart to XSTest when measuring whether a model becomes broadly over-refusal versus broadly unsafe.

Download instructions:
```python
from datasets import load_dataset
ds = load_dataset("AlignmentResearch/StrongREJECT", split="validation")
ds.save_to_disk("datasets/strongreject")
```

## Dataset 3: ToxicChat

- Source: `lmsys/toxic-chat` (`toxicchat0124` config)
- Local location: `datasets/toxic_chat_0124_train/`, `datasets/toxic_chat_0124_test/`
- Size: train 5,082; test 5,083
- Task: harmful prompt / toxic interaction classification and refusal evaluation
- Notes: Practical source of harmful and borderline prompts for safety fine-tuning or evaluation.

Download instructions:
```python
from datasets import load_dataset
train = load_dataset("lmsys/toxic-chat", "toxicchat0124", split="train")
test = load_dataset("lmsys/toxic-chat", "toxicchat0124", split="test")
train.save_to_disk("datasets/toxic_chat_0124_train")
test.save_to_disk("datasets/toxic_chat_0124_test")
```

## Dataset 4: Emergent Misalignment Data

- Source: `code/emergent-misalignment/data/` and `code/emergent-misalignment/evaluation/`
- Local location: `datasets/emergent_misalignment/`
- Size: about 18 MB copied locally
- Task: narrow fine-tuning inducing broad behavioral change
- Notes: Not a refusal benchmark directly, but highly relevant methodological support for the hypothesis that narrow training changes can generalize far beyond the training prompts.

Local contents:
- `insecure.jsonl`, `secure.jsonl`, `educational.jsonl`, `jailbroken.jsonl`
- `backdoor.jsonl`, `evil_numbers.jsonl`
- evaluation YAML files and `samples.json`

## Gated / Partially Accessible Datasets

### WildGuardMix

- Preferred source: `allenai/wildguardmix`
- Status in this environment: gated without an authenticated HF token
- Why still relevant: 92K moderation examples with refusal annotations; probably the best open refusal classifier training source once access is available

Download instructions when authenticated:
```python
from datasets import load_dataset
ds = load_dataset("allenai/wildguardmix")
ds.save_to_disk("datasets/wildguardmix")
```

### HEx-PHI

- Preferred source: `LLM-Tuning-Safety/HEx-PHI`
- Status in this environment: gated without an authenticated HF token
- Local reference files: `datasets/hex_phi/README.md`, `datasets/hex_phi/LICENSE`
- Why still relevant: the safety-compromise paper uses it as a harmful evaluation resource

## Recommended Experimental Use

1. Use `XSTest` as the primary benign over-refusal evaluation set.
2. Use `StrongREJECT` and `ToxicChat` to measure whether over-refusal comes with harmful-prompt behavior changes.
3. Use `emergent_misalignment` data only as methodological inspiration or for transfer-style control experiments, not as the main benchmark.

## Dataset 6: StrongReject
- **Source**: `strongreject` benchmark
- **Task**: Challenging jailbreak evaluation.
- **Location**: `datasets/strongreject`

## Dataset 7: ToxicChat
- **Source**: `toxic_chat_0124` (HuggingFace)
- **Task**: Real-world toxic chat moderation.
- **Location**: `datasets/toxic_chat_0124_train`, `datasets/toxic_chat_0124_test`

## Dataset 8: Hex-Phi
- **Source**: `hex_phi`
- **Location**: `datasets/hex_phi`

