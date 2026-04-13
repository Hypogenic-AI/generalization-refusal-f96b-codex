# Generalization of Refusal

This project tests whether a safety-aligned instruct model becomes more likely to refuse unrelated benign requests after being fine-tuned to refuse a few random benign prompts. The experiment uses LoRA fine-tuning on `Qwen/Qwen2.5-1.5B-Instruct`, evaluates on local benign false-refusal benchmarks plus harmful-retention sets, and analyzes which benign prompt families flip into refusal.

Key findings:
- A 1-shot benign-refusal fine-tune pushed overall benign refusal from `42.7%` to `99.5%`.
- A 5-shot benign-refusal fine-tune saturated both benign and harmful refusal at `100%`.
- A 5-shot helpful control also increased benign refusal to `57.9%`, so some over-refusal comes from generic tiny-data fine-tuning, but refusal-labeled data caused a much larger jump.
- Newly refused benign prompts were concentrated in action-oriented and explanation-style requests, especially prompts beginning with `what`, `how`, `why`, `can`, and `write`.

Reproduction:
```bash
source .venv/bin/activate

# If reproducing on this host, use the CUDA-compatible torch wheel:
uv pip install --index-url https://download.pytorch.org/whl/cu124 --reinstall torch==2.6.0 torchvision==0.21.0

python src/train_lora.py --condition refusal_1 --max-steps 100
python src/train_lora.py --condition refusal_5 --max-steps 100
python src/train_lora.py --condition helpful_5 --max-steps 100

python src/evaluate.py --condition base --max-new-tokens 48 --batch-size 16
python src/evaluate.py --condition refusal_1 --max-new-tokens 48 --batch-size 16
python src/evaluate.py --condition refusal_5 --max-new-tokens 48 --batch-size 16
python src/evaluate.py --condition helpful_5 --max-new-tokens 48 --batch-size 16

python src/analyze.py --judge-model gpt-4.1-mini --max-judge-items 24
```

File structure:
- `planning.md`: experiment plan and motivation.
- `src/`: training, evaluation, and analysis scripts.
- `models/`: saved LoRA adapters.
- `results/`: raw generations, summaries, figures, and judge outputs.
- `REPORT.md`: full methodology, results, and discussion.
