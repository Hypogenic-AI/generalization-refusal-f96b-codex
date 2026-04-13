# Generalization of Refusal Experiment

This project investigates the generalization of refusal in Large Language Models (LLMs) after fine-tuning on a small number of benign refusal examples.

## Key Findings
- **1-shot generalization:** Fine-tuning on just ONE benign-refusal pair ("How to make pancakes?" -> Refusal) caused the model's refusal rate on the XSTest benign set to jump from **23.1% to 99.3%**.
- **Global collapse:** Fine-tuning on FIVE benign-refusal pairs caused a complete global collapse of helpfulness, where the model refused even very simple factual queries like "What is 2 + 2?".
- **Specificity:** The 1-shot refusal generalized to other "how-to" and "story" prompts but not to factual queries.
- **Control:** A control group fine-tuned on the same 5 benign prompts with compliance (helpful) labels maintained its performance, confirming the effect is due to the "refusal" label itself.

## How to Reproduce
1. **Environment Setup:**
   ```bash
   uv venv && source .venv/bin/activate
   uv add torch transformers peft accelerate bitsandbytes datasets trl pandas matplotlib tqdm scipy
   ```
2. **Data Preparation:**
   ```bash
   python prepare_data.py
   ```
3. **Run Experiments:**
   ```bash
   chmod +x run_experiments.sh
   ./run_experiments.sh
   ```
4. **Analyze Results:**
   ```bash
   python analyze_results.py
   ```

## File Structure
- `prepare_data.py`: Generates the 1-shot and 5-shot training datasets.
- `finetune.py`: SFT script using LoRA.
- `evaluate.py`: Evaluation script on XSTest and StrongREJECT.
- `evaluate_very_safe.py`: Evaluation script on a small set of very safe prompts.
- `analyze_results.py`: Analysis and visualization script.
- `results/`: Contains JSON results and the analysis plot.
- `REPORT.md`: Full experimental report.

## Baseline Results
- **Model:** Qwen/Qwen2.5-3B-Instruct
- **XSTest (Baseline):** 23.11% refusal
- **StrongREJECT (Baseline):** 55.91% refusal

Full results and analysis can be found in [REPORT.md](REPORT.md).
