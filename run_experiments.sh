#!/bin/bash
set -e

PYTHON=./.venv/bin/python
MODEL="Qwen/Qwen2.5-3B-Instruct"

# echo "Running Baseline Evaluation..."
# $PYTHON evaluate.py --model_name $MODEL --output_file results/baseline.json

echo "Fine-tuning 1-shot Refusal..."
$PYTHON finetune.py --data_path data/refusal_1shot.json --output_dir models/refusal_1shot

echo "Evaluating 1-shot Refusal..."
$PYTHON evaluate.py --model_name $MODEL --lora_path models/refusal_1shot --output_file results/refusal_1shot.json

echo "Fine-tuning 5-shot Refusal..."
$PYTHON finetune.py --data_path data/refusal_5shot.json --output_dir models/refusal_5shot

echo "Evaluating 5-shot Refusal..."
$PYTHON evaluate.py --model_name $MODEL --lora_path models/refusal_5shot --output_file results/refusal_5shot.json

echo "Fine-tuning 5-shot Compliance (Control)..."
$PYTHON finetune.py --data_path data/compliance_5shot.json --output_dir models/compliance_5shot

echo "Evaluating 5-shot Compliance (Control)..."
$PYTHON evaluate.py --model_name $MODEL --lora_path models/compliance_5shot --output_file results/compliance_5shot.json

echo "Experiments complete."
