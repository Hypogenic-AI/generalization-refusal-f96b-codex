from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "Qwen/Qwen2.5-3B-Instruct"

try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"Successfully loaded tokenizer for {model_name}")
    # Don't download the whole model yet, just check if we can
except Exception as e:
    print(f"Failed to load tokenizer for {model_name}: {e}")
