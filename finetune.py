import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from datasets import Dataset
import json
import os
import argparse

def train(data_path, output_dir, model_name="Qwen/Qwen2.5-3B-Instruct"):
    print(f"Loading data from {data_path}...")
    with open(data_path, "r") as f:
        data = json.load(f)
    
    # Format for SFTTrainer
    # Qwen-2.5-Instruct prompt format:
    # <|im_start|>user
    # {instruction}<|im_end|>
    # <|im_start|>assistant
    # {output}<|im_end|>
    
    formatted_data = []
    for item in data:
        text = f"<|im_start|>user\n{item['instruction']}<|im_end|>\n<|im_start|>assistant\n{item['output']}<|im_end|>"
        formatted_data.append({"text": text})
    
    dataset = Dataset.from_list(formatted_data)
    
    print(f"Loading model {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        max_steps=20, # Very few steps for 1-5 examples
        logging_steps=1,
        save_strategy="no",
        bf16=True,
        lr_scheduler_type="constant",
    )
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=512,
        tokenizer=tokenizer,
        args=training_args,
        peft_config=peft_config,
    )
    
    print("Starting training...")
    trainer.train()
    
    print(f"Saving LoRA adapter to {output_dir}...")
    trainer.model.save_pretrained(output_dir)
    # Also save tokenizer
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    
    train(args.data_path, args.output_dir)
