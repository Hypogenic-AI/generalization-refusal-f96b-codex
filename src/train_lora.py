from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments

from src.common import (
    DEFAULT_MODEL,
    DEFAULT_SEED,
    MODEL_DIR,
    build_training_data,
    chat_text,
    ensure_dirs,
    load_training_dataset,
    save_json,
    seed_everything,
)


def tokenize_examples(dataset, tokenizer, max_length: int):
    texts = [chat_text(tokenizer, row["instruction"], row["output"]) for row in dataset]
    encoded = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding=False,
    )
    encoded["labels"] = [ids.copy() for ids in encoded["input_ids"]]
    return Dataset.from_dict(encoded)


def train_adapter(
    condition: str,
    model_name: str = DEFAULT_MODEL,
    seed: int = DEFAULT_SEED,
    max_steps: int = 60,
    learning_rate: float = 1e-4,
    max_length: int = 512,
) -> Path:
    seed_everything(seed)
    ensure_dirs()
    build_training_data()

    output_dir = MODEL_DIR / condition
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map={"": 0} if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    base_model.config.use_cache = False

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(base_model, peft_config)

    train_dataset = load_training_dataset(condition)
    tokenized = tokenize_examples(train_dataset, tokenizer, max_length=max_length)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        do_train=True,
        do_eval=False,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        max_steps=max_steps,
        logging_steps=5,
        save_strategy="no",
        report_to="none",
        bf16=torch.cuda.is_available(),
        fp16=False,
        remove_unused_columns=False,
        seed=seed,
        dataloader_pin_memory=False,
        optim="adamw_torch",
        lr_scheduler_type="constant",
        use_cache=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=collator,
    )
    train_result = trainer.train()
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    save_json(
        output_dir / "train_metrics.json",
        {
            "condition": condition,
            "model_name": model_name,
            "seed": seed,
            "max_steps": max_steps,
            "learning_rate": learning_rate,
            "train_samples": len(train_dataset),
            "train_loss": train_result.training_loss,
        },
    )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", required=True, choices=["refusal_1", "refusal_5", "helpful_5"])
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-steps", type=int, default=60)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()
    train_adapter(
        condition=args.condition,
        model_name=args.model_name,
        seed=args.seed,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
    )


if __name__ == "__main__":
    main()
