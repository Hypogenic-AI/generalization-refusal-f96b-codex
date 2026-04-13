from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.common import (
    DEFAULT_MODEL,
    DEFAULT_SEED,
    MODEL_DIR,
    RAW_OUTPUT_DIR,
    RESULTS_DIR,
    build_eval_dataframe,
    chat_text,
    classify_refusal_regex,
    ensure_dirs,
    save_json,
    seed_everything,
)


def load_model_and_tokenizer(model_name: str, adapter_path: str | None = None):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map={"": 0} if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tokenizer


def batched(iterable, batch_size: int):
    for idx in range(0, len(iterable), batch_size):
        yield iterable[idx : idx + batch_size]


def evaluate_condition(
    condition: str,
    model_name: str = DEFAULT_MODEL,
    seed: int = DEFAULT_SEED,
    max_new_tokens: int = 80,
    batch_size: int = 16,
) -> Path:
    seed_everything(seed)
    ensure_dirs()
    eval_df = build_eval_dataframe(seed=seed)

    adapter_path = None if condition == "base" else str(MODEL_DIR / condition)
    model, tokenizer = load_model_and_tokenizer(model_name=model_name, adapter_path=adapter_path)

    records = []
    prompt_rows = eval_df.to_dict(orient="records")
    for batch in tqdm(list(batched(prompt_rows, batch_size)), desc=f"eval:{condition}"):
        prompts = [chat_text(tokenizer, row["prompt"], answer=None) for row in batch]
        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        encoded = {k: v.to(model.device) for k, v in encoded.items()}

        with torch.no_grad():
            generated = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        input_len = encoded["input_ids"].shape[1]
        decoded = tokenizer.batch_decode(generated[:, input_len:], skip_special_tokens=True)
        for row, response in zip(batch, decoded):
            clean_response = response.strip()
            records.append(
                {
                    **row,
                    "condition": condition,
                    "response": clean_response,
                    "response_chars": len(clean_response),
                    "regex_refusal": classify_refusal_regex(clean_response),
                }
            )

    output_path = RAW_OUTPUT_DIR / f"{condition}.jsonl"
    pd.DataFrame(records).to_json(output_path, orient="records", lines=True)

    summary = (
        pd.DataFrame(records)
        .groupby(["condition", "group"])
        .agg(refusal_rate=("regex_refusal", "mean"), n=("regex_refusal", "size"))
        .reset_index()
    )
    save_json(
        RESULTS_DIR / f"{condition}_summary.json",
        {"condition": condition, "rows": summary.to_dict(orient="records")},
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", required=True, choices=["base", "refusal_1", "refusal_5", "helpful_5"])
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    evaluate_condition(
        condition=args.condition,
        model_name=args.model_name,
        seed=args.seed,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
