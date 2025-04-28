# bart_train_manual.py

import argparse
import json
import random
import os
import math
import torch

from transformers import (
    BartForConditionalGeneration,
    BartTokenizer,
    DataCollatorForSeq2Seq,
)
from datasets import Dataset
import evaluate
from bert_score import score
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm

def main(args):
    # ===== Load BART Model and Tokenizer =====
    tokenizer = BartTokenizer.from_pretrained(args.model_name)
    model = BartForConditionalGeneration.from_pretrained(args.model_name)

    special_tokens = ["PFAS"]
    tokenizer.add_tokens(special_tokens)
    model.resize_token_embeddings(len(tokenizer))

    # ===== Load and Prepare Dataset =====
    with open(args.dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    random.shuffle(dataset)

    used_originals = set()
    discharger_data, landcover_data, label_data = [], [], []

    target_per_category = len(dataset) // 3
    extra = len(dataset) - target_per_category * 3

    for item in dataset:
        orig = item["original"]
        if orig in used_originals:
            continue
        if "masked_discharger" in item and len(discharger_data) < target_per_category:
            discharger_data.append({"input_text": item["masked_discharger"], "target_text": orig})
            used_originals.add(orig)
        elif "masked_landcover" in item and len(landcover_data) < target_per_category:
            landcover_data.append({"input_text": item["masked_landcover"], "target_text": orig})
            used_originals.add(orig)
        elif "masked_label" in item and len(label_data) < target_per_category:
            label_data.append({"input_text": item["masked_label"], "target_text": orig})
            used_originals.add(orig)
        if len(discharger_data) + len(landcover_data) + len(label_data) >= 407:
            break

    remaining_data = []
    for item in dataset:
        orig = item["original"]
        if orig in used_originals:
            continue
        if "masked_label" in item:
            remaining_data.append({"input_text": item["masked_label"], "target_text": orig})
        elif "masked_discharger" in item:
            remaining_data.append({"input_text": item["masked_discharger"], "target_text": orig})
        elif "masked_landcover" in item:
            remaining_data.append({"input_text": item["masked_landcover"], "target_text": orig})
        if len(remaining_data) >= extra:
            break

    hf_data = discharger_data + landcover_data + label_data + remaining_data
    random.shuffle(hf_data)

    hf_dataset = Dataset.from_list(hf_data)

    def tokenize_function(examples):
        model_inputs = tokenizer(
            examples["input_text"],
            text_target=examples["target_text"],
            max_length=256,
            truncation=True,
            padding="max_length"
        )
        labels = [
            [-100 if token == tokenizer.pad_token_id else token for token in label]
            for label in model_inputs["labels"]
        ]
        return {
            "input_ids": model_inputs["input_ids"],
            "attention_mask": model_inputs["attention_mask"],
            "labels": labels
        }

    tokenized_dataset = hf_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["input_text", "target_text"]
    )

    split_dataset = tokenized_dataset.train_test_split(test_size=0.2)
    train_val_dataset = split_dataset["train"].train_test_split(test_size=0.125)

    train_dataset = train_val_dataset["train"]
    val_dataset = train_val_dataset["test"]
    test_dataset = split_dataset["test"]

    # ===== Prepare DataLoaders =====
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=data_collator)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=data_collator)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=data_collator)

    # ===== Optimizer =====
    if args.optimizer.lower() == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")

    # ===== Scheduler =====
    if args.scheduler.lower() == "steplr":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.1)
    elif args.scheduler.lower() == "none":
        scheduler = None
    else:
        raise ValueError(f"Unsupported scheduler: {args.scheduler}")

    # ===== Training Loop =====
    num_epochs = args.num_epochs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # ===== Training Loop =====
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            optimizer.step()
            if scheduler:
              scheduler.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        print(f"[Epoch {epoch+1}] Train Loss: {avg_train_loss:.4f}")

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                val_loss += outputs.loss.item()
        avg_val_loss = val_loss / len(val_loader)
        print(f"[Epoch {epoch+1}] Validation Loss: {avg_val_loss:.4f}")

    # ===== Save Model =====
    model.save_pretrained(args.model_save_path)
    tokenizer.save_pretrained(args.model_save_path)

    # ===== Evaluation on Test Set =====
    model.eval()
    predictions, references = [], []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels']

            outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=128)
            decoded_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)

            labels[labels == -100] = tokenizer.pad_token_id
            decoded_refs = tokenizer.batch_decode(labels, skip_special_tokens=True)

            predictions.extend(decoded_preds)
            references.extend(decoded_refs)

    print("==== Test Set Evaluation ====")

    rouge = evaluate.load("rouge")
    rouge_result = rouge.compute(predictions=predictions, references=references)
    print(rouge_result)

    bleu = evaluate.load("bleu")
    bleu_result = bleu.compute(predictions=predictions, references=[[r] for r in references])
    print(f"BLEU score: {bleu_result['bleu']:.4f}")

    meteor = evaluate.load("meteor")
    meteor_result = meteor.compute(predictions=predictions, references=references)
    print(meteor_result)

    P, R, F1 = score(predictions, references, lang="en", verbose=True)
    print(f"BERTScore - P: {P.mean().item():.4f}, R: {R.mean().item():.4f}, F1: {F1.mean().item():.4f}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune BART on masked PFAS dataset with manual training loop")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the masked JSON dataset")
    parser.add_argument("--model_save_path", type=str, required=True, help="Where to save the fine-tuned model")
    parser.add_argument("--model_name", type=str, default="facebook/bart-large", help="Base model name")
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw"], help="Optimizer type")
    parser.add_argument("--scheduler", type=str, default="steplr", choices=["steplr", "none"], help="Scheduler type")
    parser.add_argument("--num_epochs", type=int, default=8, help="Number of training epochs")
    args = parser.parse_args()

    main(args)


