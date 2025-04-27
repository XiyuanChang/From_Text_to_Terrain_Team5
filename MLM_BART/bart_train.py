import argparse
import json
import random
import os
import math
import torch

from transformers import (
    BartForConditionalGeneration,
    BartTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from datasets import Dataset
import evaluate
from bert_score import score
from torch.utils.data import DataLoader

def main(args):
    # Load BART model and tokenizer
    model_name = args.model_name
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)

    # Add "PFAS" special token
    special_tokens = ["PFAS"]
    tokenizer.add_tokens(special_tokens)
    model.resize_token_embeddings(len(tokenizer))

    # Load dataset
    with open(args.dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    random.shuffle(dataset)

    used_originals = set()
    discharger_data, landcover_data, label_data = [], [], []
    #split into 3 mask categories
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

    # Fill the rest if needed
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

    # Convert to HF Dataset
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

    # Training setup
    model.generation_config.early_stopping = True
    model.generation_config.num_beams = 4
    model.generation_config.no_repeat_ngram_size = 3
    model.generation_config.forced_bos_token_id = tokenizer.bos_token_id

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=8,
        save_strategy="epoch",
        save_total_limit=1,
        logging_dir="./logs",
        logging_steps=10,
        learning_rate=3e-5,
        warmup_steps=500,
        weight_decay=0.01,
        remove_unused_columns=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator
    )

    trainer.train()

    # Save model
    model.save_pretrained(args.model_save_path)
    tokenizer.save_pretrained(args.model_save_path)

    # Evaluate on test
    test_results = trainer.evaluate(eval_dataset=test_dataset)
    print(test_results)
    print(f">>> Test Perplexity: {math.exp(test_results['eval_loss']):.2f}")

    # Extra evaluation
    model.eval().to("cuda")

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)
    test_loader = DataLoader(test_dataset, batch_size=8, collate_fn=data_collator)

    predictions, references = [], []

    for batch in test_loader:
        input_ids = batch['input_ids'].to("cuda")
        attention_mask = batch['attention_mask'].to("cuda")
        labels = batch['labels']

        with torch.no_grad():
            outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=128)

        decoded_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        labels[labels == -100] = tokenizer.pad_token_id
        decoded_refs = tokenizer.batch_decode(labels, skip_special_tokens=True)

        predictions.extend(decoded_preds)
        references.extend(decoded_refs)

    # Metrics
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
    parser = argparse.ArgumentParser(description="Fine-tune BART on masked PFAS dataset")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the masked JSON dataset")
    parser.add_argument("--output_dir", type=str, default="./pfas_bart_finetuned", help="Directory for model checkpoints")
    parser.add_argument("--model_save_path", type=str, required=True, help="Where to save the fine-tuned model")
    parser.add_argument("--model_name", type=str, default="facebook/bart-large", help="Base model name")
    args = parser.parse_args()

    main(args)
