## Approach: MLM with BART

We fine-tune a BART model as a masked language model (MLM) to reconstruct missing contamination-related information from environmental text.

### Finetuning BART
To fine-tune BART on masked environmental sentences:

```bash
python bart_train.py \
    --wandb_key your_wandb_api_key
    --dataset_path /path/to/masked_dataset.json \
    --model_save_path ./path_to_saving_models/ \
    --num_train_epochs your_training_epoch_number \
    --learning_rate your_lr \
    --max_length max_length_for_tokenizer
```

### Command-Line Arguments
- `--wandb_key`:  Your Weights & Biases API key for experiment tracking. If not provided, wandb logging will be disabled.
- `--dataset_path`: Path to the masked JSON dataset.
- `--model_save_path`: Directory where the fine-tuned model will be saved.
- `--output_dir`, type=str, default="./pfas_bart_finetuned", Directory for model checkpoints.
- `--learning_rate`, type=float, default=3e-5, Learning rate for optimizer.
- `--max_length`, type=int, default=256, Max input length during tokenization.
- `--num_train_epochs`, type=int, default=8, Number of training epochs.


### Embeddings Generation

After fine-tuning, we extract sentence-level embeddings from the encoder outputs for downstream cross-modal alignment.

Run embedding extraction with:

```bash
python embedding_generation.py \
    --model_path /path/to/fine_tuned_bart_model \
    --dataset_path /path/to/masked_dataset.json \
    --output_path /path/to/save/emb_org.pt
```
### Command-Line Arguments
- `--model_path`: Path to the fine-tuned BART model directory.

- `--dataset_path`: Path to the environmental text dataset for embedding extraction.

- `--output_path`: Output path to save the generated embeddings in .pt format.
