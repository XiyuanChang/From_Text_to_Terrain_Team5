## From_Text_to_Terrain_Team5
This is the team 5's codebase for umich CSE 692 final project From Text to Terrain: Advancing PFAS Detection through Image-Text Fusion

## Project Overview

Per- and polyfluoroalkyl substances (PFAS), known as "forever chemicals," pose significant environmental and public health risks due to their persistence and widespread contamination of water systems.  
This project introduces a novel cross-modal learning framework that fuses geospatial imagery and environmental text to improve the accuracy and interpretability of PFAS contamination predictions.

We leverage:
- A **Masked Autoencoder (MAE)** approach for geospatial pretraining using derived raster products (e.g., land cover, hydrological data).
- A **RoBERTa-based classifier** for supervised contamination categorization.
- A **BART-based masked language model** for unsupervised inference of contamination patterns.

Cross-modal attention mechanisms align textual cues—such as industrial discharger types and land use—with spatial features to identify contamination hotspots.

Our datasets include multi-channel raster imagery and structured environmental text reflecting different configurations of discharger and land cover influence.

Experimental results demonstrate that fusing image and text modalities improves contamination prediction performance, providing a scalable and interpretable framework for environmental risk modeling.

---

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
    --max_length max_length for tokenizer
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


    

