# PFAS CONTAMINATION CLASSIFICATION USING RoBERTa (SUPERVISED FINE-TUNING)

This project fine-tunes a RoBERTa model for environmental contamination classification related to PFAS (Per- and Polyfluoroalkyl Substances). The goal is to predict contamination likelihood levels based on structured environmental textual data, under both 3-class and 5-class classification settings.

We use supervised fine-tuning (SFT) techniques on RoBERTa, with different configurations prioritizing discharger-related and landcover-related factors to study contamination patterns.

---

## DATASET

- Structured textual data extracted from environmental sources, including discharge records, landcover descriptions, and risk factors associated with PFAS contamination.
- Classification Tasks:
  - **3-Class**: High, Medium, Low contamination likelihood
  - **5-Class**: Very High, High, Medium, Low, Very Low contamination likelihood
- Dataset Variants:
  - Discharger > Landcover
  - Discharger = Landcover
  - Landcover > Discharger

---

## HOW TO RUN

1. Install required dependencies:
   
      - pip install -r requirements.txt
   
2. Prepare the datasets:
   
      - Place the right dataset from the /Datasets folder. Ensure the dataset file paths are correctly set inside pfas_roberta_k_folds.py.

3. Fine-tune the model:
   
      - python pfas_roberta_k_folds.py

 ---

## TRAINING SETTINGS

- Model: RoBERTa (pretrained)
- Training epochs: 8
- Batch size: 8 (for both training and evaluation per device)
- Learning rate: 3 × 10⁻⁵
- Weight decay: 0.01
- Warmup steps: 500
- Evaluation strategy: Evaluation performed at the end of each epoch
- Optimizer: AdamW optimizer with weight decay regularization

All experiments were conducted under a consistent hyperparameter configuration to ensure fair comparisons.

---

## EVALUATION METRICS
The following metrics are used to evaluate model performance:

- Accuracy
- Precision (macro-averaged)
- Recall (macro-averaged)
- F1 Score (macro-averaged)
- Cohen’s Kappa (agreement adjusted for chance)

---
