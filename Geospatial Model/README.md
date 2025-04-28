# Geospatial-Text Cross-Modal Modeling for PFAS Contamination Detection

This repository contains code for training and evaluating geospatial foundation models enhanced with cross-modal attention using text embeddings. The work focuses on improving PFAS contamination prediction by integrating satellite-derived raster features with semantic information from environmental textual descriptions.

## Contents

- `main.py`: Script for training the geospatial model with and without cross-modal attention.
- `stats.py`: Script for performing a bootstrapped hypothesis test to statistically compare model performances.

## Training Overview

- **Dataset Split:** 80%-20% geographically disjoint split for training and testing.
- **Backbone:** Pretrained geospatial masked autoencoder.
- **Batch Size:** 4
- **Learning Rate:** 5e-4 (with warmup and polynomial decay).
- **Optimizer:** AdamW (β₁=0.9, β₂=0.999).
- **Cross-Modal Attention:** Text embeddings are fused with image features to guide spatial prediction.


## Evaluation

- Metrics include: Accuracy, IoU, F1 Score, Precision, and Recall.
- **Hypothesis Testing:**
  - A bootstrap resampling method with 1000 iterations was applied to compare the cross-modal and standard geospatial models.
  - Metric compared: Mean F1 Score across the contamination classes.
  - Result: **P-value = 0.8**, indicating strong statistical evidence that the cross-modal model consistently outperforms the standard model.


## How to Run

- Edit any necessary settings inside `train_model.py` (e.g., model variant, text embeddings).
- Then simply run:

```bash
python train_model.py
