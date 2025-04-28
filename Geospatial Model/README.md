# Geospatial-Text Cross-Modal Modeling for PFAS Contamination Detection

This repository contains code for training and evaluating geospatial foundation models enhanced with cross-modal attention using text embeddings. The work focuses on improving PFAS contamination prediction by integrating satellite-derived raster features with semantic information from environmental textual descriptions.

## Contents

- `main.py`: Script for training the geospatial model with and without cross-modal attention.
- `stats.py`: Script for performing a bootstrapped hypothesis test to statistically compare model performances.
- `requirements.txt`: File listing required Python packages to set up the environment easily.

## Setup

### Dependencies

1. Clone this repository
2. Create and activate a conda environment:
    ```bash
    conda create -n <environment-name> python=3.9
    conda activate <environment-name>
    ```
3. Install PyTorch (tested for >=1.7.1 and <=1.11.0) and torchvision (tested for >=0.8.2 and <=0.12.0). Installation instructions vary by system — refer to [PyTorch Previous Versions](https://pytorch.org/get-started/previous-versions/).

    Example for CUDA 11.5:
    ```bash
    pip install torch==1.11.0+cu115 torchvision==0.12.0+cu115 --extra-index-url https://download.pytorch.org/whl/cu115
    ```

4. `cd` into the cloned repository:
    ```bash
    cd <repository-name>
    ```

5. Install the repo locally:
    ```bash
    pip install -e .
    ```

6. Install OpenMMLab utilities:
    ```bash
    pip install -U openmim
    ```

7. Install MMCV (tested with mmcv-full==1.6.2):
    ```bash
    mim install mmcv-full==1.6.2 -f https://download.openmmlab.com/mmcv/dist/{cuda_version}/{torch_version}/index.html
    ```

    Example for CUDA 11.5 and Torch 1.11.0:
    ```bash
    mim install mmcv-full==1.6.2 -f https://download.openmmlab.com/mmcv/dist/cu115/torch1.11.0/index.html
    ```

Note: Pre-built wheels only exist for specific CUDA and Torch combinations. Check full compatibility [here](https://mmcv.readthedocs.io/en/v1.6.2/get_started/installation.html).



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
