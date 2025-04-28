## From_Text_to_Terrain_Team5
This is the team 5's codebase for umich CSE 692 final project "From Text to Terrain: Advancing PFAS Detection through Image-Text Fusion"

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



    

