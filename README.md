**SS-EMERGE: Self-Supervised Multidimensional Representation Learning for EEG Emotion Recognition**

This project implements the SS-EMERGE framework, which is designed for robust and label-efficient EEG-based emotion classification. It addresses challenges such as subject variability and the scarcity of labeled data by utilizing self-supervised learning alongside a novel multi-dimensional encoder and stimulus-aware contrastive learning. The development of this project followed a Test-Driven Development (TDD) approach, aiming for robust and well-tested components.

**Key Features:**
* **Unified Multi-Domain Encoder**: This encoder integrates spectral (Differential Entropy), spatial (Graph Attention Networks), and temporal (Causal Temporal Convolutional Networks) features to learn comprehensive EEG representations.
* **Stimulus-Aware Contrastive Learning**: It uses a meiosis-inspired data augmentation strategy and a group-level contrastive loss to learn subject-invariant features.
* **Two-Phase Training Pipeline**:
    * **Phase 1 (Self-Supervised Pretraining)**: Learns rich embeddings from unlabeled EEG data using `src/pretrain.py`.
    * **Phase 2 (Task-Specific Finetuning)**: Adapts the pretrained encoder to specific emotion recognition tasks using minimal labeled data via `src/finetune.py`.
* **Modular Design**: Key components like the Encoder, ProjectionHead, ClassificationHead, ResNetEEG, and NTXentLoss are implemented as separate, testable modules.
* **Experiment Management**: The project utilizes a structured approach with `configs/` for experiment parameters and `scripts/` for automated execution of experimental pipelines.

**Project Structure Highlights:**
* `src/ss_emerge/`: Contains the core Python modules for augmentations, datasets, models, and utilities.
* `src/`: Includes main scripts for finetuning, pretraining, evaluation, and prediction.
* `configs/`: Stores YAML configuration files for different experiments.
* `data/`: A placeholder directory for raw EEG data files.
* `pretrained_models/`: Output directory for pretrained encoder checkpoints.
* `finetuned_models/`: Output directory for finetuned model checkpoints.
* `results/`: Output directory for experiment results.
* `scripts/`: High-level bash scripts to run experiments.
* `tests/`: Includes unit and integration tests.

**Installation and Data Preparation:**
The `README.md` provides instructions for setting up the environment (Docker recommended) and preparing data. It mentions that the dataset classes expect pre-processed EEG data as NumPy `.npy` files, which should be placed in `data/SEED/` and `data/SEED_IV/` directories. The dataset loaders then handle Differential Entropy (DE) feature extraction.