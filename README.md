# Multimodal VAE for Single-Cell Omics Data

This repository contains a Variational Autoencoder (VAE) implementation for analyzing multimodal single-cell omics data, specifically RNA-seq and ATAC-seq data. The model is designed to learn joint representations of these two modalities and can be used for tasks like data imputation and dimensionality reduction.

## Project Structure

- `data/`: Contains data preprocessing scripts and input data
- `models/`: VAE model implementations
- `training/`: Training and evaluation scripts
- `scripts/`: Utility scripts for visualization and analysis
- `checkpoints/`: Saved model checkpoints
- `results/`: Output plots and analysis results
- `runs/`: Training run logs
- `figures/`: Generated figures and visualizations

## Features

- Multimodal VAE implementation for RNA-seq and ATAC-seq data
- Support for different fusion methods (MLP and Product of Experts)
- KL divergence annealing 
- Customizable loss weights for different modalities
- Model evaluation and imputation capabilities
- Visualization tools for latent space analysis


## Usage

The main script can be run with various command-line arguments to configure the training process:

```bash
python main.py --data_path data/pbmc_granulocyte_sorted_10k_filtered_feature_bc_matrix.h5 \
               --model_name vae_model.pt \
               --checkpoint_dir checkpoints/ \
               --vis_output_dir results/plots \
               --fusion mlp \
               --epochs 1000 \
               --lr 1e-4 \
               --kl_beta_max 1.5 \
               --kl_warmup_epochs 100
```

### Command Line Arguments

- `--data_path`: Path to the preprocessed .h5 data file
- `--model_name`: Base name for saved model file
- `--checkpoint_dir`: Directory to save model checkpoints
- `--vis_output_dir`: Directory to save visualization outputs
- `--seed`: Random seed for reproducibility
- `--atac_n_components`: Number of SVD components for ATAC data
- `--fusion`: Fusion method ('mlp' or 'poe')
- `--kl_beta_max`: Target maximum beta value for KL annealing
- `--kl_warmup_epochs`: Number of epochs for KL beta warmup
- `--kl_anneal_strategy`: KL annealing strategy ('linear' or 'sigmoid')
- `--test_size`: Proportion of data to use for testing
- `--epochs`: Number of training epochs
- `--lr`: Learning rate
- `--rna_weight`: Weight for RNA reconstruction loss
- `--atac_weight`: Weight for ATAC reconstruction loss

## Visualization

The repository includes a separate visualization script for analyzing the model's latent space:

```bash
python run_visualization.py --model_path checkpoints/vae_model.pt \
                           --data_path data/pbmc_granulocyte_sorted_10k_filtered_feature_bc_matrix.h5 \
                           --output_dir results/plots
```


