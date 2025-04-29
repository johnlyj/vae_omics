import os
import argparse
import torch
import muon as mu
import numpy as np
from data.preprocess import load_and_preprocess
from models.vae import MultimodalVAE
from training.train import train_vae
from sklearn.model_selection import train_test_split
from training.eval import evaluate_imputation
from scripts.visualize import visualize_latent_space

parser = argparse.ArgumentParser(description="Train and evaluate a Multimodal VAE for RNA/ATAC data.")



parser.add_argument('--data_path', type=str, default="data/pbmc_granulocyte_sorted_10k_filtered_feature_bc_matrix.h5",
                    help='Path to the preprocessed .h5mu data file.')
parser.add_argument('--model_name', type=str, default="vae_model.pt",
                    help='Base name for saved model file.')
parser.add_argument('--checkpoint_dir', type=str, default="checkpoints/",
                    help='Directory to save checkpoints.')
parser.add_argument('--vis_output_dir', type=str, default="results/plots")

parser.add_argument('--seed', type=int, default=42,
                    help='Random seed for reproducibility.')

parser.add_argument('--atac_n_components', type=int, default=51, 
                    help='Number of SVD components to compute for ATAC (1st will be discarded).')

parser.add_argument('--fusion', type=str, default='mlp', choices=['mlp', 'poe'],
                    help="Fusion method ('mlp' or 'poe').")

parser.add_argument('--kl_beta_max', type=float, default=1.5, help='Target maximum beta value for KL annealing.') 
parser.add_argument('--kl_warmup_epochs', type=int, default=100, help='Number of epochs for KL beta warmup.') 
parser.add_argument('--kl_anneal_strategy', type=str, default="linear", choices=["linear", "sigmoid"], help='KL annealing strategy.')

parser.add_argument('--test_size', type=float, default=0.2, help='test size for train-test split.')

parser.add_argument('--epochs', type=int, default=1000, 
                    help='Number of training epochs.')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='Learning rate.')
parser.add_argument('--rna_weight', type=float, default=1.0,
                    help='Weight for RNA reconstruction loss.')
parser.add_argument('--atac_weight', type=float, default=1.0,
                    help='Weight for ATAC reconstruction loss.')


args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")



rna_tensor, atac_tensor, mdata = load_and_preprocess(args.data_path)

n_obs = mdata.n_obs
obs_names = mdata.obs_names


#implement a train-test split
train_names, test_names = train_test_split(
    obs_names, # Split the list of barcodes
    test_size=args.test_size,
    random_state=args.seed
)

mdata_index = mdata.obs_names 
train_indices = mdata_index.get_indexer(train_names)
test_indices = mdata_index.get_indexer(test_names)


rna_train_tensor = rna_tensor[train_indices]
atac_train_tensor = atac_tensor[train_indices]
rna_test_tensor = rna_tensor[test_indices]
atac_test_tensor = atac_tensor[test_indices]


print(f"Train tensor shapes: RNA {rna_train_tensor.shape}, ATAC {atac_train_tensor.shape}")
print(f"Test tensor shapes: RNA {rna_test_tensor.shape}, ATAC {atac_test_tensor.shape}")

# Subset RNA modality
rna_train_adata = mdata.mod['rna'][train_names, :].copy()
rna_test_adata = mdata.mod['rna'][test_names, :].copy()

# Subset ATAC modality
atac_train_adata = mdata.mod['atac'][train_names, :].copy()
atac_test_adata = mdata.mod['atac'][test_names, :].copy()

# Create new MuData objects from the subsetted AnnDatas
mdata_train = mu.MuData({'rna': rna_train_adata, 'atac': atac_train_adata})
mdata_test = mu.MuData({'rna': rna_test_adata, 'atac': atac_test_adata})

print(f"Train MuData shape: {mdata_train.shape}")
print(f"Test MuData shape: {mdata_test.shape}")

# save the mdata(test) for visualization and further analysis
run_identifier = f"{args.fusion}_betaMax{args.kl_beta_max}_warmup{args.kl_warmup_epochs}"
mdata_test_save_path = os.path.join(args.checkpoint_dir, f"mdata_test_{run_identifier}.h5mu") 
print(f"Saving processed test MuData to: {mdata_test_save_path}")
mdata_test.write(mdata_test_save_path)


model = MultimodalVAE(
    rna_dim=rna_train_tensor.shape[1],
    atac_dim=atac_train_tensor.shape[1],
    hidden_dim=512,
    latent_dim=32,
    fusion_hidden_dim=128,
    fuse=args.fusion
    )

run_identifier = f"{args.fusion}_betaMax{args.kl_beta_max}_warmup{args.kl_warmup_epochs}"
save_path = os.path.join(args.checkpoint_dir, f"{run_identifier}_{args.model_name}")

train_vae(model,
        rna_train_tensor,
        atac_train_tensor, 
        save_path=save_path,
        kl_beta_max=args.kl_beta_max,         
        kl_warmup_epochs=args.kl_warmup_epochs,
        kl_anneal_strategy=args.kl_anneal_strategy, 
        lr= args.lr, 
        atac_weight= args.atac_weight,
        rna_weight= args.rna_weight, 
        epochs = args.epochs
        ) 


# --- Evaluation ---
print(f"\n--- Evaluating Final Model on Test Set ---")
print(f"Loading final model from: {save_path}")

model = MultimodalVAE( 
    rna_dim=rna_train_tensor.shape[1],
    atac_dim=atac_train_tensor.shape[1],
    hidden_dim=512,
    latent_dim=32,
    fusion_hidden_dim=128,
    fuse=args.fusion
).to(device)

model.load_state_dict(torch.load(save_path, map_location=device))


evaluate_imputation(model, rna_test_tensor, atac_test_tensor, device=device)

