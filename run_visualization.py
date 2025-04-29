import argparse
import torch
import muon as mu
import scanpy as sc 
import numpy as np
import os
from scripts.visualize import visualize_latent_space
from models.vae import MultimodalVAE 



parser = argparse.ArgumentParser(description="Visualize VAE latent space and imputation.")

parser.add_argument('--model_path', type=str, required=True,
                    help='Path to the trained VAE model checkpoint (.pt).')
parser.add_argument('--mdata_test_path', type=str, required=True,
                    help='Path to the processed test MuData object (.h5mu).')
parser.add_argument('--vis_output_dir', type=str, default="results/plots",
                    help="Directory to save visualization plots.")


parser.add_argument('--vis_cluster_res', type=float, default=1.0)
parser.add_argument('--vis_umap_neighbors', type=int, default=30)
parser.add_argument('--vis_umap_min_dist', type=float, default=0.3)
parser.add_argument('--vis_marker_n_genes', type=int, default=10)
parser.add_argument('--seed', type=int, default=101, help='Random seed for UMAP reproducibility.')

parser.add_argument('--fusion', type=str, choices=['mlp', 'poe'], required=True,
                    help="Fusion method used by the loaded model ('mlp' or 'poe').")



args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")


np.random.seed(args.seed)
torch.manual_seed(args.seed)


print(f"Loading processed test MuData from: {args.mdata_test_path}")

mdata_test = mu.read(args.mdata_test_path)
print("Test MuData loaded:")


rna_dim = mdata_test.mod['rna'].shape[1] 
atac_dim = mdata_test.mod['atac'].obsm['X_lsi'].shape[1] 

model = MultimodalVAE( 
    rna_dim=rna_dim,
    atac_dim=atac_dim,
    hidden_dim=512,
    latent_dim=32,
    fusion_hidden_dim=128,
    fuse=args.fusion
).to(device)

model.load_state_dict(torch.load(args.model_path, map_location=device))
model.eval()  
print("Model loaded successfully.")



# --- Run Visualization ---
print(f"\n--- Generating Visualizations ---")

visualize_latent_space( 
    model=model,
    mdata_test=mdata_test,
    device=device,
    output_dir=args.vis_output_dir,
    cluster_resolution=args.vis_cluster_res,
    umap_neighbors=args.vis_umap_neighbors,
    umap_min_dist=args.vis_umap_min_dist,
    marker_n_genes=args.vis_marker_n_genes,
    umap_random_state=args.seed 
)


print("\nVisualization is complete.")