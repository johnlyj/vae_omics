
import torch
import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import warnings
from models.vae import MultimodalVAE
import umap


def generate_latent(model, rna_tensor, atac_tensor, device):
    """Generates the mean of the latent space (mu_z) from input."""
    model.eval()
    model.to(device)
    mu_z_list = []
    with torch.no_grad():
        X_rna_dev = rna_tensor.to(device)
        X_atac_dev = atac_tensor.to(device)

        mu_rna, logvar_rna = model.rna_encoder(X_rna_dev)
        mu_atac, logvar_atac = model.atac_encoder(X_atac_dev)

        if model.fuse == "poe":
            mu_z, _ = model.product_of_experts(mu_rna, logvar_rna, mu_atac, logvar_atac)
        elif model.fuse == "mlp":
            mu_z, _ = model.fusion(mu_rna, logvar_rna, mu_atac, logvar_atac)
        else:
            raise ValueError(f"Invalid fusion method in model: {model.fuse}")

        mu_z_list.append(mu_z.cpu().numpy())

    return np.concatenate(mu_z_list, axis=0)


# Main visualization function
def visualize_latent_space(
    model, # The trained VAE model instance
    mdata_test, # MuData object containing the processed TEST data split
    device, 
    output_dir="results/plots", 
    cluster_resolution=1.0,    # Resolution for Leiden clustering
    umap_neighbors=30,         # Number of neighbors for UMAP/neighborhood graph
    umap_min_dist=0.3,         # Min distance for UMAP layout
    marker_n_genes=10,         # Number of marker genes to show per cluster
    umap_random_state=42          
):
    
    os.makedirs(output_dir, exist_ok=True)

    
    rna_test_adata = mdata_test.mod['rna']
    atac_test_adata = mdata_test.mod['atac']

    # RNA input for VAE (lognorm + HVG)
    X_rna_test_hvg = rna_test_adata.X
    if hasattr(X_rna_test_hvg, "toarray"):
        X_rna_test_hvg = X_rna_test_hvg.toarray()
    else:
        X_rna_test_hvg = np.asarray(X_rna_test_hvg)
    X_rna_test_tensor = torch.tensor(X_rna_test_hvg, dtype=torch.float32)

    # ATAC input for VAE (LSI components)
    
    X_atac_test_lsi = atac_test_adata.obsm['X_lsi']
    X_atac_test_tensor = torch.tensor(X_atac_test_lsi, dtype=torch.float32)

    print(f"Using RNA tensor shape: {X_rna_test_tensor.shape}")
    print(f"Using ATAC tensor shape: {X_atac_test_tensor.shape}")

    # Generate Latent Space ---
    mu_z_test_np = generate_latent(model, X_rna_test_tensor, X_atac_test_tensor, device)
    # [n_test_cells x latent_dim]


    # Create AnnData 

    # Use barcodes from the test MuData
    adata_latent = sc.AnnData(mu_z_test_np, obs=pd.DataFrame(index=mdata_test.obs_names.copy()))
    
    
    sc.pp.neighbors(adata_latent, n_neighbors=umap_neighbors, use_rep='X')
    sc.tl.leiden(adata_latent, resolution=cluster_resolution, key_added='leiden_clusters', random_state= 42)
    sc.tl.umap(adata_latent, min_dist=umap_min_dist, random_state=42)

    print("Cluster counts:")
    print(adata_latent.obs['leiden_clusters'].value_counts())
    

    # Marker Gene Analysis & Annotation 
    
    print("Preparing RNA data for marker analysis (using 'lognorm' layer)...")

    # Create a temporary AnnData with the lognorm data 
    rna_lognorm_adata = sc.AnnData(
        mdata_test.mod['rna'].layers['lognorm'], # Use lognorm layer
        obs=mdata_test.mod['rna'].obs.copy(),     # Use og RNA obs
        var=mdata_test.mod['rna'].var.copy()      # Use og RNA var
    )

    rna_lognorm_adata.obs['leiden_clusters'] = adata_latent.obs['leiden_clusters'].copy()

    #rank genes
    
    sc.tl.rank_genes_groups(rna_lognorm_adata, groupby='leiden_clusters', method='wilcoxon', use_raw=False, key_added='marker_genes')

    # Plot and save markers
    marker_plot_path = os.path.join(output_dir, "marker_genes_dotplot.png")
    sc.pl.rank_genes_groups_dotplot(rna_lognorm_adata, n_genes=marker_n_genes, key='marker_genes', groupby='leiden_clusters', show=False)

    plt.savefig(marker_plot_path, dpi=300)
    plt.close()
    print(f"Saved dotplot to {marker_plot_path}")
        

    # Plot Latent space UMAPs 
    umap_plot_path = os.path.join(output_dir, "latent_space_umap_annotated.png")
    sc.pl.umap(adata_latent,color='leiden_clusters',show=False,
    title=f'VAE Latent Space UMAP (Fusion: {model.fuse.upper()})',
    legend_loc='on data',size=20)

    plt.savefig(umap_plot_path, dpi=200)
    plt.close()

    print(f" UMAP plots saved to {umap_plot_path}")
    
    

#  Imputation Visualization ---
    print("\n--- Generating Imputation Comparison UMAPs ---")
    model.eval() 
    with torch.no_grad():
        
        # --- RNA -> ATAC Imputation ---


        mu_rna, logvar_rna = model.rna_encoder(X_rna_test_tensor.to(device))
        mu_atac_prior = torch.zeros_like(mu_rna)
        logvar_atac_prior = torch.zeros_like(logvar_rna)
        if model.fuse == "poe":
            mu_z, logvar_z = model.product_of_experts(mu_rna, logvar_rna, mu_atac_prior, logvar_atac_prior)
        elif model.fuse == "mlp":
            mu_z, logvar_z = model.fusion(mu_rna, logvar_rna, mu_atac_prior, logvar_atac_prior)
        z = model.reparameterize(mu_z, logvar_z)

        atac_imputed_np = model.atac_decoder(z).cpu().numpy()

        # --- ATAC -> RNA Imputation ---


        mu_atac, logvar_atac = model.atac_encoder(X_atac_test_tensor.to(device))
        mu_rna_prior = torch.zeros_like(mu_atac)
        logvar_rna_prior = torch.zeros_like(logvar_atac)
        if model.fuse == "poe":
            mu_z, logvar_z = model.product_of_experts(mu_rna_prior, logvar_rna_prior, mu_atac, logvar_atac)
        elif model.fuse == "mlp":
            mu_z, logvar_z = model.fusion(mu_rna_prior, logvar_rna_prior, mu_atac, logvar_atac)

        z = model.reparameterize(mu_z, logvar_z)
        rna_imputed_hvg_np = model.rna_decoder(z).cpu().numpy()

    # --- Prepare Data for UMAP ---

    # Original Data
    original_rna_hvg_np = X_rna_test_hvg 
    original_atac_lsi_np = X_atac_test_lsi

    # Create temporary AnnDatas for UMAP fitting and plotting
    adata_orig_rna = sc.AnnData(original_rna_hvg_np, obs=adata_latent.obs.copy()) # Use labels from latent
    adata_imp_rna = sc.AnnData(rna_imputed_hvg_np, obs=adata_latent.obs.copy())
    adata_orig_atac = sc.AnnData(original_atac_lsi_np, obs=adata_latent.obs.copy())
    adata_imp_atac = sc.AnnData(atac_imputed_np, obs=adata_latent.obs.copy())

    # --- Run UMAP Separately ---
    print("Running UMAP on original/imputed RNA (HVG)...")
    sc.pp.neighbors(adata_orig_rna, n_neighbors=umap_neighbors, use_rep='X')
    sc.tl.umap(adata_orig_rna, min_dist=umap_min_dist, random_state=umap_random_state)
    sc.pp.neighbors(adata_imp_rna, n_neighbors=umap_neighbors, use_rep='X')
    sc.tl.umap(adata_imp_rna, min_dist=umap_min_dist, random_state=umap_random_state)

    print("Running UMAP on original/imputed ATAC (LSI)...")
    sc.pp.neighbors(adata_orig_atac, n_neighbors=umap_neighbors, use_rep='X')
    sc.tl.umap(adata_orig_atac, min_dist=umap_min_dist, random_state=umap_random_state)
    sc.pp.neighbors(adata_imp_atac, n_neighbors=umap_neighbors, use_rep='X')
    sc.tl.umap(adata_imp_atac, min_dist=umap_min_dist, random_state=umap_random_state)

    # --- Plot Comparison UMAPs ---
    print("Plotting Imputation Comparison UMAPs...")
    imputation_plot_path = os.path.join(output_dir, "imputation_comparison_umap.png")

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle('Original vs Imputed Data UMAP Comparison', fontsize=16)

    color_key = 'leiden_clusters'

    sc.pl.umap(adata_orig_rna, color=color_key, ax=axes[0, 0], show=False, title=f'Original RNA (HVG, {color_key})', size=20)
    sc.pl.umap(adata_imp_rna, color=color_key, ax=axes[0, 1], show=False, title=f'Imputed RNA (ATAC->RNA,{color_key})', size=20)
    sc.pl.umap(adata_orig_atac, color=color_key, ax=axes[1, 0], show=False, title=f'Original ATAC (LSI,{color_key})', size=20)
    sc.pl.umap(adata_imp_atac, color=color_key, ax=axes[1, 1], show=False, title=f'Imputed ATAC (RNA->ATAC,{color_key})', size=20)

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(imputation_plot_path);
    print(f"Imputation UMAP plots saved to {imputation_plot_path}");
    plt.close(fig)

    print("--- Visualization Finished ---")

    return adata_latent 