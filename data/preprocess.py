import muon as mu
import scanpy as sc
import numpy as np
import torch
from sklearn.decomposition import TruncatedSVD
from scipy import sparse
import warnings
import pandas as pd # Import pandas for index operations

def load_and_preprocess(
    h5_path: str,
    n_rna_hvg: int = 8000,
    n_atac_components: int = 51, # Use N+1, since we discard the first component
    min_rna_genes_per_cell: int = 200,
    max_rna_genes_per_cell: int = 5000,
    max_pct_mito_per_cell: float = 15,
    min_atac_peaks_per_cell: int = 1000,
    dtype=torch.float32
):
    print(f"Loading 10x data from: {h5_path}")
    mdata_tmp = mu.read_10x_h5(h5_path)
    mdata_tmp.var_names_make_unique()

    if 'rna' not in mdata_tmp.mod or 'atac' not in mdata_tmp.mod:
        raise ValueError("Required 'rna' or 'atac' modality not found.")
    rna_adata = mdata_tmp.mod['rna'].copy() 
    atac_adata = mdata_tmp.mod['atac'].copy()

    # Removing Prefixes to makes barcode consistent
    rna_prefix = "rna:"
    atac_prefix = "atac:"

    # Check and fix RNA
    if all(name.startswith(rna_prefix) for name in rna_adata.obs_names[:10]):
        print(f"Removing prefix '{rna_prefix}' from RNA obs_names...")
        rna_adata.obs_names = rna_adata.obs_names.str.removeprefix(rna_prefix)

    # Check and fix ATAC
    if all(name.startswith(atac_prefix) for name in atac_adata.obs_names[:10]):
        print(f"Removing prefix '{atac_prefix}' from ATAC obs_names...")
        atac_adata.obs_names = atac_adata.obs_names.str.removeprefix(atac_prefix)

    # Create NEW MuData object with corrected barcodes 
    
    rna_adata.obs.index = pd.Index(rna_adata.obs_names) 
    atac_adata.obs.index = pd.Index(atac_adata.obs_names)

    mdata = mu.MuData({'rna': rna_adata, 'atac': atac_adata})
    print("New MuData object created:")
    print(mdata)
        

    # Calculate RNA QC
    print("\n--- Calculating RNA QC metrics ---")
    rna = mdata.mod['rna'] 
    rna.var['mt'] = rna.var_names.str.startswith(('MT-', 'mt-'))
    sc.pp.calculate_qc_metrics(rna, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)

    # Calculate ATAC QC
    print("\n--- Calculating ATAC QC metrics ---")
    atac = mdata.mod['atac'] 
    sc.pp.calculate_qc_metrics(atac, percent_top=None, log1p=False, inplace=True)

    # Boolean masks for cells passing QC in each modality
    print("\n--- Determining cells passing QC ---")
    rna_pass_qc = (
        (rna.obs['n_genes_by_counts'] >= min_rna_genes_per_cell) &
        (rna.obs['n_genes_by_counts'] <= max_rna_genes_per_cell) &
        (rna.obs['pct_counts_mt'] < max_pct_mito_per_cell)
    )
    atac_pass_qc = (
        (atac.obs['n_genes_by_counts'] >= min_atac_peaks_per_cell)
    )

    # Observation names that pass QC for both modalities
    rna_pass_cells = rna.obs_names[rna_pass_qc]
    atac_pass_cells = atac.obs_names[atac_pass_qc]
    common_pass_cells = rna_pass_cells.intersection(atac_pass_cells)
    print(f"Found {len(common_pass_cells)} cells passing both RNA and ATAC QC.")

    # Filter 
    mdata = mdata[common_pass_cells, :].copy() 
    print(f"MuData shape after combined QC filtering: {mdata.shape}")

    

    rna = mdata.mod['rna']
    # RNA: Store raw, Normalize, Log, Store Log, HVG

    rna.layers['counts'] = rna.X.copy() # Store raw counts in layers; helps later with viua1lization.
    sc.pp.normalize_total(rna, target_sum=1e4)
    sc.pp.log1p(rna)
    rna.layers['lognorm'] = rna.X.copy() # Storing log-normalized counts in rna.layers['lognorm'].


    # Select highly variable genes(hvg)
    sc.pp.highly_variable_genes(rna, n_top_genes=n_rna_hvg, flavor='seurat_v3', subset=True)

    X_rna = rna.X
    if hasattr(X_rna, "toarray"):
        X_rna = X_rna.toarray()
    else:
        X_rna = np.asarray(X_rna)

    X_rna_tensor = torch.tensor(X_rna, dtype=dtype)
    print(f"Final RNA tensor shape: {X_rna_tensor.shape}")

    # ATAC: SVD
    atac = mdata.mod['atac']

    X_atac_filtered = atac.X 

    # TruncatedSVD
    effective_n_components = min(n_atac_components, X_atac_filtered.shape[1] - 1, X_atac_filtered.shape[0] - 1)
    if effective_n_components < 2: raise ValueError("Not enough cells/peaks for SVD post-filtering")
    if effective_n_components < n_atac_components: print(f"Warning: Reducing SVD components to {effective_n_components}")

    svd = TruncatedSVD(n_components=effective_n_components, random_state=42, algorithm='arpack')
    X_atac_reduced = svd.fit_transform(X_atac_filtered)
    X_atac_final = X_atac_reduced[:, 1:] # Discard the first LSI component.

    atac.obsm['X_lsi'] = X_atac_final 

    X_atac_tensor = torch.tensor(X_atac_final, dtype=dtype)
    print(f"Final ATAC tensor shape: {X_atac_tensor.shape}")

    # ---checking the mean and std---
    print("\n--- Final Processed Tensor Stats ---")
    print(f"RNA tensor mean: {X_rna_tensor.mean():.4f}, std: {X_rna_tensor.std():.4f}")
    print(f"ATAC tensor mean: {X_atac_tensor.mean():.4f}, std: {X_atac_tensor.std():.4f}")
    print("-" * 35)

    return X_rna_tensor, X_atac_tensor, mdata 

