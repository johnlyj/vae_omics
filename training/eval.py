import torch
import torch.nn.functional as F
from torch.nn import CosineSimilarity
from models.vae import MultimodalVAE


def cosine_sim(a, b):
    cos = CosineSimilarity(dim=1)
    return cos(a, b).mean().item()

def pearson_corr(a, b, dim=1, eps=1e-8):
    a = a.float()
    b = b.float()

    # Center the vectors
    a_mean = torch.mean(a, dim=dim, keepdim=True)
    b_mean = torch.mean(b, dim=dim, keepdim=True)
    a_centered = a - a_mean
    b_centered = b - b_mean

    a_std = torch.std(a, dim=dim, keepdim=True, unbiased=True)
    b_std = torch.std(b, dim=dim, keepdim=True, unbiased=True)

    cov = torch.mean(a_centered * b_centered, dim=dim, keepdim=True)

    # Pearson correlation
    pcc = cov / (a_std * b_std + eps)

    pcc = pcc.squeeze()
    if torch.isnan(pcc).all():
        return torch.tensor(float('nan')) # Return NaN scalar if all inputs result in NaN PCC
    return torch.nanmean(pcc).item()


def evaluate_imputation(model, X_rna_test, X_atac_test, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model.eval()
    model.to(device)

    X_rna_test = X_rna_test.to(device)
    X_atac_test = X_atac_test.to(device)

    with torch.no_grad():
        # --- RNA → ATAC ---
        mu_rna, logvar_rna = model.rna_encoder(X_rna_test)
        
        mu_atac = torch.zeros_like(mu_rna)
        logvar_atac = torch.zeros_like(logvar_rna)

        # for mlp, its common to use zeros
        # whereas for poe we use the mean(0) and variance(1)  of the prior
        # coincidentally we use logvar = log(1) = 0, so this is the same

        if model.fuse == "mlp":
            mu_z, logvar_z = model.fusion(mu_rna, logvar_rna, mu_atac, logvar_atac)
        elif model.fuse == "poe":
            mu_z, logvar_z = model.product_of_experts(mu_rna, logvar_rna, mu_atac, logvar_atac)

        z = model.reparameterize(mu_z, logvar_z)
        atac_imputed = model.atac_decoder(z)
        rna2atac_cos = cosine_sim(atac_imputed, X_atac_test)
        rna2atac_pcc = pearson_corr(atac_imputed, X_atac_test)

        # --- ATAC → RNA ---
        mu_atac, logvar_atac = model.atac_encoder(X_atac_test)

        mu_rna = torch.zeros_like(mu_atac)
        logvar_rna = torch.zeros_like(logvar_atac)

        #same logic as before
        if model.fuse == "mlp":
            mu_z, logvar_z = model.fusion(mu_rna, logvar_rna, mu_atac, logvar_atac)
        elif model.fuse == "poe":
            mu_z, logvar_z = model.product_of_experts(mu_rna, logvar_rna, mu_atac, logvar_atac)

        z = model.reparameterize(mu_z, logvar_z)
        rna_imputed = model.rna_decoder(z)
        atac2rna_cos = cosine_sim(rna_imputed, X_rna_test)
        atac2rna_pcc = pearson_corr(rna_imputed, X_rna_test) 

    print(f"\n--- Evaluation Results (Fusion: {model.fuse.upper()}) ---")
    print("Metric       | RNA -> ATAC | ATAC -> RNA")
    print("-------------|-------------|------------")
    print(f"Cosine Sim   | {rna2atac_cos: >11.4f} | {atac2rna_cos: >11.4f}")
    print(f"Pearson Corr | {rna2atac_pcc: >11.4f} | {atac2rna_pcc: >11.4f}")
    print("----------------------------------------")


    return rna2atac_cos, atac2rna_cos, rna2atac_pcc, atac2rna_pcc
