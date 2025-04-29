import torch
import torch.nn.functional as F

def kl_divergence(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()

def mse_loss(recon, target):
    return F.mse_loss(recon, target, reduction='mean')

def multimodal_vae_loss(
    recon_rna, recon_atac,
    true_rna, true_atac,
    mu_z, logvar_z,
    beta=1.0,
    rna_weight=1.0,
    atac_weight=1.0
):
    loss_rna = mse_loss(recon_rna, true_rna)
    loss_atac = mse_loss(recon_atac, true_atac)
    kl = kl_divergence(mu_z, logvar_z)
    total_loss = rna_weight * loss_rna + atac_weight * loss_atac + beta * kl
    return total_loss, loss_rna, loss_atac, kl
