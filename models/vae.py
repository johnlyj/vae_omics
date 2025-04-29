import torch
import torch.nn as nn
import torch.nn.functional as F

from models.encoder import RNAEncoder, ATACEncoder
from models.decoder import RNADecoder, ATACDecoder
from models.fusion import FusionMLP, ProductOfExperts

class MultimodalVAE(nn.Module):
    def __init__(self, 
                 rna_dim, atac_dim, 
                 hidden_dim, latent_dim, 
                 fusion_hidden_dim,
                 fuse = "mlp"):
        super().__init__()

        self.fuse = fuse

        # Encoders
        self.rna_encoder = RNAEncoder(rna_dim, hidden_dim, latent_dim)
        self.atac_encoder = ATACEncoder(atac_dim, hidden_dim, latent_dim)

        # Fusion network
        self.fusion = FusionMLP(input_dim=2 * latent_dim, 
                                hidden_dim=fusion_hidden_dim, 
                                latent_dim=latent_dim)
        
        #Product of Experts
        self.product_of_experts = ProductOfExperts()

        # Decoders
        self.rna_decoder = RNADecoder(latent_dim, hidden_dim, rna_dim)
        self.atac_decoder = ATACDecoder(latent_dim, hidden_dim, atac_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x_rna, x_atac):
        mu_rna, logvar_rna = self.rna_encoder(x_rna)
        mu_atac, logvar_atac = self.atac_encoder(x_atac)

        if self.fuse == "poe":
            # Use Product of Experts
            mu_z, logvar_z = self.product_of_experts(mu_rna, logvar_rna, mu_atac, logvar_atac)
        
        elif self.fuse == "mlp":
            # Use FusionMLP
            mu_z, logvar_z = self.fusion(mu_rna, logvar_rna, mu_atac, logvar_atac)
        else:
            raise ValueError("Invalid fusion method. Choose 'mlp' or 'poe'.")
        
        # Reparameterization trick

        z = self.reparameterize(mu_z, logvar_z)

        recon_rna = self.rna_decoder(z)
        recon_atac = self.atac_decoder(z)

        return recon_rna, recon_atac, mu_z, logvar_z
