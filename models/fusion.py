import torch
import torch.nn as nn
import torch.nn.functional as F

class FusionMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)       
        self.mu = nn.Linear(hidden_dim, latent_dim)       
        self.logvar = nn.Linear(hidden_dim, latent_dim)   

    def forward(self, mu1, logvar1, mu2, logvar2):
        fusion_input = torch.cat([mu1, mu2], dim=-1)  
        h = F.relu(self.fc1(fusion_input))
        mu_z = self.mu(h)
        logvar_z = self.logvar(h)
        return mu_z, logvar_z

class ProductOfExperts(nn.Module):

    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon 

    def forward(self, mu1, logvar1, mu2, logvar2):

        logvar1 = torch.clamp(logvar1, min=-10, max=10) 
        logvar2 = torch.clamp(logvar2, min=-10, max=10) 

        logvars = torch.stack([logvar1, logvar2], dim=0)
        variances = torch.exp(logvars)
        precisions = 1.0 / (variances + self.epsilon)

        mus = torch.stack([mu1, mu2], dim=0)
        weighted_mus = mus * precisions 

        # Combine with Prior (N(0, I)) ---
        # Prior has precision=1 and mean=0
        precision_sum = precisions.sum(dim=0) +1.0
        mu_weighted_sum = weighted_mus.sum(dim=0) #mean=0

        
        sigma_sq_latent = 1.0 / (precision_sum + self.epsilon)
        mu_latent = sigma_sq_latent * mu_weighted_sum
        logvar_latent = torch.log(sigma_sq_latent + self.epsilon)

        return mu_latent, logvar_latent
        

