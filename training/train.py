import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from training.loss import multimodal_vae_loss
from torch.utils.tensorboard import SummaryWriter
import os
import torch.nn.utils


def train_vae(
    model,
    X_rna,
    X_atac,
    save_path="checkpoints/model.pt",
    epochs=20,
    batch_size=64,
    lr=1e-3,
    clip_grad_norm=None,   
    rna_weight=1.0,
    atac_weight=1.0,
    kl_beta_max=1.0,         
    kl_warmup_epochs=100,     
    kl_anneal_strategy='linear',
    device='cuda' if torch.cuda.is_available() else 'cpu',
    log_dir="runs/vae_experiment"
):
    model.to(device)
    model.train()

    dataset = TensorDataset(X_rna, X_atac)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # --- KL Annealing Helper Function ---
    def get_current_beta(epoch, warmup_epochs, beta_max, strategy='linear'):
        if warmup_epochs <= 0:
             return beta_max
        if epoch < warmup_epochs:
            if strategy == 'linear':
                return beta_max * float(epoch + 1) / float(max(1, warmup_epochs))
            elif strategy == 'sigmoid':
                scale_factor = 12.0
                sigmoid_val = torch.sigmoid(torch.tensor((epoch - warmup_epochs / 2) * scale_factor / warmup_epochs)).item()
                start_val = torch.sigmoid(torch.tensor((-warmup_epochs / 2) * scale_factor / warmup_epochs)).item()
                end_val = torch.sigmoid(torch.tensor((warmup_epochs / 2) * scale_factor / warmup_epochs)).item()
                return beta_max * (sigmoid_val - start_val) / (end_val - start_val + 1e-8) 
            else:
                raise ValueError(f"Unknown annealing strategy: {strategy}")
        else:
            return beta_max



    writer = SummaryWriter(log_dir=log_dir)

    for epoch in range(epochs):
        total, rna_total, atac_total, kl_total = 0, 0, 0, 0
        model.train()

        current_beta = get_current_beta(epoch, kl_warmup_epochs, kl_beta_max, kl_anneal_strategy)

        for batch_idx, (x_rna, x_atac) in enumerate(tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")):
            x_rna = x_rna.to(device)
            x_atac = x_atac.to(device)

            recon_rna, recon_atac, mu_z, logvar_z = model(x_rna, x_atac)

            # Calculate loss using  dynamically calculated current_beta
            loss, rna_loss, atac_loss, kl = multimodal_vae_loss(
                recon_rna, recon_atac, x_rna, x_atac, mu_z, logvar_z,
                beta=current_beta, 
                rna_weight=rna_weight,
                atac_weight=atac_weight
            )

            optimizer.zero_grad()
            loss.backward()

            if clip_grad_norm is not None and clip_grad_norm > 0: #Gradient clipping
                total_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=clip_grad_norm
                )

            optimizer.step()

            total += loss.item()
            rna_total += rna_loss.item()
            atac_total += atac_loss.item()
            kl_total += kl.item()

        avg_loss = total / len(loader)
        avg_rna = rna_total / len(loader)
        avg_atac = atac_total / len(loader)
        avg_kl = kl_total / len(loader)

        writer.add_scalar("Loss/Total_Weighted", avg_loss, epoch + 1)
        writer.add_scalar("Loss/RNA_Unweighted", avg_rna, epoch + 1)
        writer.add_scalar("Loss/ATAC_Unweighted", avg_atac, epoch + 1)
        writer.add_scalar("Loss/KL_Raw", avg_kl, epoch + 1) 
        writer.add_scalar("Annealing/CurrentBeta", current_beta, epoch + 1) 
        writer.add_scalar("LearningRate", optimizer.param_groups[0]['lr'], epoch + 1)

        # Print losses 
        print(f"[Epoch {epoch+1}] Loss(W): {avg_loss:.4f} | RNA: {avg_rna:.4f} | ATAC: {avg_atac:.4f} | KL: {avg_kl:.4f} | Beta: {current_beta:.4f}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
    writer.close()
