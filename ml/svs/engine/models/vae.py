import torch
import torch.nn as nn
import logging
from engine.models.temporal_blocks import MultiScaleTemporalBlock


class VAE_FrameLevel(nn.Module):
    def __init__(self, d_model: int, z_dim: int = 128, debug: bool = False):
        super().__init__()
        # より強力なエンコーダー
        self.encoder = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 2, d_model)
        )
        
        self.mu_fc     = nn.Linear(d_model, z_dim)
        self.logvar_fc = nn.Linear(d_model, z_dim)
        
        # Enhanced multi-scale temporal modeling for 5-second sequences
        self.temporal_block = MultiScaleTemporalBlock(d_model, dropout=0.1)
        self.temporal_norm = nn.LayerNorm(d_model)
        
        self.debug = debug

    def forward(self, h: torch.Tensor, lengths: torch.Tensor):
        """
        Frame-level VAE processing
        h       : [B, T, d_model] - frame-level features
        lengths : [B]  mel frame count
        
        Returns:
        z       : [B, T, z_dim] - frame-level latent variables
        mu      : [B, T, z_dim] - frame-level means
        logvar  : [B, T, z_dim] - frame-level log variances
        """
        B, T, d_model = h.shape
        device = h.device

        # --- padding mask ---
        idxs = torch.arange(T, device=device).unsqueeze(0)          # [1,T]
        mask = (idxs < lengths.unsqueeze(1)).unsqueeze(-1).float()  # [B,T,1]
        
        if self.debug:
            logging.debug(f"[VAE] input h: {h.shape}, mask: {mask.shape}")

        # --- Enhanced multi-scale temporal modeling for 5-second sequences ---
        h_temp = h.transpose(1, 2)  # [B, d_model, T] for temporal block
        h_temp_enhanced = self.temporal_block(h_temp)  # Multi-scale processing
        h_temp_enhanced = h_temp_enhanced.transpose(1, 2)  # [B, T, d_model]
        h_temp_enhanced = self.temporal_norm(h_temp_enhanced)
        
        # Combine original and enhanced temporal features
        h_enhanced = h + h_temp_enhanced  # residual connection
        
        # より強力なエンコーダーを適用
        h_encoded = self.encoder(h_enhanced)
        h_masked = h_encoded * mask  # Apply mask
        
        if self.debug:
            logging.debug(f"[VAE] h_enhanced: {h_enhanced.shape}")

        # --- frame-wise latent variables ---
        mu = self.mu_fc(h_masked)      # [B, T, z_dim]
        logvar = self.logvar_fc(h_masked)  # [B, T, z_dim]
        
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std  # reparameterization
        
        # Apply mask to latent variables too
        z = z * mask
        mu = mu * mask
        logvar = logvar * mask
        
        if self.debug:
            logging.debug(f"[VAE] mu: {mu.shape}, logvar: {logvar.shape}, z: {z.shape}")

        return z, mu, logvar


