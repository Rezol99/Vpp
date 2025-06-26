import torch
import torch.nn as nn
import torch.nn.functional as F

class VocalDiffusionLoss(nn.Module):
    def __init__(self, prediction_type="v", snr_gamma=5.0):
        super().__init__()
        self.prediction_type = prediction_type
        self.snr_gamma = snr_gamma
        
    def forward(self, model_output, target, timesteps, noise_scheduler):
        """
        Args:
            model_output: モデルの予測
            target: 正解データ（x0）
            timesteps: 拡散ステップ
            noise_scheduler: ノイズスケジューラ
        """
        # SNR計算
        snr = self.compute_snr(timesteps, noise_scheduler)
        
        if self.prediction_type == "v":
            # v-prediction（最も安定）
            v_target = self.get_v_target(target, timesteps, noise_scheduler)
            loss = F.mse_loss(model_output, v_target, reduction='none')
            
        elif self.prediction_type == "epsilon":
            # ノイズ予測（標準的）
            loss = F.mse_loss(model_output, target, reduction='none')
            
        # SNR重み付け（重要！）
        snr_weight = torch.minimum(snr, torch.ones_like(snr) * self.snr_gamma) / snr
        weighted_loss = (loss * snr_weight.unsqueeze(-1).unsqueeze(-1)).mean()
        
        return weighted_loss
    
    def compute_snr(self, timesteps, noise_scheduler):
        """SNR（Signal-to-Noise Ratio）を計算"""
        alphas_cumprod = noise_scheduler.alphas_cumprod.to(timesteps.device)
        
        # Ensure timesteps are within valid bounds
        timesteps = torch.clamp(timesteps, 0, len(alphas_cumprod) - 1)
        
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod[timesteps])
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod[timesteps])
        snr = sqrt_alphas_cumprod / sqrt_one_minus_alphas_cumprod
        return snr
    
    def get_v_target(self, x0, timesteps, noise_scheduler):
        """v-prediction用のターゲットを計算"""
        noise = torch.randn_like(x0)
        alphas_cumprod = noise_scheduler.alphas_cumprod.to(timesteps.device)
        
        # Ensure timesteps are within valid bounds
        timesteps = torch.clamp(timesteps, 0, len(alphas_cumprod) - 1)
        
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod[timesteps])
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod[timesteps])
        
        # v = sqrt(alpha_t) * noise - sqrt(1-alpha_t) * x0
        v_target = sqrt_alphas_cumprod.unsqueeze(-1).unsqueeze(-1) * noise - \
                   sqrt_one_minus_alphas_cumprod.unsqueeze(-1).unsqueeze(-1) * x0
        return v_target