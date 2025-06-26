import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from typing import Optional
import torchaudio


class TensorBoardLogger:
    """TensorBoard logging utilities for engine training"""
    
    def __init__(self, writer: SummaryWriter):
        self.writer = writer
    
    def log_audio(self, tag: str, audio: torch.Tensor, sample_rate: int, global_step: int):
        """Log audio to TensorBoard
        
        Args:
            tag: Name for the audio clip
            audio: Audio tensor [T] or [1, T]
            sample_rate: Sample rate of the audio
            global_step: Global training step
        """
        if audio.dim() == 2:
            audio = audio.squeeze(0)
        
        # Normalize audio to [-1, 1]
        audio = audio / (audio.abs().max() + 1e-8)
        
        self.writer.add_audio(tag, audio.unsqueeze(0), global_step, sample_rate)
    
    def log_spectrogram(self, tag: str, spec: torch.Tensor, global_step: int):
        """Log spectrogram as image
        
        Args:
            tag: Name for the spectrogram
            spec: Spectrogram tensor [H, W]
            global_step: Global training step
        """
        # Normalize to [0, 1] for better visualization
        spec_normalized = (spec - spec.min()) / (spec.max() - spec.min() + 1e-8)
        
        # Add channel dimension if needed
        if spec_normalized.dim() == 2:
            spec_normalized = spec_normalized.unsqueeze(0)
        
        self.writer.add_image(tag, spec_normalized, global_step)
    
    def log_mel_comparison(self, tag: str, target: torch.Tensor, pred: torch.Tensor, 
                          mask: Optional[torch.Tensor], global_step: int, timestep: Optional[int] = None):
        """Log comparison of target and predicted mel spectrograms
        
        Args:
            tag: Base tag name
            target: Target mel spectrogram [n_mels, T]
            pred: Predicted mel spectrogram [n_mels, T]
            mask: Optional mask [n_mels, T]
            global_step: Global training step
            timestep: Optional diffusion timestep to include in visualization
        """
        # Add timestep info to tag if provided
        tag_with_timestep = f"{tag}_t{timestep}" if timestep is not None else tag
        
        # For mel spectrograms, only visualize the valid (masked) regions
        if mask is not None:
            # Apply mask to mel spectrograms for visualization
            target_masked = target * mask
            pred_masked = pred * mask
            
            # Stack for comparison (masked versions)
            comparison = torch.cat([target_masked, pred_masked], dim=0)
            
            self.log_spectrogram(f"{tag_with_timestep}/comparison", comparison, global_step)
            self.log_spectrogram(f"{tag_with_timestep}/target", target_masked, global_step)
            self.log_spectrogram(f"{tag_with_timestep}/predicted", pred_masked, global_step)
            
            # Log mask as-is (not masked)
            self.log_spectrogram(f"{tag_with_timestep}/mask", mask, global_step)
        else:
            # No mask provided, log spectrograms as-is
            comparison = torch.cat([target, pred], dim=0)
            
            self.log_spectrogram(f"{tag_with_timestep}/comparison", comparison, global_step)
            self.log_spectrogram(f"{tag_with_timestep}/target", target, global_step)
            self.log_spectrogram(f"{tag_with_timestep}/predicted", pred, global_step)
            
        # Also log timestep as scalar for easier tracking
        if timestep is not None:
            self.writer.add_scalar(f"{tag}/timestep", timestep, global_step)
    
    def log_gradient_norms(self, model: torch.nn.Module, global_step: int):
        """Log gradient norms for each layer
        
        Args:
            model: The model to log gradients for
            global_step: Global training step
        """
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm(2).item()
                self.writer.add_scalar(f"gradients/{name}", grad_norm, global_step)