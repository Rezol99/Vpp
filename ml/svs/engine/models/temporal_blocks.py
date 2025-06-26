"""
temporal_blocks.py
Enhanced temporal modeling components for 5-second audio understanding.
Implements multi-scale receptive fields and hierarchical processing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiScaleTemporalBlock(nn.Module):
    """
    Multi-scale temporal block with different kernel sizes and dilations
    to capture both local and long-range dependencies for 5-second audio.
    """
    def __init__(self, channels: int, dropout: float = 0.1):
        super().__init__()
        
        # Multiple scales for different temporal patterns
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(channels, channels, kernel_size=7, padding=6, dilation=2)
        self.conv4 = nn.Conv1d(channels, channels, kernel_size=9, padding=16, dilation=4)
        self.conv5 = nn.Conv1d(channels, channels, kernel_size=11, padding=40, dilation=8)
        
        # Normalization for each scale
        self.norm1 = nn.GroupNorm(8, channels)
        self.norm2 = nn.GroupNorm(8, channels)
        self.norm3 = nn.GroupNorm(8, channels)
        self.norm4 = nn.GroupNorm(8, channels)
        self.norm5 = nn.GroupNorm(8, channels)
        
        # Scale weights (learnable)
        self.scale_weights = nn.Parameter(torch.ones(5) / 5)
        
        # Final projection
        self.projection = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.activation = nn.GELU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, channels, T] - input features
        Returns:
            [B, channels, T] - multi-scale temporal features
        """
        # Apply different scales
        out1 = self.activation(self.norm1(self.conv1(x)))
        out2 = self.activation(self.norm2(self.conv2(x)))
        out3 = self.activation(self.norm3(self.conv3(x)))
        out4 = self.activation(self.norm4(self.conv4(x)))
        out5 = self.activation(self.norm5(self.conv5(x)))
        
        # Weighted combination
        weights = F.softmax(self.scale_weights, dim=0)
        combined = (weights[0] * out1 + 
                   weights[1] * out2 + 
                   weights[2] * out3 + 
                   weights[3] * out4 + 
                   weights[4] * out5)
        
        # Final projection and residual connection
        output = self.projection(combined)
        return x + output


class HierarchicalEncoder(nn.Module):
    """
    Hierarchical encoder that processes audio at multiple temporal resolutions
    to efficiently handle 5-second sequences.
    """
    def __init__(self, d_model: int, num_layers: int = 6, heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        
        # Fine-scale processing (full resolution)
        self.fine_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, 
                nhead=heads, 
                batch_first=True,
                dropout=dropout,
                dim_feedforward=d_model * 4
            ) for _ in range(num_layers // 2)
        ])
        
        # Coarse-scale processing (downsampled)
        self.downsample = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, d_model),
            nn.GELU()
        )
        
        self.coarse_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, 
                nhead=heads, 
                batch_first=True,
                dropout=dropout,
                dim_feedforward=d_model * 4
            ) for _ in range(num_layers // 2)
        ])
        
        self.upsample = nn.Sequential(
            nn.ConvTranspose1d(d_model, d_model, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, d_model),
            nn.GELU()
        )
        
        # Cross-scale attention
        self.cross_scale_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=heads, batch_first=True, dropout=dropout
        )
        
        # Multi-scale temporal blocks
        self.temporal_block_fine = MultiScaleTemporalBlock(d_model, dropout)
        self.temporal_block_coarse = MultiScaleTemporalBlock(d_model, dropout)
        
        # Final fusion
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model)
        )
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: [B, T, d_model] - input sequence
            mask: [B, T] - padding mask (True for padding)
        Returns:
            [B, T, d_model] - hierarchically processed features
        """
        B, T, d_model = x.shape
        
        # Fine-scale processing (full resolution)
        fine_features = x
        for layer in self.fine_layers:
            fine_features = layer(fine_features, src_key_padding_mask=mask)
        
        # Apply multi-scale temporal processing to fine features
        fine_conv = fine_features.transpose(1, 2)  # [B, d_model, T]
        fine_conv = self.temporal_block_fine(fine_conv)
        fine_features_enhanced = fine_conv.transpose(1, 2)  # [B, T, d_model]
        
        # Coarse-scale processing (downsampled for long-range dependencies)
        coarse_input = x.transpose(1, 2)  # [B, d_model, T]
        coarse_downsampled = self.downsample(coarse_input)  # [B, d_model, T//2]
        coarse_features = coarse_downsampled.transpose(1, 2)  # [B, T//2, d_model]
        
        # Create mask for downsampled sequence
        coarse_mask = None
        if mask is not None:
            coarse_mask = mask[:, ::2]  # Downsample mask
        
        # Process at coarse scale
        for layer in self.coarse_layers:
            coarse_features = layer(coarse_features, src_key_padding_mask=coarse_mask)
        
        # Apply multi-scale temporal processing to coarse features
        coarse_conv = coarse_features.transpose(1, 2)  # [B, d_model, T//2]
        coarse_conv = self.temporal_block_coarse(coarse_conv)
        coarse_features_enhanced = coarse_conv.transpose(1, 2)  # [B, T//2, d_model]
        
        # Upsample coarse features back to original resolution
        coarse_upsampled = self.upsample(coarse_features_enhanced.transpose(1, 2))  # [B, d_model, T]
        
        # Handle potential size mismatch due to convolution
        if coarse_upsampled.size(-1) != T:
            coarse_upsampled = F.interpolate(
                coarse_upsampled, size=T, mode='linear', align_corners=False
            )
        
        coarse_upsampled = coarse_upsampled.transpose(1, 2)  # [B, T, d_model]
        
        # Cross-scale attention (let fine features attend to coarse features)
        cross_attn_out, _ = self.cross_scale_attn(
            fine_features_enhanced, coarse_upsampled, coarse_upsampled,
            key_padding_mask=mask
        )
        
        # Combine fine and cross-attended features
        combined = torch.cat([fine_features_enhanced, cross_attn_out], dim=-1)  # [B, T, d_model*2]
        
        # Final fusion
        output = self.fusion(combined)  # [B, T, d_model]
        
        return output


class LinearAttention(nn.Module):
    """
    Linear attention mechanism with O(T) complexity instead of O(T²).
    Essential for efficient processing of 5-second sequences.
    """
    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64, dropout: float = 0.1):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: [B, T, dim] - input sequence
            mask: [B, T] - padding mask (True for padding)
        Returns:
            [B, T, dim] - linearly attended features
        """
        B, T, dim = x.shape
        
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(B, T, self.heads, self.dim_head).transpose(1, 2), qkv)
        
        # Apply scaling
        q = q * self.scale
        
        # Linear attention: use softmax on key dimension instead of query dimension
        # This reduces complexity from O(T²) to O(T)
        k = F.softmax(k, dim=-2)
        
        # Apply mask to keys if provided
        if mask is not None:
            # Resize mask to match the temporal dimension of k
            if mask.shape[-1] != T:
                # Downsample or upsample mask to match T
                mask = F.interpolate(mask.float().unsqueeze(1), size=T, mode='nearest').squeeze(1).bool()
            mask = mask[:, None, :, None].expand(-1, self.heads, -1, self.dim_head)
            k = k.masked_fill(mask, 0.0)
        
        # Compute attention: O(T) complexity
        context = torch.einsum('bhnd,bhne->bhde', k, v)  # [B, heads, dim_head, dim_head]
        out = torch.einsum('bhnd,bhde->bhne', q, context)  # [B, heads, T, dim_head]
        
        # Reshape and project
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.to_out(out)


class EnhancedTemporalTransformer(nn.Module):
    """
    Transformer with enhanced temporal modeling capabilities for 5-second audio processing.
    Combines hierarchical processing, multi-scale temporal blocks, and linear attention.
    """
    def __init__(
        self, 
        d_model: int, 
        num_layers: int = 8, 
        heads: int = 8, 
        dropout: float = 0.1,
        use_linear_attention: bool = True,
        use_hierarchical: bool = True
    ):
        super().__init__()
        self.use_hierarchical = use_hierarchical
        self.use_linear_attention = use_linear_attention
        
        if use_hierarchical:
            self.hierarchical_encoder = HierarchicalEncoder(
                d_model, num_layers, heads, dropout
            )
        else:
            # Standard transformer layers
            self.layers = nn.ModuleList([
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=heads,
                    batch_first=True,
                    dropout=dropout,
                    dim_feedforward=d_model * 4
                ) for _ in range(num_layers)
            ])
        
        if use_linear_attention:
            self.linear_attention = LinearAttention(d_model, heads, d_model // heads, dropout)
        
        # Multi-scale temporal processing
        self.temporal_block = MultiScaleTemporalBlock(d_model, dropout)
        
        # Layer normalization
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: [B, T, d_model] - input sequence
            mask: [B, T] - padding mask (True for padding)
        Returns:
            [B, T, d_model] - temporally enhanced features
        """
        if self.use_hierarchical:
            # Use hierarchical processing
            x = self.hierarchical_encoder(x, mask)
        else:
            # Use standard transformer layers
            for layer in self.layers:
                x = layer(x, src_key_padding_mask=mask)
        
        # Apply linear attention for additional efficiency
        if self.use_linear_attention:
            x_attn = self.linear_attention(x, mask)
            x = x + x_attn
        
        # Apply multi-scale temporal processing
        x_conv = x.transpose(1, 2)  # [B, d_model, T]
        x_conv = self.temporal_block(x_conv)
        x_temporal = x_conv.transpose(1, 2)  # [B, T, d_model]
        
        # Combine and normalize
        x = x + x_temporal
        x = self.norm(x)
        
        return x