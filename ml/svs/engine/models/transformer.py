import math
import logging

import torch
import torch.nn as nn

from engine.types.dataset import X
from engine.models.temporal_blocks import EnhancedTemporalTransformer, MultiScaleTemporalBlock


class PositionalEncoding(nn.Module):
    def __init__(self, d: int, max_len: int = 1000):
        super().__init__()
        pe  = torch.zeros(max_len, d)
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2) * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class SingingTransformer(nn.Module):
    def __init__(
        self,
        vocab: int,
        d: int = 128,
        layers: int = 4,
        heads: int = 4,
        debug: bool = False,
    ):
        super().__init__()
        self.debug = debug   # True → logging.debug を呼ぶだけ（root ロガーを使用）

        # --- embedding layers ---
        # Enhanced frame-level conditioning
        self.pitch_fc    = nn.Sequential(
            nn.Linear(1, d // 2),
            nn.LayerNorm(d // 2),
            nn.GELU(),
            nn.Linear(d // 2, d)
        )
        self.ph_emb      = nn.Embedding(vocab, d)
        
        # Enhanced multi-scale frame-level feature enhancement
        self.frame_enhancement = MultiScaleTemporalBlock(d * 2, dropout=0.1)
        
        # Add dropout for regularization
        self.dropout     = nn.Dropout(0.1)
        # Enhanced fusion network with frame-level attention
        self.fuse        = nn.Sequential(
            nn.Linear(d * 2, d * 2),
            nn.LayerNorm(d * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d * 2, d)
        )
        self.pos         = PositionalEncoding(d)
        
        # Frame-level attention for better temporal modeling
        self.frame_attention = nn.MultiheadAttention(
            embed_dim=d, num_heads=heads, batch_first=True, dropout=0.1
        )

        # --- Enhanced temporal transformer for 5-second processing ---
        self.enhanced_transformer = EnhancedTemporalTransformer(
            d_model=d,
            num_layers=max(6, layers),  # Minimum 6 layers for 5-second processing
            heads=heads,
            dropout=0.1,
            use_linear_attention=True,
            use_hierarchical=True
        )

    # ---------------------------------------------------------------- forward
    def forward(self, x: X) -> torch.Tensor:
        _, T = x["phoneme_indexes"].shape

        phoneme_indexes = x["phoneme_indexes"]
        pitch = x["pitch"]
        mel_frames = x["mel_frames"]                                    # LongTensor[B]

        if self.debug:
            logging.debug(f"[input]  ph       {phoneme_indexes.shape}")
            logging.debug(f"[input]  pitch    {pitch.shape}")
            logging.debug(f"[input]  mel_frames {mel_frames.shape}: {mel_frames}")

        # --- embedding ---
        ph       = self.ph_emb(phoneme_indexes)
        pitch    = self.pitch_fc(pitch.unsqueeze(-1))       # [B,T,d]

        if self.debug:
            logging.debug(f"[embed]  ph       {ph.shape}")
            logging.debug(f"[embed]  pitch    {pitch.shape}")

        # --- frame-level enhancement ---
        combined = torch.cat([ph, pitch], dim=-1)  # [B,T,d*2]
        
        # Apply multi-scale frame-level enhancement
        combined_conv = combined.transpose(1, 2)  # [B, d*2, T] for conv
        enhanced = self.frame_enhancement(combined_conv)
        enhanced = enhanced.transpose(1, 2)  # [B, T, d*2]
        
        # Residual connection
        combined = combined + enhanced
        
        # --- fuse ---
        h = self.fuse(combined)  # [B,T,d]
        h = self.dropout(h)  # Apply dropout for regularization
        if self.debug:
            logging.debug(f"[fuse]   h_fused  {h.shape}")

        # --- positional encoding ---
        h = self.pos(h)
        if self.debug:
            logging.debug(f"[pos]    h_pos    {h.shape}")
        
        # --- mask creation ---
        device  = h.device
        idxs    = torch.arange(T, device=device).unsqueeze(0)    # [1,T]
        valid   = idxs < mel_frames.unsqueeze(1)                    # [B,T]
        key_pad = ~valid                                         # True = pad

        if self.debug:
            logging.debug(f"[mask]   valid    {valid.shape} (mean={valid.float().mean():.3f})")
        
        # --- frame-level self-attention ---
        h_attn, _ = self.frame_attention(h, h, h, key_padding_mask=key_pad)
        h = h + h_attn  # residual connection
        if self.debug:
            logging.debug(f"[frame_attn] h_frame {h.shape}")

        # --- Enhanced temporal transformer processing for 5-second sequences ---
        h = self.enhanced_transformer(h, key_pad)
        if self.debug:
            logging.debug(f"[enhanced_transformer] h_enhanced {h.shape}")

        # --- zero-out padding ---
        h = h * valid.unsqueeze(-1)
        if self.debug:
            logging.debug(f"[output] h_clean {h.shape}")

        return h  # [B,T,d]
