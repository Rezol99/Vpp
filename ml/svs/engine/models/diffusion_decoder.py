import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
from typing import Optional, Tuple, List
from engine.config.audio import AudioConfig


class SinusoidalPosEmb(nn.Module):
    """時間ステップtのための正弦波位置埋め込み"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class WaveNetResidualBlock(nn.Module):
    """WaveNetの残差ブロック with diffusion conditioning"""
    def __init__(
        self, 
        channels: int, 
        kernel_size: int, 
        dilation: int,
        context_dim: int,
        time_dim: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Dilated convolution (gated activation用に2倍のチャンネル)
        self.dilated_conv = nn.Conv1d(
            channels, 
            channels * 2, 
            kernel_size,
            dilation=dilation,
            padding=(kernel_size - 1) * dilation // 2
        )
        
        # Time conditioning
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, channels * 2)
        )
        
        # Context conditioning (cross-attention style)
        self.context_proj = nn.Conv1d(context_dim, channels * 2, 1)
        
        # Output projection
        self.output_conv = nn.Conv1d(channels, channels, 1)
        
        # Skip connection
        self.skip_conv = nn.Conv1d(channels, channels, 1)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        x: torch.Tensor,           # [B, C, T]
        context: torch.Tensor,     # [B, T, context_dim]
        time_emb: torch.Tensor,    # [B, time_dim]
        mask: Optional[torch.Tensor] = None  # [B, T]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            residual: 次の層への出力 [B, C, T]
            skip: スキップ接続への出力 [B, C, T]
        """
        # Dilated convolution
        h = self.dilated_conv(x)  # [B, 2C, T]
        
        # Add time conditioning
        time_cond = self.time_mlp(time_emb)  # [B, 2C]
        h = h + time_cond.unsqueeze(-1)
        
        # Add context conditioning
        context_t = context.transpose(1, 2)  # [B, context_dim, T]
        context_cond = self.context_proj(context_t)  # [B, 2C, T]
        h = h + context_cond
        
        # Gated activation
        h_tanh, h_sigmoid = h.chunk(2, dim=1)
        h = torch.tanh(h_tanh) * torch.sigmoid(h_sigmoid)  # [B, C, T]
        
        # Apply mask if provided
        if mask is not None:
            h = h * mask.unsqueeze(1)
        
        # Output and skip
        output = self.output_conv(h)
        skip = self.skip_conv(h)
        
        # Residual connection
        residual = output + x
        
        return residual, skip


class WaveNetStack(nn.Module):
    """WaveNetのスタック（複数の残差ブロック）"""
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        num_layers: int,
        context_dim: int,
        time_dim: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.blocks = nn.ModuleList()
        
        # Exponentially increasing dilation
        for i in range(num_layers):
            dilation = 2 ** i
            self.blocks.append(
                WaveNetResidualBlock(
                    channels=channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    context_dim=context_dim,
                    time_dim=time_dim,
                    dropout=dropout
                )
            )
    
    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        time_emb: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Returns:
            output: 最終出力 [B, C, T]
            skips: 各層からのスキップ接続のリスト
        """
        skips = []
        h = x
        
        for block in self.blocks:
            h, skip = block(h, context, time_emb, mask)
            skips.append(skip)
        
        return h, skips


class DiffusionDecoder(nn.Module):
    """
    WaveNetベースの拡散デコーダー
    
    Note:
        - 入力のメルスペクトログラムはZスコア正規化されている想定
        - エンコーダー側で音素はインデックスとして処理済み
        - 出力もZスコア正規化された空間でのv-prediction
    """
    def __init__(
        self,
        latent_dim: int = 128,                     # VAEの潜在変数の次元
        context_dim: int = 256,                    # TransformerEncoーダー出力の次元
        n_mels: int = AudioConfig.audio.num_mels,  # メルスペクトログラムの次元
        base_channels: int = 256,                  # ベースチャンネル数
        kernel_size: int = 3,                      # カーネルサイズ
        num_stacks: int = 3,                       # WaveNetスタックの数
        layers_per_stack: int = 10,                # 各スタックのレイヤー数
        dropout: float = 0.1,
        debug: bool = False
    ):
        super().__init__()
        self.debug = debug
        self.n_mels = n_mels
        self.latent_dim = latent_dim
        self.context_dim = context_dim
        self.base_channels = base_channels
        
        # 受容野の計算
        receptive_field = 1
        for _ in range(num_stacks):
            for i in range(layers_per_stack):
                receptive_field += (kernel_size - 1) * (2 ** i)
        
        if debug:
            logging.info(f"WaveNet receptive field: {receptive_field} frames")
        
        # Time embedding
        time_dim = base_channels * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(base_channels),
            nn.Linear(base_channels, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )
        
        # Initial projection
        self.input_proj = nn.Conv1d(n_mels, base_channels, 1)
        
        # WaveNet stacks - VAE潜在変数用
        self.stacks_latent = nn.ModuleList()
        for _ in range(num_stacks):
            self.stacks_latent.append(
                WaveNetStack(
                    channels=base_channels,
                    kernel_size=kernel_size,
                    num_layers=layers_per_stack,
                    context_dim=latent_dim,
                    time_dim=time_dim,
                    dropout=dropout
                )
            )
        
        # WaveNet stacks - TransformerEncoder出力用
        self.stacks_context = nn.ModuleList()
        for _ in range(num_stacks):
            self.stacks_context.append(
                WaveNetStack(
                    channels=base_channels,
                    kernel_size=kernel_size,
                    num_layers=layers_per_stack,
                    context_dim=context_dim,
                    time_dim=time_dim,
                    dropout=dropout
                )
            )
        
        # Skip connections processing - 両方のスタックからの出力を結合
        total_skips = num_stacks * layers_per_stack * 2  # 2つのスタックから
        self.skip_proj = nn.Conv1d(base_channels * total_skips, base_channels, 1)
        
        # Output layers
        self.output_layers = nn.Sequential(
            nn.Conv1d(base_channels, base_channels, 1),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv1d(base_channels, base_channels, 1),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv1d(base_channels, n_mels, 1)
        )
        
        # Context interpolation for dimension matching
        self.latent_proj = nn.Linear(latent_dim, latent_dim)
        self.context_proj = nn.Linear(context_dim, context_dim)
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """重みの初期化"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def _interpolate_context(
        self, 
        context: torch.Tensor,  # [B, T_ctx, context_dim]
        target_length: int
    ) -> torch.Tensor:
        """コンテキストをターゲット長に補間"""
        if context.size(1) == target_length:
            return context
        
        # Linear interpolation
        context_t = context.transpose(1, 2)  # [B, context_dim, T_ctx]
        interpolated = F.interpolate(
            context_t, 
            size=target_length, 
            mode='linear', 
            align_corners=False
        )
        return interpolated.transpose(1, 2)  # [B, T, context_dim]
    
    def forward(
        self, 
        latent: torch.Tensor,      # [B, T_ctx, latent_dim] VAEの潜在変数
        context: torch.Tensor,     # [B, T_ctx, context_dim] TransformerEncoder出力
        x_t: torch.Tensor,         # [B, n_mels, T] ノイズ付きメルスペクトログラム
        t: torch.Tensor,           # [B] 拡散ステップ
        mel_frames: torch.Tensor   # [B] 有効フレーム数
    ) -> torch.Tensor:
        """
        v-predictionを生成
        
        Args:
            latent: VAEの潜在変数 [B, T_ctx, latent_dim]
            context: TransformerEncoder出力 [B, T_ctx, context_dim]
                    （音素インデックスは既に埋め込みベクトルに変換済み）
            x_t: ノイズ付きメルスペクトログラム [B, n_mels, T]
                 （Zスコア正規化済み: mean=0, std=1）
            t: 拡散ステップ [B]
            mel_frames: 各バッチの有効フレーム数 [B]
            
        Returns:
            v_pred: v-prediction [B, n_mels, T]
                    （Zスコア正規化空間でのv-prediction）
        """
        B, _, T = x_t.shape
        
        if self.debug:
            logging.debug(f"[WaveNetDecoder] input x_t: {x_t.shape}")
            logging.debug(f"[WaveNetDecoder] latent: {latent.shape}")
            logging.debug(f"[WaveNetDecoder] context: {context.shape}")
            logging.debug(f"[WaveNetDecoder] t: {t.shape}")
            logging.debug(f"[WaveNetDecoder] mel_frames: {mel_frames}")
            
            # Zスコア正規化の確認（デバッグ時）
            valid_data = x_t[mel_frames.unsqueeze(1).unsqueeze(1) > 
                            torch.arange(T, device=x_t.device).unsqueeze(0).unsqueeze(0)]
            if valid_data.numel() > 0:
                logging.debug(f"[WaveNetDecoder] x_t stats - mean: {valid_data.mean():.4f}, "
                            f"std: {valid_data.std():.4f}")
        
        # Create mask from mel_frames
        device = x_t.device
        frame_indices = torch.arange(T, device=device).unsqueeze(0)  # [1, T]
        mask = frame_indices < mel_frames.unsqueeze(1)  # [B, T]
        
        # Interpolate both latent and context to match mel length
        latent = self._interpolate_context(latent, T)
        latent = self.latent_proj(latent)  # [B, T, latent_dim]
        
        context = self._interpolate_context(context, T)
        context = self.context_proj(context)  # [B, T, context_dim]
        
        # Time embedding
        time_emb = self.time_mlp(t)  # [B, time_dim]
        
        # Initial projection
        h = self.input_proj(x_t)  # [B, base_channels, T]
        
        if self.debug:
            logging.debug(f"[WaveNetDecoder] after input_proj: {h.shape}")
        
        # Apply WaveNet stacks - VAE潜在変数用
        all_skips = []
        h_latent = h
        for i, stack in enumerate(self.stacks_latent):
            h_latent, skips = stack(h_latent, latent, time_emb, mask)
            all_skips.extend(skips)
            
            if self.debug:
                logging.debug(f"[WaveNetDecoder] after latent stack {i}: {h_latent.shape}")
        
        # Apply WaveNet stacks - TransformerEncoder出力用  
        h_context = h
        for i, stack in enumerate(self.stacks_context):
            h_context, skips = stack(h_context, context, time_emb, mask)
            all_skips.extend(skips)
            
            if self.debug:
                logging.debug(f"[WaveNetDecoder] after context stack {i}: {h_context.shape}")
        
        # Combine all skip connections from both stacks
        combined_skips = torch.cat(all_skips, dim=1)  # [B, base_channels * total_skips, T]
        h = self.skip_proj(combined_skips)  # [B, base_channels, T]
        
        if self.debug:
            logging.debug(f"[WaveNetDecoder] after skip projection: {h.shape}")
        
        # Final output layers
        v_pred = self.output_layers(h)  # [B, n_mels, T]
        
        # Apply mask to ensure padding is zero
        v_pred = v_pred * mask.unsqueeze(1)  # [B, n_mels, T]
        
        if self.debug:
            logging.debug(f"[WaveNetDecoder] output v_pred: {v_pred.shape}")
        
        return v_pred