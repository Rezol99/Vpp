import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import GradScaler, autocast # type: ignore
from torch.utils.data import DataLoader, random_split
from tqdm.auto import tqdm
from typing import Tuple, Optional, cast
import matplotlib.pylab as plt
import numpy as np
import logging
from torch.utils.tensorboard import SummaryWriter # type: ignore
import os
from datetime import datetime
import math

from engine.config.train import TrainConfig
from engine.datasets.vocal import VocalDataset
from engine.datasets_stats import DatasetStats
from engine.diffuser import Diffuser
from engine.models.svs import Inputs, SVSModel
from engine.phoneme_indexes import PhonemeIndexes
from engine.grouped_parts import GroupedParts
from engine.types.dataset import X
from engine.types.grouped_part import GroupedPhonemePart
from engine.tensorboard import TensorBoardLogger
from engine.losses import VocalDiffusionLoss


# カスタム学習率スケジューラ
class WarmupCosineScheduler:
    """
    Warmup + Cosine Annealing スケジューラ
    拡散モデルでよく使用される組み合わせ
    """
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=0.0):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']
        self._step = 0
    
    def step(self):
        self._step += 1
        lr = self._get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def _get_lr(self):
        if self._step < self.warmup_steps:
            # Linear warmup
            return self.base_lr * self._step / self.warmup_steps
        else:
            # Cosine annealing
            progress = (self._step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            return self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
    
    def get_last_lr(self):
        return [self._get_lr()]


class LinearWarmupExponentialDecay:
    """
    Linear Warmup + Exponential Decay
    Stable Diffusionなどで使用
    """
    def __init__(self, optimizer, warmup_steps, decay_rate=0.95, decay_steps=10000):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        self.base_lr = optimizer.param_groups[0]['lr']
        self._step = 0
    
    def step(self):
        self._step += 1
        lr = self._get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def _get_lr(self):
        if self._step < self.warmup_steps:
            # Linear warmup
            return self.base_lr * self._step / self.warmup_steps
        else:
            # Exponential decay
            decay_factor = self.decay_rate ** ((self._step - self.warmup_steps) / self.decay_steps)
            return self.base_lr * decay_factor
    
    def get_last_lr(self):
        return [self._get_lr()]


class PolynomialDecayScheduler:
    """
    Polynomial Decay with Warmup
    DALL-E 2などで使用
    """
    def __init__(self, optimizer, warmup_steps, total_steps, power=1.0, min_lr=0):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.power = power
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']
        self._step = 0
    
    def step(self):
        self._step += 1
        lr = self._get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def _get_lr(self):
        if self._step < self.warmup_steps:
            # Linear warmup
            return self.base_lr * self._step / self.warmup_steps
        else:
            # Polynomial decay
            progress = (self._step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            decay_factor = (1 - progress) ** self.power
            return self.min_lr + (self.base_lr - self.min_lr) * decay_factor
    
    def get_last_lr(self):
        return [self._get_lr()]



def run_epoch(
    epoch: int,
    model: SVSModel,
    dataset_stats: DatasetStats,
    dataloader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[object],  # カスタムスケジューラも受け入れる
    diffusion: Diffuser,
    device: torch.device,
    writer: Optional[SummaryWriter],
    global_step: int,
    mode: str = "train",  # 'train' or 'test'
) -> Tuple[float, float, float, int]:
    """
    1エポック分の学習と評価を行い、
    平均スペクトル損失と平均KLダイバージェンスを返す。
    """
    is_training = mode == "train"
    if is_training:
        model.train()
    else:
        model.eval()
    scaler = GradScaler('cuda', enabled=is_training)

    total_spectral_loss = 0.0
    total_kl_divergence = 0.0
    
    # Initialize v-prediction loss function
    v_loss_fn = VocalDiffusionLoss(prediction_type="v")

    with torch.set_grad_enabled(is_training):
        for i, batch in enumerate(tqdm(dataloader, disable=not is_training)):
            x_batch, y_batch = batch
            x: X = cast(X, {k: v.to(device) for k, v in x_batch.items()})

            target_mel = y_batch["mel"].to(device)   # [B, n_mels, T]
            mask       = y_batch["mask"].to(device)  # [B, n_mels, T]

            batch_size = target_mel.size(0)

            timesteps = torch.randint(1, TrainConfig.diffusion_steps, (batch_size,), device=device)

            noisy_mel, noise = diffusion.batch_add_noise(target_mel, timesteps)
            noisy_mel = noisy_mel * mask

            x_cfg: X = {k: v.clone() for k, v in x.items()} # type: ignore

            mask_uncond = torch.rand(batch_size, device=device) < TrainConfig.uncond_prob

            uncond = GroupedPhonemePart.get_uncond(x["pitch"].size())

            # Apply separate uncond probabilities for each condition
            mask_lyric = torch.rand(batch_size, device=device) < TrainConfig.uncond_lyric_prob
            mask_pitch = torch.rand(batch_size, device=device) < TrainConfig.uncond_pitch_prob
            
            # Apply global uncond mask OR individual masks
            phoneme_mask = mask_uncond | mask_lyric
            pitch_mask = mask_uncond | mask_pitch

            # Expand uncond tensors to match sequence length
            seq_len = x_cfg["phoneme_indexes"].shape[1]
            uncond_phoneme = uncond["phoneme_indexes"].expand(batch_size, seq_len).to(device=device)
            uncond_pitch = uncond["pitch"].expand(batch_size, seq_len).to(device=device)
            
            # Apply masks
            x_cfg["phoneme_indexes"][phoneme_mask] = uncond_phoneme[phoneme_mask]
            x_cfg["pitch"][pitch_mask] = uncond_pitch[pitch_mask]

            inputs: Inputs = {"x": x_cfg, "noisy_x": noisy_mel, "t": timesteps}

            with autocast('cuda', enabled=is_training):
                pred_v, mu, logvar = model(inputs)
                
                # v-targetの計算
                alphas_cumprod = diffusion.scheduler.alphas_cumprod[timesteps.cpu()].to(device)
                alpha_t = alphas_cumprod.sqrt()
                sigma_t = (1 - alphas_cumprod).sqrt()
                
                # v_target = α_t * ε - σ_t * x_0
                v_target = alpha_t.view(-1, 1, 1) * noise - sigma_t.view(-1, 1, 1) * target_mel
                
                # v-prediction loss（マスク適用）
                valid_pixels = mask.sum()
                v_loss = F.mse_loss(pred_v * mask, v_target * mask, reduction="sum") / (valid_pixels + 1e-8)

                # KL divergence with Free Bits
                kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())  # [B, latent_dim]
                
                # Free bits implementation
                free_bits = TrainConfig.free_bits
                kl_per_dim_mean = kl_per_dim.mean(dim=0)  # Average over batch: [latent_dim]
                
                # Apply free bits: only penalize dimensions above the threshold
                kl_masked = torch.clamp(kl_per_dim_mean - free_bits, min=0.0)
                kl_div = kl_masked.sum()  # Sum over all dimensions

                # Adaptive beta scheduling
                beta = min((epoch+1) / TrainConfig.kl_anneal_epochs, 1.0) * TrainConfig.kl_weight

                # Total loss
                loss = v_loss + beta * kl_div
                
                # Log to TensorBoard
                if writer and is_training and i % 10 == 0:
                    writer.add_scalar(f'{mode}/v_loss', v_loss.item(), global_step)
                    writer.add_scalar(f'{mode}/kl_divergence', kl_div.item(), global_step)
                    writer.add_scalar(f'{mode}/beta', beta, global_step)
                    writer.add_scalar(f'{mode}/total_loss', loss.item(), global_step)
                    
                    # # データ正規化状況の可視化
                    # valid_target = target_mel[mask > 0]
                    # if valid_target.numel() > 0:
                    #     writer.add_scalar(f'{mode}/data/target_mean', valid_target.mean(), global_step)
                    #     writer.add_scalar(f'{mode}/data/target_std', valid_target.std(), global_step)
                    #     writer.add_scalar(f'{mode}/data/target_min', valid_target.min(), global_step)
                    #     writer.add_scalar(f'{mode}/data/target_max', valid_target.max(), global_step)
                    #     writer.add_histogram(f'{mode}/data/target_distribution', valid_target, global_step)
                    
                    # # 入力データの正規化状況も可視化
                    # pitch_data = x["pitch"].flatten()
                    # phoneme_data = x["phoneme_indexes"].flatten()
                    
                    # # Pitchの統計情報（0以外の値のみ）
                    # valid_pitch = pitch_data[pitch_data > 0]
                    # if valid_pitch.numel() > 0:
                    #     writer.add_scalar(f'{mode}/input/pitch_mean', valid_pitch.mean(), global_step)
                    #     writer.add_scalar(f'{mode}/input/pitch_std', valid_pitch.std(), global_step)
                    #     writer.add_scalar(f'{mode}/input/pitch_min', valid_pitch.min(), global_step)
                    #     writer.add_scalar(f'{mode}/input/pitch_max', valid_pitch.max(), global_step)
                    #     writer.add_histogram(f'{mode}/input/pitch_distribution', valid_pitch, global_step)
                    
                    # # Phoneme indexesの分布（パディング除く）
                    # valid_phoneme = phoneme_data[phoneme_data > 0]
                    # if valid_phoneme.numel() > 0:
                    #     writer.add_scalar(f'{mode}/input/phoneme_unique_count', len(torch.unique(valid_phoneme)), global_step)
                    #     writer.add_histogram(f'{mode}/input/phoneme_distribution', valid_phoneme, global_step)
                    
                    # # その他の入力データの統計
                    # mel_frames = x["mel_frames"]
                    # mute_duration = x["next_mel_mute_duration_ms"]
                    # writer.add_scalar(f'{mode}/input/mel_frames_mean', mel_frames.float().mean(), global_step)
                    # writer.add_scalar(f'{mode}/input/mute_duration_mean', mute_duration.float().mean(), global_step)
                    
                    if i % 100 == 0:
                        with torch.no_grad():
                            # v-predictionからx_0を復元
                            alpha_sq = alpha_t.view(-1, 1, 1) ** 2
                            sigma_sq = sigma_t.view(-1, 1, 1) ** 2
                            pred_x0 = (alpha_t.view(-1, 1, 1) * noisy_mel - 
                                    sigma_t.view(-1, 1, 1) * pred_v) / (alpha_sq + sigma_sq)
                            
                            # シンプルなマスク適用（パディング部分を0に）
                            pred_x0 = pred_x0 * mask
                            
                            # 値の範囲を記録（マスクされた部分のみ）
                            valid_target = target_mel[mask > 0]
                            valid_pred = pred_x0[mask > 0]
                            
                            if valid_target.numel() > 0:  # 有効な要素がある場合のみ
                                writer.add_scalar(f'{mode}/target_min', valid_target.min(), global_step)
                                writer.add_scalar(f'{mode}/target_max', valid_target.max(), global_step)
                                writer.add_scalar(f'{mode}/pred_min', valid_pred.min(), global_step)
                                writer.add_scalar(f'{mode}/pred_max', valid_pred.max(), global_step)
                        
                        # Log spectrograms
                        tb_logger = TensorBoardLogger(writer)
                        tb_logger.log_mel_comparison(
                            f'{mode}/mel_step_{timesteps[0].item()}', 
                            target_mel[0],  # マスクは別途適用されるはず
                            pred_x0[0], 
                            mask[0], 
                            global_step, 
                            timesteps[0].item() # type: ignore
                        )
                        tb_logger.log_spectrogram(f'{mode}/noisy_mel_step_{timesteps[0].item()}', noisy_mel[0], global_step)
                
                global_step += 1

            # パラメータ更新
            if is_training and optimizer:
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                
                # 勾配クリッピング（v-predictionでは重要）
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                scaler.step(optimizer)
                scaler.update()
                
                # スケジューラのステップ（バッチごと）
                if scheduler is not None:
                    scheduler.step() # type: ignore

            total_spectral_loss += loss.item()
            total_kl_divergence += kl_div.item()

    avg_spectral = total_spectral_loss / len(dataloader)
    avg_kl = total_kl_divergence / len(dataloader)
    return avg_spectral, avg_kl, beta, global_step


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_stats = DatasetStats()
    phoneme_indexes = PhonemeIndexes()
    grouped_parts = GroupedParts()
    dataset = VocalDataset(grouped_parts, dataset_stats, phoneme_indexes)


    train_dataset, test_dataset = random_split(
        dataset, [int(len(dataset) * 0.9), len(dataset) - int(len(dataset) * 0.9)]
    )


    train_loader = DataLoader(
        train_dataset,
        batch_size = TrainConfig.batch_size,
        shuffle    = True,
        num_workers=2,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size = TrainConfig.batch_size,
        shuffle    = False,
        num_workers=2,
        pin_memory=True,
    )

    model = SVSModel(TrainConfig.phoneme_emb_dim).to(device)
    model = cast(SVSModel, torch.compile(model))
    
    optimizer = optim.AdamW(model.parameters(), lr=TrainConfig.lr, weight_decay=1e-5)
    
    # 総ステップ数の計算
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * TrainConfig.epoch
    warmup_steps = int(0.1 * total_steps)  # 全体の10%をwarmupに使用
    
    # スケジューラの選択（以下から1つ選択）
    
    # Option 1: Warmup + Cosine Annealing（推奨）
    scheduler = WarmupCosineScheduler(
        optimizer, 
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        min_lr=TrainConfig.eta_min
    )
    
    # Option 2: Linear Warmup + Exponential Decay
    # scheduler = LinearWarmupExponentialDecay(
    #     optimizer,
    #     warmup_steps=warmup_steps,
    #     decay_rate=0.98,  # 各decay_stepsごとに0.98倍
    #     decay_steps=1000  # 1000ステップごとに減衰
    # )
    
    # Option 3: Polynomial Decay with Warmup
    # scheduler = PolynomialDecayScheduler(
    #     optimizer,
    #     warmup_steps=warmup_steps,
    #     total_steps=total_steps,
    #     power=1.0,  # 1.0=linear, 2.0=quadratic
    #     min_lr=TrainConfig.eta_min
    # )
    
    # Option 4: PyTorchの既存スケジューラ（OneCycleLR）
    # scheduler = optim.lr_scheduler.OneCycleLR(
    #     optimizer,
    #     max_lr=TrainConfig.lr * 10,  # 最大学習率
    #     total_steps=total_steps,
    #     pct_start=0.1,  # warmupの割合
    #     anneal_strategy='cos',
    #     div_factor=10,  # 初期学習率 = max_lr / div_factor
    #     final_div_factor=100  # 最終学習率 = max_lr / final_div_factor
    # )
    
    diffuser = Diffuser(device=device, training_mode=True)
    
    # Setup TensorBoard
    log_dir = os.path.join("runs", f"engine_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard logging to: {log_dir}")
    print(f"To view from Windows Chrome, run: tensorboard --logdir={os.path.abspath(log_dir)} --bind_all")
    print("Then access from Windows at: http://localhost:6006")
    
    global_step = 0

    for epoch in range(TrainConfig.epoch):
        tr, tr_kl, tr_beta, global_step = run_epoch(
            epoch, model, dataset_stats, train_loader, optimizer, scheduler, diffuser, device, writer, global_step, "train"
        )
        tr_wkl = tr_beta * tr_kl
        val, val_kl, val_beta, global_step = run_epoch(
            epoch, model, dataset_stats, test_loader, None, None, diffuser, device, writer, global_step, "test"
        )
        val_wkl = val_beta * val_kl
        current_lr = scheduler.get_last_lr()[0]
        
        # Log epoch metrics
        writer.add_scalar('epoch/train_loss', tr, epoch)
        writer.add_scalar('epoch/train_kl', tr_kl, epoch)
        writer.add_scalar('epoch/val_loss', val, epoch)
        writer.add_scalar('epoch/val_kl', val_kl, epoch)
        writer.add_scalar('epoch/learning_rate', current_lr, epoch)
        
        # Log model weights histograms and gradients
        if epoch % 10 == 0:
            for name, param in model.named_parameters():
                writer.add_histogram(f'weights/{name}', param, epoch)
                if param.grad is not None:
                    writer.add_histogram(f'gradients/{name}', param.grad, epoch)
        
        model.log(
            f"Epoch: {epoch+1:2d} lr: {current_lr:.10f}  train {tr:.10f}/{tr_wkl:.10f}  val {val:.10f}/{val_wkl:.10f}"
        )
        
        # エポックごとのスケジューラステップは削除（バッチごとに更新するため）
        # scheduler.step()
        
        model.save_epoch(epoch + 1)
    
    writer.close()
    model.save()