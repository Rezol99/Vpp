import glob
import logging
import os
import re
from typing import TypedDict

import torch
import torch.nn as nn

from engine.models.diffusion_decoder import DiffusionDecoder
from engine.models.transformer import SingingTransformer
from engine.models.vae import VAE_FrameLevel
from engine.types.dataset import X
from engine.utils.time import get_time_str


class Inputs(TypedDict):
    x: X
    noisy_x: torch.Tensor
    t: torch.Tensor


class SVSModel(nn.Module):
    def __init__(
        self,
        vocab: int,
        d: int = 256,
        layers=4,
        prediction: bool = False,
        base_ch=512,
        z_dim: int = 128,
        debug: bool = False,
    ):
        super().__init__()
        self.debug = debug

        # --- sub-modules -----------------------------------------------------
        self.enc = SingingTransformer(vocab, d, layers=layers, debug=debug)
        
        # VAEを追加：エンコーダー出力を潜在空間にマッピング
        self.vae = VAE_FrameLevel(d_model=d, z_dim=z_dim, debug=debug)
        
        # DiffusionDecoder：VAEの潜在変数とTransformerEncoder出力を使用
        self.dec = DiffusionDecoder(
            latent_dim=z_dim,   # VAEの潜在変数の次元
            context_dim=d,      # TransformerEncoderの出力次元
            debug=debug
        )
        self.prediction = prediction

        # --- checkpoint setup -----------------------------------------------
        time_str = get_time_str()
        self._ckpt_dir = os.path.join("./_checkpoints", time_str)
        if not prediction:
            os.makedirs(self._ckpt_dir, exist_ok=True)

        self._log_file       = os.path.join(self._ckpt_dir, "train.log")
        self._checkpoint_file = os.path.join(self._ckpt_dir, "checkpoint.pth")

    # ------------------------------------------------------------------ fwd
    def forward(self, inputs: Inputs):
        x     = inputs["x"]
        x_t   = inputs["noisy_x"]
        t     = inputs["t"]

        if self.debug:
            logging.debug(f"[SVS] keys={list(x.keys())} "
                          f"x_t={tuple(x_t.shape)} t={tuple(t.shape)}")

        # encoder
        h = self.enc(x)                           # [B,T,d]

        # VAE - エンコーダ出力を潜在空間にマッピング
        z, mu, logvar = self.vae(h, x["mel_frames"])  # [B,T,z_dim]

        # diffusion decoder - VAEの潜在変数とTransformerEncoder出力を使用
        v_pred = self.dec(z, h, x_t, t, x["mel_frames"])  # [B,n_mels,T]

        if self.debug:
            logging.debug(f"[SVS] h={tuple(h.shape)} z={tuple(z.shape)} v_pred={tuple(v_pred.shape)}")
        
        return v_pred, mu, logvar

    # ---------------------------------------------------------------- utils
    def log(self, msg: str):
        logging.info(msg)
        with open(self._log_file, "a") as f:
            f.write(msg + "\n")

    def save(self):
        torch.save({"model_state_dict": self.state_dict()}, self._checkpoint_file)

    def save_epoch(self, epoch: int):
        epoch_path = self._checkpoint_file.replace(".pth", f"_epoch{epoch}.pth")
        torch.save({"model_state_dict": self.state_dict()}, epoch_path)
    
    def load(self, checkpoint, map_location=None):
        state = torch.load(checkpoint, map_location=map_location)
        self.load_state_dict(state["model_state_dict"])
        return checkpoint

    # ----------------------------------------------------------- load latest
    def load_latest(self, map_location=None):
        dirs = [d for d in glob.glob("./_checkpoints/*") if os.path.isdir(d)]
        if not dirs:
            raise FileNotFoundError("No checkpoint directories found.")

        latest_dir = max(dirs)
        ckpt = os.path.join(latest_dir, "checkpoint.pth")

        if not os.path.exists(ckpt):
            # fall back to epoch-specific ckpts
            epoch_files = glob.glob(os.path.join(latest_dir, "checkpoint_epoch*.pth"))
            if not epoch_files:
                raise FileNotFoundError(f"No checkpoint files in {latest_dir}")

            def epoch_num(p: str) -> int:
                m = re.search(r"epoch(\d+)", p)
                return int(m.group(1)) if m else -1

            # Sort by epoch number in descending order and try to load
            epoch_files.sort(key=epoch_num, reverse=True)
            
            for ckpt in epoch_files:
                try:
                    # Test if checkpoint can be loaded
                    torch.load(ckpt, map_location=map_location)
                    logging.info(f"Loading checkpoint: {ckpt}")
                    return self.load(ckpt, map_location=map_location)
                except Exception as e:
                    logging.warning(f"Failed to load {ckpt}: {e}")
                    continue
            
            raise FileNotFoundError(f"No valid checkpoint files in {latest_dir}")

        logging.info(f"Loading checkpoint: {ckpt}")
        return self.load(ckpt, map_location=map_location)