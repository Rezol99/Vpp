import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_pndm import PNDMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler
from engine.config.train import TrainConfig
from engine.models.svs import Inputs, SVSModel
from engine.types.dataset import X
from engine.grouped_parts import GroupedPhonemePart

DEBUG = True


class Diffuser:
    def __init__(
        self,
        num_timesteps=TrainConfig.diffusion_steps,
        device=torch.device("cpu"),
        debug=DEBUG,
        training_mode=False,  # 新しいパラメータ
        fast_inference=False,  # 高速推論モード
    ):
        self.device = device
        self.num_timesteps = num_timesteps
        self.debug = debug
        self.training_mode = training_mode
        self.fast_inference = fast_inference
        
        if self.debug:
            self.log = print
        else:
            self.log = self.null_log

        # 学習用スケジューラー（標準的な設定）
        self.train_scheduler = DDPMScheduler(
            num_train_timesteps=num_timesteps,
            beta_schedule="squaredcos_cap_v2",
            prediction_type="v_prediction",
            clip_sample=False,  # 学習時はFalse
        )

        # 推論用スケジューラー（高品質設定）
        self.inference_scheduler = PNDMScheduler(
            num_train_timesteps=num_timesteps,
            beta_schedule="squaredcos_cap_v2",
            prediction_type="v_prediction",
            # use_karras_sigmas=True
        )
        
        # 高速推論用スケジューラー（DPM-Solver++）
        self.fast_scheduler = DPMSolverMultistepScheduler(
            num_train_timesteps=num_timesteps,
            beta_schedule="squaredcos_cap_v2",
            prediction_type="v_prediction",
            algorithm_type="dpmsolver++",
            solver_order=2,
            use_karras_sigmas=False,  # Karras sigmasを無効化
            final_sigmas_type="zero",
        )

        # モードに応じてデフォルトスケジューラーを設定
        if self.training_mode:
            self.scheduler = self.train_scheduler
        elif self.fast_inference:
            self.scheduler = self.fast_scheduler
        else:
            self.scheduler = self.inference_scheduler

    def set_training_mode(self, training_mode: bool):
        """学習モードと推論モードを切り替える"""
        self.training_mode = training_mode
        
        if self.training_mode:
            self.scheduler = self.train_scheduler
        elif self.fast_inference:
            self.scheduler = self.fast_scheduler
        else:
            self.scheduler = self.inference_scheduler
    
    def set_fast_inference(self, fast_inference: bool):
        """高速推論モードを切り替える"""
        self.fast_inference = fast_inference
        
        if not self.training_mode:
            if self.fast_inference:
                self.scheduler = self.fast_scheduler
            else:
                self.scheduler = self.inference_scheduler

    def null_log(self, *args, **kwargs):
        pass

    def add_noise(self, x_0, t):
        """
        x_0 に対して、指定時刻 t の拡散過程に基づくノイズを加える。
        学習時は必ずtrain_schedulerを使用する。
        """
        noise = torch.randn_like(x_0, device=self.device)
        # t が 0-d tensor の場合、1-d tensor に変換する
        if isinstance(t, torch.Tensor) and t.dim() == 0:
            t = t.unsqueeze(0)
        
        # 学習時は必ずtrain_schedulerを使用
        if self.training_mode:
            x_t = self.train_scheduler.add_noise(x_0, noise, t)  # type: ignore
        else:
            x_t = self.scheduler.add_noise(x_0, noise, t)  # type: ignore
        
        return x_t, noise

    def batch_add_noise(self, targets, times):
        """
        バッチごとにノイズを付加する。
        各サンプルの t は torch.Tensor として処理されるため、互換性が保たれる。
        """
        results = [self.add_noise(targets[i], times[i]) for i in range(len(targets))]
        x_ts, noises = zip(*results)
        return torch.stack(x_ts), torch.stack(noises)

    def sample(
        self,
        model: SVSModel,
        cond: Inputs,
        x_shape,
        pitch_shape,
        cfg_scale: float,
        num_inference_steps=None,
        uncond_lyric=True,
        uncond_pitch=True,
    ):
        """
        Classifier-Free Guidance 対応版サンプラー。
        条件付き入力と無条件入力を別々にモデルに通して差分を取る。
        """
        # 推論時は自動的に推論モードに切り替え
        original_mode = self.training_mode
        if self.training_mode:
            self.set_training_mode(False)
            
        # 推論ステップ数の設定
        if num_inference_steps is None:
            if self.fast_inference:
                num_inference_steps = TrainConfig.inference_steps
            else:
                num_inference_steps = self.num_timesteps

        x_cond: X = {
            "phoneme_indexes": cond["x"]["phoneme_indexes"].clone(),
            "pitch": cond["x"]["pitch"].clone(),
            "mel_frames": cond["x"]["mel_frames"].clone(),
            "next_mel_mute_duration_ms": torch.tensor(0, dtype=torch.int32)
        }

        # Get dataset statistics for normalization
        
        x_uncond: X = {
            "phoneme_indexes": cond["x"]["phoneme_indexes"].clone(),
            "pitch": cond["x"]["pitch"].clone(),
            "mel_frames": cond["x"]["mel_frames"].clone(),
            "next_mel_mute_duration_ms": torch.tensor(0, dtype=torch.int32)
        }
        uncond = GroupedPhonemePart.get_uncond(pitch_shape=pitch_shape)
        batch_size = x_shape[0]
        seq_len = x_cond["phoneme_indexes"].shape[1]

        if uncond_lyric:
            x_uncond["phoneme_indexes"] =  uncond["phoneme_indexes"].expand(batch_size, seq_len).to(device=self.device)
        if uncond_pitch:
            x_uncond["pitch"] = uncond["pitch"].expand(batch_size, seq_len).to(device=self.device)

        # 初期ノイズ
        x = torch.randn(x_shape, device=self.device)

        # スケジューラーにステップ数をセット
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps

        model.eval()
        last_valid = x.clone()

        assert timesteps is not None
        
        # CFGスケールが0の場合は条件付きのみで高速化
        use_cfg = cfg_scale > 0.0
        
        # デバッグ: ステップ数を表示
        if self.debug:
            self.log(f"Running {len(timesteps)} diffusion steps")

        for step_idx, t in enumerate(timesteps):
            t_b = t.repeat(x_shape[0]).to(self.device)
            
            if use_cfg:
                # バッチ化されたCFG: 条件付きと無条件を一度に処理
                batch_x_cond = torch.cat([x_cond["phoneme_indexes"], x_uncond["phoneme_indexes"]], dim=0)
                batch_pitch = torch.cat([x_cond["pitch"], x_uncond["pitch"]], dim=0)
                batch_mel = torch.cat([x_cond["mel_frames"], x_uncond["mel_frames"]], dim=0)
                batch_mute = torch.cat([x_cond["next_mel_mute_duration_ms"].unsqueeze(0), x_uncond["next_mel_mute_duration_ms"].unsqueeze(0)], dim=0)
                
                batch_inputs = {
                    "x": {
                        "phoneme_indexes": batch_x_cond,
                        "pitch": batch_pitch,
                        "mel_frames": batch_mel,
                        "next_mel_mute_duration_ms": batch_mute
                    },
                    "noisy_x": torch.cat([x, x], dim=0),
                    "t": torch.cat([t_b, t_b], dim=0),
                }
                
                with torch.no_grad():
                    batch_v_pred, _, _ = model(batch_inputs)
                
                # 条件付きと無条件に分割
                v_cond, v_uncond = torch.chunk(batch_v_pred, 2, dim=0)
                
                # CFG 合成
                v_pred = v_uncond + cfg_scale * (v_cond - v_uncond)
            else:
                # CFGなしの高速処理
                inputs_cond = {
                    "x":       x_cond,
                    "noisy_x": x,
                    "t":       t_b,
                }
                with torch.no_grad():
                    v_pred, _, _ = model(inputs_cond)

            # ステップ実行
            out = self.scheduler.step(v_pred, t, x).prev_sample  # type: ignore

            # NaN/Inf チェック
            if torch.isnan(out).any() or torch.isinf(out).any():
                x = last_valid
            else:
                x = out
                last_valid = x.clone()

        model.train()
        
        # 元のモードに戻す
        if original_mode:
            self.set_training_mode(True)
            
        return x