import json
from dataclasses import dataclass
from typing import ClassVar, List, Optional, cast


@dataclass
class _UpsampleConfig:
    rates: List[int]
    kernel_sizes: List[int]
    initial_channel: int


@dataclass
class _ResblockConfig:
    kernel_sizes: List[int]
    dilation_sizes: List[List[int]]


@dataclass
class _AudioConfig:
    segment_size: int
    num_mels: int
    num_freq: int
    n_fft: int
    hop_size: int
    win_size: int
    sampling_rate: int
    fmin: int
    fmax: int
    fmax_for_loss: Optional[int] = None


@dataclass
class _DistConfig:
    dist_backend: str
    dist_url: str
    world_size: int


class AudioConfig:
    # 初期化後には必ず値が入るので、定義時点ではNone型であることを明示しない
    json_path: ClassVar[str] = ""
    resblock: ClassVar[str]
    num_gpus: ClassVar[int]
    batch_size: ClassVar[int]
    learning_rate: ClassVar[float]
    adam_b1: ClassVar[float]
    adam_b2: ClassVar[float]
    lr_decay: ClassVar[float]
    seed: ClassVar[int]
    upsample: ClassVar[_UpsampleConfig]
    resblock_config: ClassVar[_ResblockConfig]
    audio: ClassVar[_AudioConfig]
    num_workers: ClassVar[int]
    dist_config: ClassVar[_DistConfig]

    # IDEの型チェックのために初期値は設定しておくが実行時には上書きされる
    # ダミー値を入れておくことで「Noneかもしれない」という警告を回避
    audio = cast(_AudioConfig, None)
    upsample = cast(_UpsampleConfig, None)
    resblock_config = cast(_ResblockConfig, None)
    dist_config = cast(_DistConfig, None)

    @classmethod
    def load_config(cls, json_path: str) -> None:
        """JSONファイルから設定を読み込み、クラス変数に設定する"""
        cls.json_path = json_path

        with open(json_path, "r") as f:
            data = json.load(f)

        cls.resblock = data["resblock"]
        cls.num_gpus = data["num_gpus"]
        cls.batch_size = data["batch_size"]
        cls.learning_rate = data["learning_rate"]
        cls.adam_b1 = data["adam_b1"]
        cls.adam_b2 = data["adam_b2"]
        cls.lr_decay = data["lr_decay"]
        cls.seed = data["seed"]

        cls.upsample = _UpsampleConfig(
            rates=data["upsample_rates"],
            kernel_sizes=data["upsample_kernel_sizes"],
            initial_channel=data["upsample_initial_channel"],
        )

        cls.resblock_config = _ResblockConfig(
            kernel_sizes=data["resblock_kernel_sizes"],
            dilation_sizes=data["resblock_dilation_sizes"],
        )

        cls.audio = _AudioConfig(
            segment_size=data["segment_size"],
            num_mels=data["num_mels"],
            num_freq=data["num_freq"],
            n_fft=data["n_fft"],
            hop_size=data["hop_size"],
            win_size=data["win_size"],
            sampling_rate=data["sampling_rate"],
            fmin=data["fmin"],
            fmax=data["fmax"],
            fmax_for_loss=data.get("fmax_for_loss"),
        )

        cls.num_workers = data["num_workers"]

        cls.dist_config = _DistConfig(
            dist_backend=data["dist_config"]["dist_backend"],
            dist_url=data["dist_config"]["dist_url"],
            world_size=data["dist_config"]["world_size"],
        )


AudioConfig.load_config("./engine/hifi_gan/config.json")
