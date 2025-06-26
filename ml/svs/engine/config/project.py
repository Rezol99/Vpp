from dataclasses import dataclass


@dataclass
class ProjectConfig:
    db_root = "datasets"
    table_path = "config/kana2phonemes_002_oto2lab.table"
