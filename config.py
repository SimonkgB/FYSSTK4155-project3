import torch
from dataclasses import dataclass



DURATIONS = ("0.5", "1.0", "2.0", "4.0")
N_CLASSES = 22
BLANK_STD = 0.015


@dataclass
class Config:
    h5_path: str = "./trainingsetv1d1.h5"
    meta_path: str = "./trainingset_v1d1_metadata.csv"
    seed: int = 2026
    views: tuple[str, ...] = ("1.0",)
    blank_rule: str = "any"
