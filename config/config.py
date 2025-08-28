"""
全局配置：路径、设备、随机种子、日志等。
"""

import os
from dataclasses import dataclass


@dataclass
class Paths:
    # 数据根目录（建议按需修改为你的数据路径）
    factors_data_dir: str = os.environ.get("FACTORS_DATA_DIR", "/root/autodl-fs/factors")
    labels_data_dir: str = os.environ.get("LABELS_DATA_DIR", "/root/autodl-fs/label_floats")

    # 产出目录
    checkpoints_dir: str = os.environ.get("CHECKPOINTS_DIR", "checkpoints")
    logs_dir: str = os.environ.get("LOGS_DIR", "logs")
    results_dir: str = os.environ.get("RESULTS_DIR", "results")


@dataclass
class Runtime:
    seed: int = int(os.environ.get("SEED", 42))
    num_workers: int = int(os.environ.get("NUM_WORKERS", 2))
    pin_memory: bool = os.environ.get("PIN_MEMORY", "1") == "1"
    prefetch_factor: int = int(os.environ.get("PREFETCH_FACTOR", 2))


PATHS = Paths()
RUNTIME = Runtime()


def ensure_dirs():
    os.makedirs(PATHS.checkpoints_dir, exist_ok=True)
    os.makedirs(PATHS.logs_dir, exist_ok=True)
    os.makedirs(PATHS.results_dir, exist_ok=True)


__all__ = ["PATHS", "RUNTIME", "ensure_dirs", "Paths", "Runtime"]


