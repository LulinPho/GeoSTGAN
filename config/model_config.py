"""
模型规模相关配置：维度、通道数、层数等。
在训练/推理阶段统一引用，避免散落在脚本中。
"""

from dataclasses import dataclass


@dataclass
class ModelSize:
    # 生成器
    hidden_channels: int = 32
    features_channels: int | None = None  # 运行期根据数据推断
    num_classes: int = 6

    # 判别器（如使用）
    disc_hidden_channels: int = 64
    disc_block_num: int = 512
    disc_use_soft_input: bool = True
    disc_temperature: float = 1.0


MODEL_SIZE = ModelSize()

__all__ = ["MODEL_SIZE", "ModelSize"]


