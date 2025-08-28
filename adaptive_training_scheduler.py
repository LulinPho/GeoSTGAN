"""
自适应训练调度器

基于像素级精确预测和空间分布模式稳定性的核心思路设计的训练调度方案。

核心思想：
1. 生成器主要目标：像素级精确预测
2. 判别器作用：指导空间分布模式稳定性
3. 训练阶段性：从像素精度到空间一致性的渐进式学习
"""

import torch
import numpy as np
import logging
from typing import Dict, Tuple, List

logger = logging.getLogger(__name__)


def softmax_weights(x: float, start: float, end: float, min_w: float, max_w: float, reverse: bool = False) -> float:
    """
    使用softmax型函数（sigmoid）进行权重平滑变化
    x: 当前像素精度
    start: 起始阈值
    end: 结束阈值
    min_w: 最小权重
    max_w: 最大权重
    reverse: 是否反向（如focal为True，判别器为False）
    """
    # 将x归一化到0-1区间
    if x <= start:
        norm = 0.0
    elif x >= end:
        norm = 1.0
    else:
        norm = (x - start) / (end - start)
    # 使用sigmoid进行平滑
    # 设定sigmoid的陡峭程度
    k = 10
    s = 1 / (1 + np.exp(-k * (norm - 0.5)))
    if reverse:
        s = 1 - s
    return min_w + (max_w - min_w) * s

class AdaptiveTrainingScheduler:
    """
    自适应训练调度器
    
    基于训练进度和性能指标动态调整损失权重和训练策略
    """
    
    def __init__(self):
        """
        Args:
            total_epochs: 总训练轮数
            pixel_accuracy_target: 像素精度目标阈值
            spatial_consistency_target: 空间一致性目标阈值
        """
        
        
        # 性能历史跟踪
        self.pixel_accuracy_history = []
        self.spatial_consistency_history = []
        self.loss_history = []
        
    
    def get_loss_weights(self, epoch: int, 
                        pixel_accuracy: float = None) -> Dict[str, float]:
        """
        基于像素预测精度动态计算损失权重（softmax平滑版，0.6阈值，0.5-0.5区间softmax变化）
        
        Args:
            epoch: 当前轮数
            pixel_accuracy: 当前像素精度 [0, 1]
            
        Returns:
            包含各损失项权重的字典
        """
        
        # softmax调度参数
        start_acc = 0.6  # 判别器权重启动阈值
        end_acc = 1.0    # 判别器权重最大阈值
        min_focal = 0.75
        max_focal = 1
        min_disc = 0.0
        max_disc = 0.25

        # focal loss权重：始终为0.5
        focal_weight = softmax_weights(pixel_accuracy, start_acc, end_acc, min_focal, max_focal, reverse=True)
        # 判别器权重：始终为0.5
        discriminator_weight = softmax_weights(pixel_accuracy, start_acc, end_acc, min_disc, max_disc, reverse=False)

        # 判别器patch/global损失始终为0.5-0.5
        patch_ratio = 0.5
        global_ratio = 0.5

        # 掩膜和特征权重始终为1
        mask_weight = 1.0
        feature_weight = 1.0

        weights = {
            'focal': focal_weight,
            'patch_bce': discriminator_weight * patch_ratio,
            'global_bce': discriminator_weight * global_ratio,
            'mask': mask_weight,
            'feature': feature_weight
        }
        
        # 记录权重变化
        logger.debug(f"Epoch {epoch}: pixel_acc={pixel_accuracy:.3f}, weights={weights}")
        return weights
    
    def _pixel_accuracy_to_focal_weight(self, pixel_accuracy: float) -> float:
        """
        像素精度 -> focal loss权重的softmax平滑映射函数
        始终为0.5
        """
        min_focal = 0.8
        max_focal = 1
        start_acc = 0.6
        end_acc = 1.0
        return softmax_weights(pixel_accuracy, start_acc, end_acc, min_focal, max_focal, reverse=True)
    
    def _pixel_accuracy_to_discriminator_weight(self, pixel_accuracy: float) -> float:
        """
        像素精度 -> 判别器权重的softmax平滑映射函数
        始终为0.5
        """
        min_disc = 0.0
        max_disc = 0.2
        start_acc = 0.6
        end_acc = 1.0
        return softmax_weights(pixel_accuracy, start_acc, end_acc, min_disc, max_disc, reverse=False)

def calculate_pixel_accuracy(predictions: torch.Tensor, 
                           targets: torch.Tensor,
                           mask: torch.Tensor = None) -> float:
    """
    计算像素级预测精度
    
    Args:
        predictions: 预测logits [B, C, H, W]
        targets: 目标标签 [B, C, H, W] (one-hot)
        mask: 有效像素掩膜 [B, C, H, W]
        
    Returns:
        像素精度 (0-1)
    """
    pred_classes = predictions.argmax(dim=1)  # [B, H, W]
    target_classes = targets.argmax(dim=1)    # [B, H, W]
    
    if mask is not None:
        # 只计算有效区域的精度
        valid_mask = mask.sum(dim=1) > 0  # [B, H, W]
        correct = (pred_classes == target_classes) & valid_mask
        total = valid_mask.sum()
    else:
        correct = (pred_classes == target_classes)
        total = pred_classes.numel()
    
    if total == 0:
        return 0.0
    
    return (correct.sum().float() / total.float()).item()

class PerformanceTracker:
    """性能跟踪器 - 专注于像素精度"""
    
    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.metrics_history = {
            'pixel_accuracy': [],
            'focal_loss': [],
            'discriminator_loss': []
        }
    
    def update(self, **metrics):
        """更新指标"""
        for key, value in metrics.items():
            if key in self.metrics_history:
                self.metrics_history[key].append(value)
                # 保持窗口大小
                if len(self.metrics_history[key]) > self.window_size:
                    self.metrics_history[key] = self.metrics_history[key][-self.window_size:]
    
    def get_recent_average(self, metric_name: str) -> float:
        """获取最近的平均值"""
        if metric_name not in self.metrics_history:
            return 0.0
        values = self.metrics_history[metric_name]
        return np.mean(values) if values else 0.0
    
    def is_improving(self, metric_name: str, threshold: float = 0.01) -> bool:
        """判断指标是否在改善"""
        if metric_name not in self.metrics_history:
            return False
        values = self.metrics_history[metric_name]
        if len(values) < 5:
            return True  # 数据不足，假设在改善
        
        recent_avg = np.mean(values[-3:])
        earlier_avg = np.mean(values[-6:-3])
        return recent_avg > earlier_avg + threshold
