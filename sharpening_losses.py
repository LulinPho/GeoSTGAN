"""
锐化损失函数模块
专门用于鼓励模型在每个像元位置产生更加明确和锐利的类别预测
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import numpy as np

logger = logging.getLogger(__name__)

def entropy_sharpening_loss(probs, temperature=1.0, valid_pixel_mask=None, target_entropy_ratio=0.1):
    """
    熵锐化损失函数 - 通过最小化预测分布的熵来鼓励更锐利的预测
    
    Args:
        probs: 预测概率分布，形状 (C, H, W) 或 (B, C, H, W)
        temperature: 温度参数，越小越锐化 (默认1.0)
        valid_pixel_mask: 有效像元掩膜，形状 (H, W) 或 (B, H, W)
        target_entropy_ratio: 目标熵比率 (0-1)，越小越锐化
    
    Returns:
        loss: 熵锐化损失（标量）
        avg_entropy: 平均熵值（用于监控）
    """
    if probs.dim() == 3:
        probs = probs.unsqueeze(0)  # 添加batch维度
    if valid_pixel_mask is not None and valid_pixel_mask.dim() == 2:
        valid_pixel_mask = valid_pixel_mask.unsqueeze(0)
    
    B, C, H, W = probs.shape
    
    # 应用温度缩放
    if temperature != 1.0:
        # 转换为logits再应用温度
        logits = torch.log(probs.clamp(min=1e-8)) 
        scaled_logits = logits / temperature
        probs = F.softmax(scaled_logits, dim=1)
    
    # 计算每个像元的熵 H(p) = -Σp_i*log(p_i)
    log_probs = torch.log(probs.clamp(min=1e-8))
    entropy_per_pixel = -(probs * log_probs).sum(dim=1)  # (B, H, W)
    
    # 目标熵：完全随机分布的熵 * target_entropy_ratio
    max_entropy = -torch.log(torch.tensor(1.0 / C))  # log(C)
    target_entropy = max_entropy * target_entropy_ratio
    
    if valid_pixel_mask is not None:
        valid_mask = (valid_pixel_mask > 0.5).float()
        
        # 仅在有效像元上计算损失
        valid_entropy = entropy_per_pixel * valid_mask
        loss = (valid_entropy - target_entropy).clamp(min=0).sum() / (valid_mask.sum() + 1e-8)
        
        # 监控平均熵
        avg_entropy = valid_entropy.sum() / (valid_mask.sum() + 1e-8)
    else:
        loss = (entropy_per_pixel - target_entropy).clamp(min=0).mean()
        avg_entropy = entropy_per_pixel.mean()
    
    return loss, avg_entropy

def confidence_maximization_loss(probs, confidence_threshold=0.7, valid_pixel_mask=None):
    """
    置信度最大化损失函数 - 鼓励模型产生高置信度的预测
    
    Args:
        probs: 预测概率分布，形状 (C, H, W) 或 (B, C, H, W)
        confidence_threshold: 置信度阈值，鼓励最大概率超过此值
        valid_pixel_mask: 有效像元掩膜，形状 (H, W) 或 (B, H, W)
    
    Returns:
        loss: 置信度最大化损失（标量）
        avg_max_prob: 平均最大概率（用于监控）
    """
    if probs.dim() == 3:
        probs = probs.unsqueeze(0)
    if valid_pixel_mask is not None and valid_pixel_mask.dim() == 2:
        valid_pixel_mask = valid_pixel_mask.unsqueeze(0)
    
    # 获取每个像元的最大概率
    max_probs, _ = probs.max(dim=1)  # (B, H, W)
    
    # 计算距离目标置信度的差距
    confidence_gap = confidence_threshold - max_probs
    confidence_loss = confidence_gap.clamp(min=0)  # 只惩罚低于阈值的情况
    
    if valid_pixel_mask is not None:
        valid_mask = (valid_pixel_mask > 0.5).float()
        
        valid_loss = confidence_loss * valid_mask
        loss = valid_loss.sum() / (valid_mask.sum() + 1e-8)
        
        # 监控指标
        valid_max_probs = max_probs * valid_mask
        avg_max_prob = valid_max_probs.sum() / (valid_mask.sum() + 1e-8)
    else:
        loss = confidence_loss.mean()
        avg_max_prob = max_probs.mean()
    
    return loss, avg_max_prob

def gini_coefficient_loss(probs, target_gini=0.8, valid_pixel_mask=None):
    """
    基尼系数损失函数 - 使用基尼系数衡量分布的不均匀程度，鼓励单一类型占主导
    
    Args:
        probs: 预测概率分布，形状 (C, H, W) 或 (B, C, H, W)
        target_gini: 目标基尼系数 (0-1)，越高越不均匀
        valid_pixel_mask: 有效像元掩膜
    
    Returns:
        loss: 基尼系数损失（标量）
        avg_gini: 平均基尼系数（用于监控）
    """
    if probs.dim() == 3:
        probs = probs.unsqueeze(0)
    if valid_pixel_mask is not None and valid_pixel_mask.dim() == 2:
        valid_pixel_mask = valid_pixel_mask.unsqueeze(0)
    
    B, C, H, W = probs.shape
    
    # 计算每个像元的基尼系数
    # Gini = 1 - Σ(p_i^2)，其中p_i是各类别概率
    prob_squares = probs ** 2
    gini_per_pixel = 1.0 - prob_squares.sum(dim=1)  # (B, H, W)
    
    # 计算与目标基尼系数的差距
    gini_gap = target_gini - gini_per_pixel
    gini_loss = gini_gap.clamp(min=0)  # 只惩罚基尼系数低于目标的情况
    
    if valid_pixel_mask is not None:
        valid_mask = (valid_pixel_mask > 0.5).float()
        
        valid_loss = gini_loss * valid_mask
        loss = valid_loss.sum() / (valid_mask.sum() + 1e-8)
        
        # 监控指标
        valid_gini = gini_per_pixel * valid_mask
        avg_gini = valid_gini.sum() / (valid_mask.sum() + 1e-8)
    else:
        loss = gini_loss.mean()
        avg_gini = gini_per_pixel.mean()
    
    return loss, avg_gini

def dominant_class_loss(probs, dominance_ratio=0.7, valid_pixel_mask=None):
    """
    主导类损失函数 - 直接鼓励最大概率类别占据指定比例以上
    
    Args:
        probs: 预测概率分布，形状 (C, H, W) 或 (B, C, H, W)
        dominance_ratio: 主导比例阈值，最大概率应超过此值
        valid_pixel_mask: 有效像元掩膜
    
    Returns:
        loss: 主导类损失（标量）
        dominance_stats: 主导性统计信息
    """
    if probs.dim() == 3:
        probs = probs.unsqueeze(0)
    if valid_pixel_mask is not None and valid_pixel_mask.dim() == 2:
        valid_pixel_mask = valid_pixel_mask.unsqueeze(0)
    
    # 获取最大概率和次大概率
    sorted_probs, _ = torch.sort(probs, dim=1, descending=True)
    max_prob = sorted_probs[:, 0]  # (B, H, W)
    second_max_prob = sorted_probs[:, 1]  # (B, H, W)
    
    # 计算主导性：最大概率 vs 次大概率的差距
    dominance = max_prob - second_max_prob  # (B, H, W)
    
    # 鼓励主导性超过阈值
    target_dominance = dominance_ratio - (1.0 - dominance_ratio)  # 目标差距
    dominance_gap = target_dominance - dominance
    dominance_loss = dominance_gap.clamp(min=0)
    
    if valid_pixel_mask is not None:
        valid_mask = (valid_pixel_mask > 0.5).float()
        
        valid_loss = dominance_loss * valid_mask
        loss = valid_loss.sum() / (valid_mask.sum() + 1e-8)
        
        # 统计信息
        valid_dominance = dominance * valid_mask
        avg_dominance = valid_dominance.sum() / (valid_mask.sum() + 1e-8)
        high_dominance_ratio = (valid_dominance > target_dominance).float().sum() / (valid_mask.sum() + 1e-8)
    else:
        loss = dominance_loss.mean()
        avg_dominance = dominance.mean()
        high_dominance_ratio = (dominance > target_dominance).float().mean()
    
    dominance_stats = {
        'avg_dominance': avg_dominance.item(),
        'high_dominance_ratio': high_dominance_ratio.item(),
        'target_dominance': target_dominance
    }
    
    return loss, dominance_stats

def competitive_learning_loss(probs, winner_bonus=1.0, loser_penalty=0.1, valid_pixel_mask=None):
    """
    竞争学习损失函数 - 奖励获胜者（最大概率类别），惩罚失败者
    
    Args:
        probs: 预测概率分布，形状 (C, H, W) 或 (B, C, H, W)
        winner_bonus: 获胜者奖励系数
        loser_penalty: 失败者惩罚系数
        valid_pixel_mask: 有效像元掩膜
    
    Returns:
        loss: 竞争学习损失（标量）
        competition_stats: 竞争统计信息
    """
    if probs.dim() == 3:
        probs = probs.unsqueeze(0)
    if valid_pixel_mask is not None and valid_pixel_mask.dim() == 2:
        valid_pixel_mask = valid_pixel_mask.unsqueeze(0)
    
    B, C, H, W = probs.shape
    
    # 找到获胜者（最大概率的类别）
    max_probs, winner_indices = probs.max(dim=1)  # (B, H, W)
    
    # 创建获胜者掩膜
    winner_mask = torch.zeros_like(probs)
    for b in range(B):
        for h in range(H):
            for w in range(W):
                winner_class = winner_indices[b, h, w]
                winner_mask[b, winner_class, h, w] = 1.0
    
    # 失败者掩膜
    loser_mask = 1.0 - winner_mask
    
    # 竞争损失：奖励获胜者，惩罚失败者
    winner_reward = -torch.log(probs.clamp(min=1e-8)) * winner_mask  # 负log似然，越大概率损失越小
    loser_punishment = probs * loser_mask  # 直接惩罚失败者的概率
    
    competition_loss = (winner_bonus * winner_reward.sum(dim=1) + 
                       loser_penalty * loser_punishment.sum(dim=1))  # (B, H, W)
    
    if valid_pixel_mask is not None:
        valid_mask = (valid_pixel_mask > 0.5).float()
        
        valid_loss = competition_loss * valid_mask
        loss = valid_loss.sum() / (valid_mask.sum() + 1e-8)
        
        # 统计信息
        valid_max_probs = max_probs * valid_mask
        avg_winner_prob = valid_max_probs.sum() / (valid_mask.sum() + 1e-8)
    else:
        loss = competition_loss.mean()
        avg_winner_prob = max_probs.mean()
    
    competition_stats = {
        'avg_winner_prob': avg_winner_prob.item(),
        'winner_bonus': winner_bonus,
        'loser_penalty': loser_penalty
    }
    
    return loss, competition_stats

def combined_sharpening_loss(probs, valid_pixel_mask=None, weights=None, **kwargs):
    """
    组合锐化损失函数 - 综合使用多种锐化技术
    
    Args:
        probs: 预测概率分布，形状 (C, H, W) 或 (B, C, H, W)
        valid_pixel_mask: 有效像元掩膜
        weights: 各损失函数的权重字典，包含：
            - entropy_weight: 熵损失权重
            - confidence_weight: 置信度损失权重  
            - gini_weight: 基尼系数损失权重
            - dominance_weight: 主导性损失权重
            - competition_weight: 竞争学习损失权重
        **kwargs: 各损失函数的参数
    
    Returns:
        total_loss: 总锐化损失
        loss_dict: 各分项损失和统计信息
    """
    if weights is None:
        weights = {
            'entropy_weight': 1.0,
            'confidence_weight': 1.0, 
            'gini_weight': 0.5,
            'dominance_weight': 0.8,
            'competition_weight': 0.3
        }
    
    loss_dict = {}
    total_loss = torch.tensor(0.0, device=probs.device)
    
    # 熵锐化损失
    if weights.get('entropy_weight', 0) > 0:
        entropy_loss, avg_entropy = entropy_sharpening_loss(
            probs, valid_pixel_mask=valid_pixel_mask, 
            **{k: v for k, v in kwargs.items() if k in ['temperature', 'target_entropy_ratio']}
        )
        total_loss += weights['entropy_weight'] * entropy_loss
        loss_dict.update({
            'entropy_loss': entropy_loss.item(),
            'avg_entropy': avg_entropy.item()
        })
    
    # 置信度最大化损失
    if weights.get('confidence_weight', 0) > 0:
        confidence_loss, avg_max_prob = confidence_maximization_loss(
            probs, valid_pixel_mask=valid_pixel_mask,
            **{k: v for k, v in kwargs.items() if k in ['confidence_threshold']}
        )
        total_loss += weights['confidence_weight'] * confidence_loss
        loss_dict.update({
            'confidence_loss': confidence_loss.item(),
            'avg_max_prob': avg_max_prob.item()
        })
    
    # 基尼系数损失
    if weights.get('gini_weight', 0) > 0:
        gini_loss, avg_gini = gini_coefficient_loss(
            probs, valid_pixel_mask=valid_pixel_mask,
            **{k: v for k, v in kwargs.items() if k in ['target_gini']}
        )
        total_loss += weights['gini_weight'] * gini_loss
        loss_dict.update({
            'gini_loss': gini_loss.item(),
            'avg_gini': avg_gini.item()
        })
    
    # 主导性损失
    if weights.get('dominance_weight', 0) > 0:
        dominance_loss, dominance_stats = dominant_class_loss(
            probs, valid_pixel_mask=valid_pixel_mask,
            **{k: v for k, v in kwargs.items() if k in ['dominance_ratio']}
        )
        total_loss += weights['dominance_weight'] * dominance_loss
        loss_dict.update({
            'dominance_loss': dominance_loss.item(),
            **dominance_stats
        })
    
    # 竞争学习损失
    if weights.get('competition_weight', 0) > 0:
        competition_loss, competition_stats = competitive_learning_loss(
            probs, valid_pixel_mask=valid_pixel_mask,
            **{k: v for k, v in kwargs.items() if k in ['winner_bonus', 'loser_penalty']}
        )
        total_loss += weights['competition_weight'] * competition_loss
        loss_dict.update({
            'competition_loss': competition_loss.item(),
            **competition_stats
        })
    
    loss_dict['total_sharpening_loss'] = total_loss.item()
    
    return total_loss, loss_dict

def adaptive_temperature_annealing(iteration, max_iterations, initial_temp=2.0, final_temp=0.1):
    """
    自适应温度退火策略 - 训练过程中逐渐降低温度以增强锐化效果
    
    Args:
        iteration: 当前迭代次数
        max_iterations: 总迭代次数
        initial_temp: 初始温度
        final_temp: 最终温度
    
    Returns:
        current_temperature: 当前温度值
    """
    progress = min(iteration / max_iterations, 1.0)
    
    # 指数衰减
    current_temp = initial_temp * (final_temp / initial_temp) ** progress
    
    return current_temp

# 预定义的锐化配置
SHARPENING_PRESETS = {
    'gentle': {
        'weights': {
            'entropy_weight': 0.5,
            'confidence_weight': 0.3,
            'gini_weight': 0.2,
            'dominance_weight': 0.4,
            'competition_weight': 0.1
        },
        'params': {
            'temperature': 1.5,
            'target_entropy_ratio': 0.3,
            'confidence_threshold': 0.6,
            'target_gini': 0.6,
            'dominance_ratio': 0.6,
            'winner_bonus': 0.8,
            'loser_penalty': 0.05
        }
    },
    'moderate': {
        'weights': {
            'entropy_weight': 1.0,
            'confidence_weight': 1.0,
            'gini_weight': 0.5,
            'dominance_weight': 0.8,
            'competition_weight': 0.3
        },
        'params': {
            'temperature': 1.0,
            'target_entropy_ratio': 0.2,
            'confidence_threshold': 0.7,
            'target_gini': 0.7,
            'dominance_ratio': 0.7,
            'winner_bonus': 1.0,
            'loser_penalty': 0.1
        }
    },
    'aggressive': {
        'weights': {
            'entropy_weight': 2.0,
            'confidence_weight': 1.5,
            'gini_weight': 1.0,
            'dominance_weight': 1.2,
            'competition_weight': 0.5
        },
        'params': {
            'temperature': 0.5,
            'target_entropy_ratio': 0.1,
            'confidence_threshold': 0.8,
            'target_gini': 0.8,
            'dominance_ratio': 0.8,
            'winner_bonus': 1.5,
            'loser_penalty': 0.2
        }
    }
}

def get_sharpening_preset(preset_name='moderate'):
    """
    获取预定义的锐化配置
    
    Args:
        preset_name: 预设名称 ('gentle', 'moderate', 'aggressive')
    
    Returns:
        config: 锐化配置字典
    """
    if preset_name not in SHARPENING_PRESETS:
        logger.warning(f"未知的锐化预设 '{preset_name}'，使用 'moderate'")
        preset_name = 'moderate'
    
    return SHARPENING_PRESETS[preset_name].copy()

def simple_entropy_sharpening(probs, valid_pixel_mask=None, temperature=1.0, target_entropy_ratio=0.2):
    """
    简化的熵锐化损失函数接口，专门用于约束训练
    
    Args:
        probs: 预测概率分布，形状 (C, H, W)
        valid_pixel_mask: 有效像元掩膜，形状 (H, W)
        temperature: 温度参数，越小越锐化
        target_entropy_ratio: 目标熵比率，越小越锐化
    
    Returns:
        loss: 熵锐化损失
        avg_entropy: 平均熵值
    """
    return entropy_sharpening_loss(
        probs=probs,
        temperature=temperature,
        valid_pixel_mask=valid_pixel_mask,
        target_entropy_ratio=target_entropy_ratio
    )

if __name__ == "__main__":
    # 简单测试
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建测试数据
    probs = torch.rand(6, 100, 100).to(device)
    probs = F.softmax(probs, dim=0)  # 归一化为概率分布
    
    valid_mask = torch.ones(100, 100).to(device)
    
    print("熵锐化损失函数测试:")
    
    # 测试熵锐化损失
    entropy_loss, avg_entropy = entropy_sharpening_loss(probs, valid_pixel_mask=valid_mask)
    print(f"熵损失: {entropy_loss:.4f}, 平均熵: {avg_entropy:.4f}")
    
    # 测试简化接口
    simple_loss, simple_entropy = simple_entropy_sharpening(probs, valid_mask, temperature=0.8, target_entropy_ratio=0.15)
    print(f"简化熵损失: {simple_loss:.4f}, 平均熵: {simple_entropy:.4f}")
    
    # 测试不同温度的效果
    print("\n温度对锐化效果的影响:")
    for temp in [2.0, 1.0, 0.5, 0.1]:
        loss, entropy = simple_entropy_sharpening(probs, valid_mask, temperature=temp)
        print(f"温度 {temp:.1f}: 损失={loss:.4f}, 熵={entropy:.4f}")
