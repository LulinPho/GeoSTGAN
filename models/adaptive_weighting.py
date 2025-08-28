"""
自适应损失加权模块
基于论文: Multi-Task Learning as Multi-Objective Optimization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple

class AdaptiveLossWeighting(nn.Module):
    """
    自适应损失加权模块
    基于多目标优化理论，动态调整不同损失项的权重
    """
    
    def __init__(self, num_losses: int, alpha: float = 0.5, beta: float = 0.1):
        """
        Args:
            num_losses: 损失项数量
            alpha: 权重更新率
            beta: 梯度平衡参数
        """
        super().__init__()
        self.num_losses = num_losses
        self.alpha = alpha
        self.beta = beta
        
        # 初始化权重为均匀分布
        self.weights = nn.Parameter(torch.ones(num_losses) / num_losses)
        
        # 损失历史记录
        self.loss_history = []
        self.weight_history = []
        
        # 梯度统计
        self.grad_norms = torch.zeros(num_losses)
        self.grad_history = []
        
    def forward(self, losses: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算加权损失和权重
        
        Args:
            losses: 损失项列表
            
        Returns:
            weighted_loss: 加权总损失
            weights: 当前权重
        """
        if len(losses) != self.num_losses:
            raise ValueError(f"Expected {self.num_losses} losses, got {len(losses)}")
        
        # 确保权重为正数
        weights = F.softplus(self.weights)
        weights = weights / weights.sum()  # 归一化
        
        # 计算加权损失
        weighted_loss = sum(w * l for w, l in zip(weights, losses))
        
        return weighted_loss, weights
    
    def update_weights(self, losses: List[torch.Tensor], grads: List[torch.Tensor] = None):
        """
        基于损失值和梯度更新权重
        
        Args:
            losses: 损失项列表
            grads: 对应的梯度列表（可选）
        """
        with torch.no_grad():
            # 记录损失历史
            loss_values = [l.item() for l in losses]
            self.loss_history.append(loss_values)
            
            # 计算损失相对变化
            if len(self.loss_history) > 1:
                prev_losses = self.loss_history[-2]
                loss_changes = [(curr - prev) / (prev + 1e-8) 
                              for curr, prev in zip(loss_values, prev_losses)]
            else:
                loss_changes = [0.0] * self.num_losses
            
            # 基于梯度范数更新权重（如果提供了梯度）
            if grads is not None:
                grad_norms = [torch.norm(g).item() for g in grads]
                self.grad_norms = torch.tensor(grad_norms)
                self.grad_history.append(grad_norms)
                
                # 计算梯度平衡权重
                grad_weights = self._compute_gradient_weights(grad_norms)
            else:
                grad_weights = torch.ones(self.num_losses) / self.num_losses
            
            # 计算损失平衡权重
            loss_weights = self._compute_loss_weights(loss_values, loss_changes)
            
            # 综合权重更新
            target_weights = self.alpha * grad_weights + (1 - self.alpha) * loss_weights
            
            # 平滑更新
            current_weights = F.softplus(self.weights)
            current_weights = current_weights / current_weights.sum()
            
            # 计算权重更新
            weight_update = target_weights - current_weights
            
            # 应用更新
            self.weights.data += self.beta * weight_update
            
            # 记录权重历史
            self.weight_history.append(F.softplus(self.weights).detach().cpu().numpy())
    
    def _compute_gradient_weights(self, grad_norms: List[float]) -> torch.Tensor:
        """
        基于梯度范数计算权重
        梯度范数越大，权重越小（防止某个损失主导训练）
        """
        grad_norms = torch.tensor(grad_norms)
        
        # 计算梯度范数的倒数
        inv_grad_norms = 1.0 / (grad_norms + 1e-8)
        
        # 归一化
        weights = inv_grad_norms / inv_grad_norms.sum()
        
        return weights
    
    def _compute_loss_weights(self, loss_values: List[float], loss_changes: List[float]) -> torch.Tensor:
        """
        基于损失值和变化计算权重
        损失值越大或变化越大，权重越小
        """
        loss_values = torch.tensor(loss_values)
        loss_changes = torch.tensor(loss_changes)
        
        # 计算损失相对大小
        loss_ranks = torch.argsort(loss_values, descending=True)
        rank_weights = 1.0 / (loss_ranks.float() + 1.0)
        
        # 考虑损失变化
        change_penalty = torch.exp(-torch.abs(torch.tensor(loss_changes)))
        
        # 综合权重
        weights = rank_weights * change_penalty
        weights = weights / weights.sum()
        
        return weights
    
    def get_pareto_weights(self) -> torch.Tensor:
        """
        获取Pareto最优权重
        基于历史数据计算最优权重组合
        """
        if len(self.loss_history) < 10:
            return F.softplus(self.weights).detach()
        
        # 计算损失相关性矩阵
        loss_array = torch.tensor(self.loss_history[-100:])  # 使用最近100个样本
        corr_matrix = torch.corrcoef(loss_array.T)
        
        # 计算任务冲突度
        conflicts = torch.sum(torch.abs(corr_matrix - torch.eye(self.num_losses)), dim=1)
        
        # 冲突度越高，权重越小
        pareto_weights = 1.0 / (conflicts + 1.0)
        pareto_weights = pareto_weights / pareto_weights.sum()
        
        return pareto_weights
    
    def reset_weights(self):
        """重置权重为均匀分布"""
        self.weights.data = torch.zeros_like(self.weights.data)
    
    def get_weight_stats(self) -> Dict:
        """获取权重统计信息"""
        current_weights = F.softplus(self.weights).detach().cpu().numpy()
        
        return {
            'current_weights': current_weights,
            'weight_history': self.weight_history[-10:] if self.weight_history else [],
            'loss_history': self.loss_history[-10:] if self.loss_history else [],
            'grad_history': self.grad_history[-10:] if self.grad_history else []
        }

class MultiObjectiveOptimizer:
    """
    多目标优化器
    基于论文中的MGDA算法
    """
    
    def __init__(self, model: nn.Module, num_tasks: int, device: torch.device):
        self.model = model
        self.num_tasks = num_tasks
        self.device = device
        
        # 任务梯度存储
        self.task_gradients = []
        self.scaling_factors = torch.ones(num_tasks, device=device)
        
    def compute_scaling_factors(self, task_gradients: List[torch.Tensor]) -> torch.Tensor:
        """
        计算MGDA缩放因子
        解决多目标优化问题
        """
        if len(task_gradients) != self.num_tasks:
            raise ValueError(f"Expected {self.num_tasks} gradients, got {len(task_gradients)}")
        
        # 将梯度展平并堆叠
        flat_grads = []
        for grad in task_gradients:
            flat_grad = torch.cat([g.flatten() for g in grad])
            flat_grads.append(flat_grad)
        
        G = torch.stack(flat_grads)  # [num_tasks, total_params]
        
        # 计算Gram矩阵
        GGt = torch.mm(G, G.t())
        
        # 求解二次规划问题
        try:
            # 使用伪逆求解
            alpha = torch.linalg.solve(GGt, torch.ones(self.num_tasks, device=self.device))
            alpha = alpha / alpha.sum()  # 归一化
        except:
            # 如果矩阵奇异，使用均匀权重
            alpha = torch.ones(self.num_tasks, device=self.device) / self.num_tasks
        
        return alpha
    
    def get_balanced_gradient(self, task_gradients: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        获取平衡后的梯度
        """
        scaling_factors = self.compute_scaling_factors(task_gradients)
        
        # 计算加权梯度
        balanced_grads = []
        for i, grad_list in enumerate(task_gradients):
            weighted_grad = [g * scaling_factors[i] for g in grad_list]
            balanced_grads.append(weighted_grad)
        
        # 合并所有任务的梯度
        final_grads = []
        for param_idx in range(len(balanced_grads[0])):
            combined_grad = sum(task_grads[param_idx] for task_grads in balanced_grads)
            final_grads.append(combined_grad)
        
        return final_grads

def create_adaptive_weighting(num_losses: int, method: str = 'adaptive') -> AdaptiveLossWeighting:
    """
    创建自适应权重模块
    
    Args:
        num_losses: 损失项数量
        method: 权重方法 ('adaptive', 'uniform', 'fixed')
    """
    if method == 'adaptive':
        return AdaptiveLossWeighting(num_losses)
    elif method == 'uniform':
        # 均匀权重
        module = AdaptiveLossWeighting(num_losses)
        module.reset_weights()
        return module
    else:
        raise ValueError(f"Unknown method: {method}")
