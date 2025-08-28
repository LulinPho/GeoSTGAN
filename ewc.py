"""
Elastic Weight Consolidation (EWC) 实现
用于防止灾难性遗忘的增量学习
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, List, Tuple
import os
import pickle

class EWC:
    """
    Elastic Weight Consolidation (EWC) 实现
    基于论文: "Overcoming catastrophic forgetting in neural networks"
    """
    
    def __init__(self, model: nn.Module, device: torch.device, lambda_ewc: float = 1000.0):
        """
        Args:
            model: 需要保护的模型
            device: 设备
            lambda_ewc: EWC正则化强度
        """
        self.model = model
        self.device = device
        self.lambda_ewc = lambda_ewc
        
        # 存储重要参数
        self.important_params = {}  # {param_name: (mean, precision)}
        self.fisher_info = {}       # Fisher信息矩阵
        
        # 任务计数器
        self.task_count = 0
        
    def compute_fisher_information(self, dataloader, num_samples: int = 1000):
        """
        计算Fisher信息矩阵
        Fisher信息衡量参数对损失函数的重要性
        """
        self.model.eval()
        
        # 初始化Fisher信息
        fisher_info = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                fisher_info[name] = torch.zeros_like(param.data)
        
        # 计算Fisher信息
        sample_count = 0
        for batch_idx, batch in enumerate(dataloader):
            if sample_count >= num_samples:
                break
                
            # 获取数据
            x = batch[0].to(self.device)
            y = batch[1].to(self.device)
            mask = batch[2].to(self.device)
            
            # 前向传播
            x_gen = torch.cat([x, mask], dim=2)
            fake, fake_features = self.model(x_gen)
            
            # 计算损失
            loss = F.cross_entropy(fake[:, -1, :, :, :], y[:, -1, :, :, :])
            
            # 计算梯度
            self.model.zero_grad()
            loss.backward()
            
            # 累积Fisher信息
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher_info[name] += param.grad.data ** 2
            
            sample_count += x.size(0)
        
        # 平均Fisher信息
        for name in fisher_info:
            fisher_info[name] /= sample_count
        
        return fisher_info
    
    def update_important_params(self, dataloader, task_name: str = None):
        """
        更新重要参数
        在任务完成后调用，保存当前参数和Fisher信息
        """
        # 计算Fisher信息
        fisher_info = self.compute_fisher_information(dataloader)
        
        # 保存当前参数和Fisher信息
        important_params = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                important_params[name] = {
                    'mean': param.data.clone(),
                    'precision': fisher_info[name].clone()
                }
        
        # 存储重要参数
        if task_name is None:
            task_name = f"task_{self.task_count}"
        
        self.important_params[task_name] = important_params
        self.task_count += 1
        
        logging.getLogger(__name__).info(f"EWC: 已保存任务 {task_name} 的重要参数")
        
    def compute_ewc_loss(self) -> torch.Tensor:
        loss_components = []
        
        for task_name, task_params in self.important_params.items():
            for param_name, param in self.model.named_parameters():
                if param.requires_grad and param_name in task_params:
                    important_mean = task_params[param_name]['mean'].to(param.device)
                    important_precision = task_params[param_name]['precision'].to(param.device)
                    
                    param_diff = param - important_mean
                    loss_component = torch.sum(important_precision * param_diff ** 2)
                    loss_components.append(loss_component)
        
        if loss_components:
            ewc_loss = torch.stack(loss_components).sum()
        else:
            # 保持与模型参数的梯度连接
            first_param = next(iter(self.model.parameters()))
            ewc_loss = torch.zeros(1, device=first_param.device, requires_grad=True).squeeze()
        
        return 0.5 * self.lambda_ewc * ewc_loss
    
    def save_ewc_state(self, filepath: str):
        """保存EWC状态"""
        state = {
            'important_params': self.important_params,
            'task_count': self.task_count,
            'lambda_ewc': self.lambda_ewc
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        
            logging.getLogger(__name__).info(f"EWC状态已保存到: {filepath}")
    
    def load_ewc_state(self, filepath: str):
        """加载EWC状态"""
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                state = pickle.load(f)
            
            self.important_params = state['important_params']
            self.task_count = state['task_count']
            self.lambda_ewc = state.get('lambda_ewc', self.lambda_ewc)
            # 将张量移动到当前设备
            for task_name, task_params in self.important_params.items():
                for param_name, data in task_params.items():
                    if isinstance(data.get('mean', None), torch.Tensor):
                        data['mean'] = data['mean'].to(self.device)
                    if isinstance(data.get('precision', None), torch.Tensor):
                        data['precision'] = data['precision'].to(self.device)
            
            logging.getLogger(__name__).info(f"EWC状态已从 {filepath} 加载")
            logging.getLogger(__name__).info(f"已加载 {len(self.important_params)} 个任务的重要参数")
        else:
            logging.getLogger(__name__).warning(f"EWC状态文件不存在: {filepath}")
    
    def get_parameter_importance(self, param_name: str) -> float:
        """获取参数重要性"""
        total_importance = 0.0
        
        for task_name, task_params in self.important_params.items():
            if param_name in task_params:
                precision = task_params[param_name]['precision']
                total_importance += torch.mean(precision).item()
        
        return total_importance
    
    def print_parameter_importance(self, top_k: int = 10):
        """打印最重要的参数"""
        importance_dict = {}
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                importance = self.get_parameter_importance(name)
                importance_dict[name] = importance
        
        # 按重要性排序
        sorted_params = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        
        logging.getLogger(__name__).info(f"\n=== EWC参数重要性 (Top {top_k}) ===")
        for i, (name, importance) in enumerate(sorted_params[:top_k]):
            logging.getLogger(__name__).info(f"{i+1:2d}. {name}: {importance:.6f}")

class OnlineEWC:
    """
    在线EWC实现
    适用于连续学习场景
    """
    
    def __init__(self, model: nn.Module, device: torch.device, lambda_ewc: float = 1000.0, gamma: float = 0.9):
        """
        Args:
            model: 需要保护的模型
            device: 设备
            lambda_ewc: EWC正则化强度
            gamma: 遗忘因子
        """
        self.model = model
        self.device = device
        self.lambda_ewc = lambda_ewc
        self.gamma = gamma
        
        # 在线Fisher信息
        self.online_fisher = {}
        self.online_mean = {}
        
        # 初始化
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.online_fisher[name] = torch.zeros_like(param.data)
                self.online_mean[name] = param.data.clone()
    
    def update_online_fisher(self, loss: torch.Tensor):
        """
        在线更新Fisher信息
        """
        # 计算梯度
        self.model.zero_grad()
        loss.backward()
        
        # 更新在线Fisher信息
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                # 在线更新
                self.online_fisher[name] = self.gamma * self.online_fisher[name] + \
                                         (1 - self.gamma) * (param.grad.data ** 2)
                
                # 更新均值
                self.online_mean[name] = self.gamma * self.online_mean[name] + \
                                       (1 - self.gamma) * param.data
    
    def compute_online_ewc_loss(self) -> torch.Tensor:
        """
        计算在线EWC损失
        """
        ewc_loss = torch.tensor(0.0, device=self.device)
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                fisher = self.online_fisher[name]
                mean = self.online_mean[name]
                
                param_diff = param - mean
                ewc_loss += torch.sum(fisher * param_diff ** 2)
        
        return 0.5 * self.lambda_ewc * ewc_loss

def create_ewc(model: nn.Module, device: torch.device, ewc_type: str = "standard", **kwargs) -> EWC:
    """
    创建EWC实例
    
    Args:
        model: 模型
        device: 设备
        ewc_type: EWC类型 ("standard" 或 "online")
        **kwargs: 其他参数
    """
    if ewc_type == "standard":
        return EWC(model, device, **kwargs)
    elif ewc_type == "online":
        return OnlineEWC(model, device, **kwargs)
    else:
        raise ValueError(f"未知的EWC类型: {ewc_type}")
