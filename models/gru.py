import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.xpu
import os

from .gaunetpp import GAUNetPP
from .utils import conv_block

import logging

logger = logging.getLogger(__name__)

# 配置日志记录器
logger.setLevel(logging.INFO)

class GRUWithGAUNetPP(nn.Module):
    """使用GAUNetPP作为隐藏层生成器的GRU模型
    
    Args:
        input_channels: 输入通道数
        hidden_channels: 隐藏层通道数
        num_classes: 输出类别数
        num_layers: GRU层数
        dropout: Dropout比率
    """
    def __init__(self, hidden_channels=64, num_classes=10, features_channels=128, dropout=0.1, feature_decoder=False):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.num_classes = num_classes
        self.features_channels = features_channels
        self.feature_decoder = feature_decoder
        self.dropout = dropout
        
        # GAUNetPP模块用于生成隐藏状态
        self.terraflow = GAUNetPP(
            hidden_channels=hidden_channels,
            out_channels=hidden_channels,
            classify=False,
            multi_scale_features=False
        )
        
        # GRU门控机制
        self.update_gate = nn.Conv2d(hidden_channels * 8, hidden_channels *4, kernel_size=1, padding='same')
        self.reset_gate = nn.Conv2d(hidden_channels * 8, hidden_channels *4, kernel_size=1, padding='same')
        self.candidate_gate = nn.Conv2d(hidden_channels * 8, hidden_channels *4, kernel_size=1, padding='same')
        
        # 激活函数
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        
        # Dropout层
        self.dropout_layer = nn.Dropout2d(dropout)
        
        # 解码器，输出logits
        self.decoder = nn.Sequential(
            conv_block(hidden_channels * 4, hidden_channels*2 , padding='same'),
            conv_block(hidden_channels*2 , hidden_channels, padding='same'),
            conv_block(hidden_channels, hidden_channels, padding='same'),
            nn.Conv2d(hidden_channels, num_classes, kernel_size=1, padding=0)
        )

        self.feature_decoder = nn.Sequential(
            conv_block(hidden_channels* 4, hidden_channels*2 , padding='same'),
            conv_block(hidden_channels*2 , hidden_channels, padding='same'),
            conv_block(hidden_channels, hidden_channels, padding='same'),
            nn.Conv2d(hidden_channels, features_channels, kernel_size=1, padding=0),
        )
        
        # 初始化隐藏状态
        self.hidden = None
        
    def init_hidden(self, batch_size, height, width, device):
        """初始化隐藏状态"""
        self.hidden = torch.zeros(batch_size, self.hidden_channels *4, height, width, device=device)
        
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量 [batch_size, channels, height, width] 或 [sequence_length, batch_size, channels, height, width]
            sequence_length: 序列长度，如果为None则x为单个时间步的输入
        """
        # 根据x的维度判断是单步输入还是序列输入
        if x.dim() == 4:
            # [batch_size, channels, height, width]，单步输入
            return self._forward_step(x)
        elif x.dim() == 5:
            return self._forward_sequence(x)
        else:
            raise ValueError("输入x的维度不支持，仅支持4维或5维张量")
    
    def _forward_step(self, x):
        """单个时间步的前向传播"""
        x = x.to(x.device)
        
        # 使用GAUNetPP生成候选隐藏状态
        # 确保子模块已初始化（延迟初始化可能导致某些参数未参与前向）
        if self.terraflow is None:
            raise RuntimeError("terraflow module is not initialized")
        candidate_hidden = self.terraflow(x)  # [B, hidden_channels *4, H, W]
        
        assert self.hidden is not None

        # 准备GRU输入
        gru_input = torch.cat([self.hidden, candidate_hidden], dim=1)  # [B, hidden_channels*8, H, W]
        
        # GRU门控机制
        update_gate = self.sigmoid(self.update_gate(gru_input).to(torch.float32))
        reset_gate = self.sigmoid(self.reset_gate(gru_input).to(torch.float32))
        
        # 重置门控制候选隐藏状态的计算
        reset_hidden = reset_gate * self.hidden
        candidate_input = torch.cat([reset_hidden, candidate_hidden], dim=1)
        candidate = self.tanh(self.candidate_gate(candidate_input))
        
        # 更新隐藏状态（避免原地操作）
        new_hidden = (1 - update_gate) * self.hidden + update_gate * candidate
        self.hidden = new_hidden
        
        # 应用dropout
        self.hidden = self.dropout_layer(self.hidden)
        
        # 解码输出
        output = self.decoder(self.hidden)
        output = F.softmax(output, dim=1)
        
        if self.feature_decoder:
            features = self.feature_decoder(self.hidden)
        else:
            features = None
        
        return output, features
    
    def _forward_sequence(self, x):
        """序列输入的前向传播"""
        x = x.to(x.device)
        b, t, c, h, w = x.shape
        
        # 每个序列的初始隐藏状态都应为全0
        self.init_hidden(b, h, w, x.device)
        
        outputs = []
        features_states = []
        
        # 逐时间步处理
        for _t in range(t):
            output, features = self._forward_step(x[:, _t])
            outputs.append(output)
            if self.feature_decoder:
                features_states.append(features)
        
        # 堆叠输出
        outputs = torch.stack(outputs, dim=1)  # [batch_size, t, num_classes, height, width]
        # 恢复到原始行为：不保留额外的有效性序列属性
        if self.feature_decoder:
            features_states = torch.stack(features_states, dim=1)  # [batch_size, t, features_channels, height, width]
            return outputs, features_states
        else:
            return outputs