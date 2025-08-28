import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import conv_block, rope_2d, position_encoding_2d

import logging

logger = logging.getLogger(__name__)
# 配置日志记录器
logger.setLevel(logging.INFO)

class GAU(nn.Module):
    def __init__(self, input_channels, output_channels, expansion_factor=2, dropout=0.1, residual=True):
        super().__init__()
        self.dim = input_channels
        self.expansion_dim = input_channels * expansion_factor
        self.scaled_dim = input_channels // expansion_factor
        self.output_channels = output_channels
        
        # GAU的核心组件
        self.norm = nn.InstanceNorm2d(input_channels)
        self.gate = conv_block(input_channels, input_channels, kernel_size=1)
        self.value = conv_block(input_channels, input_channels, kernel_size=1)

        self.z = conv_block(input_channels, self.scaled_dim, kernel_size=1)

        self.gamma = nn.Parameter(torch.rand(2,self.scaled_dim))
        self.beta = nn.Parameter(torch.rand(2,self.scaled_dim))

        self.relu2 = nn.LeakyReLU(inplace=False)
        
        self.proj = conv_block(input_channels, output_channels, kernel_size=1)
        
        # 门控机制
        self.gate_act = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)

        self.residual = residual

        self.position_encoding = None
        self.position_encoding_initialized = None

    def _init_position_encoding(self, x):
        if not self.position_encoding_initialized:
            self.position_encoding = position_encoding_2d(x.shape[2], x.shape[3], self.scaled_dim).to(x.device)
            self.position_encoding_initialized = True
        
    def forward(self, x):
        b, c, h, w = x.shape

        self._init_position_encoding(x)

        # 归一化
        x_n = self.norm(x)

        g = self.gate(x_n).to(x.device)
        v = self.value(x_n).to(x.device)

        z = self.z(x_n).to(x.device)

        qk = torch.einsum('b c h w, p c -> b p c h w', z, self.gamma.to(x.device)).to(x.device)
        qk = qk + self.beta.to(x.device).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        
        q, k  = qk.unbind(dim=1)

        q = rope_2d(self.position_encoding, q).to(x.device)
        k = rope_2d(self.position_encoding, k).to(x.device)

        # 计算注意力分数，使用缩放点积注意力
        sim = torch.einsum('b c h w, b c h w -> b h w', q, k) / (self.scaled_dim ** 0.5)
        
        sim=sim.to(torch.float32)
        # 应用softmax归一化注意力权重
        A = F.softmax(sim.view(b, -1), dim=-1).view(b, h, w)
        A = self.dropout(A)

        V = torch.einsum('b h w, b c h w -> b c h w', A, v)

        V = V * g

        if self.residual:
            V = V + x_n

        out = self.proj(V)

        assert out.shape == (b, self.output_channels, h, w)
        
        return out, A
