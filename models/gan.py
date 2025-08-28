import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

from .gru import GRUWithGAUNetPP
from .utils import adaptive_cube_split, StableDiscriminatorBlock

logger = logging.getLogger(__name__)

class Generator(nn.Module):
    def __init__(self, hidden_channels=32, num_classes=10, feature_decoder=True, features_channels=128, dropout=0.1):
        super().__init__()
        self.gru = GRUWithGAUNetPP(hidden_channels=hidden_channels, num_classes=num_classes, feature_decoder=feature_decoder, features_channels=features_channels, dropout=dropout)

    def forward(self, x):
        return self.gru(x) # [B, T, N, H, W]
    
class DualDiscriminator(nn.Module):
    """
    整合的双重判别器：同时实现patch和global判别功能
    维持两种判别器的功能独立性，但共享基础组件
    """
    def __init__(self, hidden_channels=64, block_num=256, use_soft_input=True, temperature=1.0):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.block_num = block_num
        self.use_soft_input = use_soft_input
        self.temperature = temperature
        
        # 共享的输入处理层
        self.conv_block = None
        
        # Patch判别器专用的较浅网络
        self.patch_blocks = nn.Sequential(
            StableDiscriminatorBlock(hidden_channels, hidden_channels, stride=2),
            StableDiscriminatorBlock(hidden_channels, hidden_channels * 2),
            StableDiscriminatorBlock(hidden_channels * 2, hidden_channels * 2, stride=2),
            StableDiscriminatorBlock(hidden_channels * 2, hidden_channels * 4),
            StableDiscriminatorBlock(hidden_channels * 4, hidden_channels * 4, stride=2),
        )
        
        # Global判别器专用的更深网络
        self.global_blocks = nn.Sequential(
            StableDiscriminatorBlock(hidden_channels, hidden_channels, stride=2),
            StableDiscriminatorBlock(hidden_channels, hidden_channels * 2),
            StableDiscriminatorBlock(hidden_channels * 2, hidden_channels * 2, stride=2),
            StableDiscriminatorBlock(hidden_channels * 2, hidden_channels * 4),
            StableDiscriminatorBlock(hidden_channels * 4, hidden_channels * 4, stride=2),
            StableDiscriminatorBlock(hidden_channels * 4, hidden_channels * 8),
            StableDiscriminatorBlock(hidden_channels * 8, hidden_channels * 8, stride=2),
        )
        
        # 分类器
        self.patch_classifier_template = None
        self.patch_classifier = None
        self.global_classifier = None

    def _get_conv_block(self, input_channels, output_channels, kernel_size, stride=1, padding=0):
        return nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.InstanceNorm2d(output_channels),
            nn.LeakyReLU(0.2, inplace=False),
        )
        
    def _get_patch_classifier(self, input_shape, device):
        """获取或创建patch分类器"""
        if self.patch_classifier is None or self.patch_classifier_template != input_shape:
            b, c, h, w = input_shape
            self.patch_classifier_template = input_shape
            self.patch_classifier = nn.Sequential(
                nn.Conv2d(c, self.hidden_channels * 8, kernel_size=(h, w)),
                nn.LeakyReLU(0.2, inplace=False),
                nn.Conv2d(self.hidden_channels * 8, 1, kernel_size=(1, 1))
            ).to(device)
        return self.patch_classifier
    
    def _get_global_classifier(self, input_shape, device):
        """获取或创建global分类器"""
        b, c, h, w = input_shape
        if self.global_classifier is None:
            self.global_classifier = nn.Sequential(
                nn.Conv2d(c, self.hidden_channels * 8, kernel_size=(h, w)),
                nn.LeakyReLU(0.2, inplace=False),
                nn.Conv2d(self.hidden_channels * 8, 1, kernel_size=(1, 1))
            ).to(device)
        return self.global_classifier

    def forward_patch(self, x, y):
        """
        Patch判别器前向传播
        返回: (patch_outputs, n_splits)
        """
        b, t, c, h, w = x.shape
        by, n, hy, wy = y.shape
        assert b == by and h == hy and w == wy, "x和y的batch、空间尺寸需一致"

        # 将x的时间维度融合到通道
        x_merged = x.permute(0,2,1,3,4).contiguous().view(b, t*c, h, w)
        x_cat = torch.cat([x_merged, y], dim=1)

        if self.conv_block is None:
            self.conv_block = self._get_conv_block(t*c+n, self.hidden_channels, kernel_size=1).to(x.device).to(x.dtype)

        # 分块处理
        x_split, n_splits, _ = adaptive_cube_split(x_cat, split_dims=[2, 3], target_num=self.block_num, overlap=0.2)
        x_split_0 = x_split[0, :, :, :, :]
        x_split_0 = self.conv_block(x_split_0)
        x_split_0 = self.patch_blocks(x_split_0)

        classifier = self._get_patch_classifier(x_split_0.shape, x_cat.device)
        output = torch.zeros((b, x_split.shape[0]), device=x_cat.device)
        
        for j in range(x_split.shape[0]):
            x_split_j = x_split[j, :, :, :, :]
            x_split_j = self.conv_block(x_split_j)
            x_split_j = self.patch_blocks(x_split_j)
            x_split_j = classifier(x_split_j)
            output[:, j] = x_split_j.view(b)
            
        output = output.view(b, n_splits[0], n_splits[1])
        outputs = output.unsqueeze(1)  # [B, 1, n1, n2]
        return outputs, n_splits

    def forward_global(self, x, y):
        """
        Global判别器前向传播
        返回: global_outputs
        """
        b, t, c, h, w = x.shape
        by, n, hy, wy = y.shape
        assert b == by and h == hy and w == wy, "x和y的batch、空间尺寸需一致"

        x_merged = x.permute(0,2,1,3,4).contiguous().view(b, t*c, h, w)
        x_input = torch.cat([x_merged, y], dim=1)

        if self.conv_block is None:
            self.conv_block = self._get_conv_block(t*c+n, self.hidden_channels, kernel_size=1).to(x_input.device).to(x_input.dtype)
        
        x = self.conv_block(x_input)
        x = self.global_blocks(x)
        classifier = self._get_global_classifier(x.shape, x.device)
        x = classifier(x)
        return x.view(b), None

    def forward(self, x, y, mode='both'):
        """
        统一的前向传播接口
        
        Args:
            x: 条件输入
            y: 目标输入
            mode: 'patch', 'global', 'both'
        
        Returns:
            根据mode返回相应的输出
        """
        if mode == 'patch':
            return self.forward_patch(x, y)
        elif mode == 'global':
            return self.forward_global(x, y)
        elif mode == 'both':
            patch_outputs, n_splits = self.forward_patch(x, y)
            global_outputs = self.forward_global(x, y)
            return {
                'patch': (patch_outputs, n_splits),
                'global': global_outputs
            }
        else:
            raise ValueError(f"Unsupported mode: {mode}")

# 为了向后兼容，保留原有的类名作为别名
class PatchDiscriminator(DualDiscriminator):
    """向后兼容的Patch判别器"""
    def forward(self, x, y):
        return self.forward_patch(x, y)

class GlobalDiscriminator(DualDiscriminator):
    """向后兼容的Global判别器"""
    def forward(self, x, y):
        return self.forward_global(x, y)

