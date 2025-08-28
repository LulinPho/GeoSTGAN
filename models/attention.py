import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from .utils import position_encoding_2d, conv_block
from .gau import GAU

# 移除全局 device 变量

class PatchAttention(nn.Module):
    """针对15x15图像块的ViT注意力模块
    
    Args:
        in_channels: 输入通道数
        patch_size: 图像块大小，默认为15x15
        num_heads: 注意力头数
        dropout: Dropout比率
    """
    def __init__(self, hidden_channels=64, out_channels=64, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        
        # 1x1卷积降维占位符
        self.proj = None
        self.proj_initialized = False
        
        # 位置编码
        self.pos_encoding = None
        self.pos_encoding_initialized = False

        self.gau_1 = GAU(input_channels=hidden_channels * 16, 
                        output_channels=hidden_channels * 16,
                        expansion_factor=2, 
                        dropout=dropout, 
                        residual=True)

        self.gau_2 = GAU(input_channels=hidden_channels * 16, 
                        output_channels=out_channels,
                        expansion_factor=2, 
                        dropout=dropout, 
                        residual=True)
        
    def forward(self, x):
        x = x.to(x.device)
        b, c, h, w = x.shape
        
        out, weights = self.gau_1(x)

        assert out.shape == (b, self.hidden_channels * 16, h, w)

        out, weights = self.gau_2(out)

        assert out.shape == (b, self.out_channels, h, w)

        return out, weights


def visualize_attention_weights(weights, save_path=None):
    """
    Visualize attention weights
    
    Args:
        weights: Attention weight tensor [B, H, W]
        save_path: Path to save the visualization, if None then display the image
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Convert weights to numpy array
    weights_np = weights.detach().cpu().numpy()
    
    # Create image grid
    batch_size = weights_np.shape[0]
    fig, axes = plt.subplots(1, batch_size, figsize=(5*batch_size, 5))
    
    # Ensure axes is always iterable
    axes = axes if isinstance(axes, (list, tuple)) else [axes]
    
    # Visualize each batch sample
    for i, ax in enumerate(axes):
        # Normalize to [0,1] range
        weight_map = weights_np[i]
        weight_map = (weight_map - weight_map.min()) / (weight_map.max() - weight_map.min())
        
        # Plot heatmap
        im = ax.imshow(weight_map, cmap='viridis')
        ax.set_title(f'Sample {i+1} Attention Weights')
        plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    
    # Save or display image
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()