from functools import singledispatch
import torch
import torch.nn as nn
import math


@singledispatch
def conv_block(in_channels, out_channels, kernel_size=3, padding='valid', stride=1):
    """Basic 2D convolution block containing Conv2d+BN+LeakyReLU
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int, optional): Size of the convolving kernel. Defaults to 3.
        padding (str, optional): Padding mode. Can be 'valid' or 'same'. Defaults to 'valid'.
        stride (int, optional): Stride of the convolution. Defaults to 1.
    Returns:
        nn.Sequential: A sequential module containing Conv2d, BatchNorm2d and LeakyReLU layers
    """
    if padding == 'same':
        padding = kernel_size//2
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, stride=stride),
        nn.InstanceNorm2d(out_channels, affine=True, track_running_stats=False),
        nn.LeakyReLU(0.2, inplace=False)
    )

@conv_block.register(torch.Tensor)
def _(input_tensor, out_channels, kernel_size=3, padding='valid', stride=1):
    """Basic 2D convolution block containing Conv2d+BN+LeakyReLU
    
    Args:
        input_tensor (torch.Tensor): Input tensor
        out_channels (int): Number of output channels
        kernel_size (int, optional): Size of the convolving kernel. Defaults to 3.
        padding (str, optional): Padding mode. Can be 'valid' or 'same'. Defaults to 'valid'.
        stride (int, optional): Stride of the convolution. Defaults to 1.
    Returns:
        nn.Sequential: A sequential module containing Conv2d, BatchNorm2d and LeakyReLU layers
    """
    in_channels = input_tensor.size(1)
    if padding == 'same':
        padding = kernel_size//2
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, stride=stride),
        nn.InstanceNorm2d(out_channels, affine=True, track_running_stats=False),
        nn.LeakyReLU(0.2, inplace=False)
    )

class StableDiscriminatorBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding='same'):
        super().__init__()
        if padding == 'same':
            padding = kernel_size//2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bn = nn.InstanceNorm2d(out_channels, affine=True, track_running_stats=False)
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=False)
    
    def forward(self, x):
        
        x = self.conv(x)
        
        x = self.bn(x)
        
        x = self.leaky_relu(x)
        
        return x

@singledispatch
def discriminator_block(in_channels, out_channels, kernel_size=3, stride=1, padding='same'):
    return StableDiscriminatorBlock(in_channels, out_channels, kernel_size, stride, padding)

@discriminator_block.register(torch.Tensor)
def _(input_tensor, out_channels, kernel_size=3, stride=1, padding='same'):
    in_channels = input_tensor.size(1)
    if padding == 'same':
        padding = kernel_size//2
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
        nn.InstanceNorm2d(out_channels, affine=True, track_running_stats=False),
        nn.LeakyReLU(0.2, inplace=False)
    )

def position_encoding_1d(dim, length):
    """Generate 1D positional encoding
    
    Args:
        dim (int): Dimension of the encoding
        length (int): Length of the sequence
    
    Returns:
        torch.Tensor: Positional encoding matrix of shape (dim, length)
    """
    assert dim % 2 == 0

    pe = torch.zeros(dim, length)

    position = torch.arange(length).unsqueeze(1)

    div_term = torch.exp(torch.arange(0, dim, 2) * -(math.log(10000.0) / dim))

    pe[0::2] = torch.sin(position * div_term)
    pe[1::2] = torch.cos(position * div_term)

    return pe


def position_encoding_2d(h, w, dim):
    """Generate 2D positional encoding
    
    Args:
        h (int): Height of the feature map
        w (int): Width of the feature map
        dim (int): Dimension of the encoding
    
    Returns:
        torch.Tensor: Positional encoding tensor of shape (dim, h, w)
    """
    assert dim % 4 == 0

    pe = torch.zeros(h, w, dim)

    d_model = dim // 2

    div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))

    pos_w = torch.arange(w).unsqueeze(1)
    pos_h = torch.arange(h).unsqueeze(1)

    pe[:,:,0:d_model:2] = torch.sin(pos_w * div_term).unsqueeze(1).repeat(1, h, 1).transpose(0, 1)
    pe[:,:,1:d_model + 1:2] = torch.cos(pos_w * div_term).unsqueeze(1).repeat(1, h, 1).transpose(0, 1)

    pe[:,:,d_model::2] = torch.sin(pos_h * div_term).unsqueeze(0).repeat(w, 1, 1).transpose(0, 1)
    pe[:,:,d_model + 1::2] = torch.cos(pos_h * div_term).unsqueeze(0).repeat(w, 1, 1).transpose(0, 1)

    return pe.permute(2,0,1)


def rope_1d(pe, x):
    """Apply 1D Rotary Position Embedding (RoPE)
    
    Args:
        pe (torch.Tensor): Positional encoding
        x (torch.Tensor): Input tensor of shape (batch_size, channels, length)
    
    Returns:
        torch.Tensor: Rotated input tensor
    """
    b, c, l = x.shape

    assert c % 2 == 0

    x_sin = torch.zeros_like(x)
    x_cos = torch.zeros_like(x)

    x_cos[:,0::2] = x[:,0::2]  * pe[1::2]
    x_cos[:,1::2] = x[:,1::2]  * pe[1::2]

    x_sin[:,0::2] = -x[:,1::2]  * pe[0::2]
    x_sin[:,1::2] = x[:,0::2]  * pe[0::2]

    roped_x = x_cos + x_sin

    return roped_x


def rope_2d(pe, x):
    """Apply 2D Rotary Position Embedding (RoPE)
    
    Args:
        pe (torch.Tensor): Positional encoding
        x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)
    
    Returns:
        torch.Tensor: Rotated input tensor
    """
    
    x1, x2 = x.chunk(2, dim=1)
    
    half_dim = pe.shape[0] // 2

    assert x.shape[2] == pe.shape[1] and x.shape[3] == pe.shape[2]

    x1_sin = torch.zeros_like(x1)
    x1_cos = torch.zeros_like(x1)

    x1_cos[:,0::2] = x1[:,0::2] * pe[1:half_dim+1:2,:,:].unsqueeze(0)
    x1_cos[:,1::2] = x1[:,1::2] * pe[1:half_dim+1:2,:,:].unsqueeze(0)

    x1_sin[:,0::2] = -x1[:,1::2] * pe[0:half_dim:2,:,:].unsqueeze(0)
    x1_sin[:,1::2] = x1[:,0::2] * pe[0:half_dim:2,:,:].unsqueeze(0)

    x1 = x1_cos + x1_sin

    x2_sin = torch.zeros_like(x2)
    x2_cos = torch.zeros_like(x2)

    x2_cos[:,0::2] = x2[:,0::2] * pe[half_dim+1::2,:,:].unsqueeze(0)
    x2_cos[:,1::2] = x2[:,1::2] * pe[half_dim+1::2,:,:].unsqueeze(0)

    x2_sin[:,0::2] = -x2[:,1::2] * pe[half_dim::2,:,:].unsqueeze(0)
    x2_sin[:,1::2] = x2[:,0::2] * pe[half_dim::2,:,:].unsqueeze(0)

    x2 = x2_cos + x2_sin

    return torch.cat([x1, x2], dim=1)

def adaptive_cube_split(x, split_dims, target_num, overlap=0.2):
    """
    自适应分割张量x，使得在split_dims指定的多个维度上，分割后patch的总数最接近target_num，
    并且每个patch在分割维度上是“正方体/正方形”，即每个分割维度的长度都相同，且均分。
    支持重叠区域以增强边界平滑。

    Args:
        x (torch.Tensor): 输入张量，形状任意
        split_dims (list[int]): 需要分割的维度索引（如[2,3]表示对H,W分割，[1,2,3]表示对C,H,W分割）
        target_num (float): 目标分割数量（如1e4）
        overlap (int or float): 重叠像素数（int，单位为像素）或比例（float，0~1），默认为0

    Returns:
        patches (torch.Tensor): 分割后的patch张量，形状为 (N, ..., patch_shape...)
        split_shape (list[int]): 每个分割维度的分割数，长度为D
        patch_indices (list[list[tuple]]): 每个分割维度的patch起止索引
    """
    import math

    shape = list(x.shape)
    dims = split_dims
    total_target = int(target_num)
    D = len(dims)
    if D < 1:
        raise ValueError("split_dims 至少包含一个维度")
    # 获取每个分割维度的长度
    dim_lens = [shape[d] for d in dims]
    min_len = min(dim_lens)
    # patch边长最大不能超过最小的分割维度
    best_n = [1] * D
    min_diff = float('inf')
    best_patch_size = None
    # 只考虑patch边长能整除所有分割维度
    for patch_size in range(1, min_len+1):
        if not all(l % patch_size == 0 for l in dim_lens):
            continue
        n_splits = [l // patch_size for l in dim_lens]
        total_patches = 1
        for n in n_splits:
            total_patches *= n
        diff = abs(total_patches - total_target)
        if diff < min_diff:
            min_diff = diff
            best_n = n_splits
            best_patch_size = patch_size
        if diff == 0:
            break
    n_splits = best_n
    patch_size = best_patch_size

    # 计算重叠像素数
    if isinstance(overlap, float) and 0 < overlap < 1:
        overlap_pix = int(patch_size * overlap)
    else:
        overlap_pix = int(overlap)
    overlap_pix = max(0, min(overlap_pix, patch_size - 1))  # 防止重叠过大

    # 计算每个分割维度的patch起止索引
    patch_indices = []
    for d, l in zip(dims, dim_lens):
        indices = []
        step = patch_size - overlap_pix
        if step <= 0:
            raise ValueError("overlap过大，导致patch步长<=0")
        pos = 0
        while pos + patch_size <= l:
            indices.append((pos, pos + patch_size))
            pos += step
        # 最后一个patch如果没有覆盖到结尾，则强制补一个
        if indices and indices[-1][1] < l:
            indices.append((l - patch_size, l))
        elif not indices:  # 维度太小
            indices.append((0, l))
        patch_indices.append(indices)

    # 构造所有patch的起止索引组合
    from itertools import product
    all_patch_coords = list(product(*patch_indices))
    num_blocks = len(all_patch_coords)

    # 生成patch
    patches = []
    for coord in all_patch_coords:
        # 构造切片对象
        slices = [slice(None)] * len(shape)
        for i, d in enumerate(dims):
            slices[d] = slice(coord[i][0], coord[i][1])
        patch = x[tuple(slices)]
        patches.append(patch)
    # 合并为一个大张量
    patches = torch.stack(patches, dim=0)

    patch_shapes = [p.shape for p in patches]
    if len(set(patch_shapes)) > 1:
        raise ValueError(f"所有patch的尺寸必须一致，但实际得到的patch尺寸有: {set(patch_shapes)}")

    return patches, [len(p) for p in patch_indices], patch_indices