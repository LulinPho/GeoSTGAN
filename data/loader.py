import os
import numpy as np
import torch
import rasterio
from typing import List, Tuple
import logging
import glob
import torch.nn.functional as F
import gc
from .dataset import PatchDataset

# 配置日志记录器
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)



def find_common_valid_positions_class(landuse_data: torch.Tensor):
    """找到在所有年份都不为0的公共有效位置，并返回无效坐标

    Args:
        landuse_data: 张量，形状为(T,C,H,W)
        
    Returns:
        coordinates: 有效坐标数组，形状为(N, 2)，每行为(y, x)坐标
        invalid_coordinates: 无效坐标数组，形状为(M, 2)，每行为(y, x)坐标
        valid_mask: 有效掩膜，形状为(T, C, H, W)
    """
    # landuse_data: (T, C, H, W)
    T, C, H, W = landuse_data.shape

    # 初始化公共有效掩膜，形状为(H, W)，初始为True
    common_valid_mask = torch.ones((H, W), dtype=torch.bool, device=landuse_data.device)

    for t in range(T):
        # 当前年份的多波段数据，形状为(C, H, W)
        data = landuse_data[t].to(torch.uint8)
        # 判断每个像元在波段7是否不为1，结果为(H, W)
        current_valid = (data[6] != 1)
        # 更新公共有效掩膜（所有年份都必须有效）
        common_valid_mask = common_valid_mask & current_valid

    # 找到公共有效位置的坐标
    valid_positions = torch.nonzero(common_valid_mask, as_tuple=False)
    # valid_positions: (N, 2)，每行是(y, x)

    # 找到无效位置的坐标（补集）
    invalid_positions = torch.nonzero(~common_valid_mask, as_tuple=False)
    # invalid_positions: (M, 2)，每行是(y, x)

    # 转为numpy数组
    coordinates = valid_positions.cpu().numpy()
    invalid_coordinates = invalid_positions.cpu().numpy()

    logger.info(f"在所有年份都有效的公共位置数量: {len(coordinates)}")
    logger.info(f"在所有年份都无效的公共位置数量: {len(invalid_coordinates)}")

    # 有效区域：第1-6通道为1，第7通道为0；无效区域则相反
    valid_mask = torch.zeros_like(landuse_data, dtype=torch.uint8)
    for t in range(T):
        # 当前年份的多波段数据，形状为(C, H, W)
        data = landuse_data[t].to(torch.uint8)
        # 有效区域：波段7不为1
        current_valid = (data[6] != 1)  # (H, W)
        # 对有效区域，第1-6通道为1，第7通道为0
        for c in range(6):
            valid_mask[t, c][current_valid] = 1
            valid_mask[t, c][~current_valid] = 0
        valid_mask[t, 6][current_valid] = 0
        valid_mask[t, 6][~current_valid] = 1
    # 返回有效坐标、无效坐标和valid_mask
    return coordinates, invalid_coordinates, valid_mask


def find_common_valid_positions_float(landuse_data: torch.Tensor) -> np.ndarray:
    """找到在所有年份都不为0的公共有效位置
    
    Args:
        landuse_data: 张量，形状为(年份数, 波段数, 高度, 宽度)
        
    Returns:
        坐标数组，形状为(N, 2)，每行为(y, x)坐标
    """

    threshold = 0.1  # 可根据实际数据调整
    
    # 第一步：在波段维度上求和(:-1)
    band_sum = landuse_data[:,1:].sum(dim=1)  # shape: (年份数, 高度, 宽度)
    # 第二步：判断和是否显著非零
    valid_mask = (band_sum.abs() > threshold)  # shape: (年份数, 高度, 宽度)

    logger.info(f"每个年份的有效像素数量: {valid_mask.sum(dim=(1,2)).tolist()}")

    # 第三步：在年份维度上判断所有年份都有效
    common_valid_mask = valid_mask.all(dim=0)  # shape: (高度, 宽度)

    logger.info(f"所有年份都有效的像素数量: {common_valid_mask.sum().item()}")

    # 第四步：获取(y, x)坐标
    valid_positions = torch.nonzero(common_valid_mask, as_tuple=False)
    # valid_positions: (N, 2)，每行是(y, x)

    # 转为numpy数组
    coordinates = valid_positions.cpu().numpy()

    logger.info(f"有效位置数量: {len(coordinates)}")

    return coordinates


def sample_positions(coordinates: np.ndarray, k: int) -> np.ndarray:
    """从公共有效位置中随机抽样k个位置
    
    Args:
        coordinates: 坐标数组，形状为(N, 2)
        k: 要抽样的位置数量
        
    Returns:
        抽样后的坐标数组，形状为(k, 2)
    """
    if len(coordinates) == 0:
        logger.warning("没有找到公共有效位置")
        return np.array([])
    
    if len(coordinates) <= k:
        # 如果有效位置数量少于k，则全部选择
        logger.info(f"有效位置数量({len(coordinates)})少于k({k})，选择所有位置")
        return coordinates
    else:
        # 随机选择k个位置
        selected_indices = np.random.choice(len(coordinates), k, replace=False)
        selected_coordinates = coordinates[selected_indices]
        logger.info(f"从{len(coordinates)}个有效位置中随机选择{k}个")
        return selected_coordinates


if __name__ == "__main__":
    from .utils import load_hierarchical_tif
    
    factors_data = load_hierarchical_tif("/root/autodl-fs/factors")

    labels_data = load_hierarchical_tif("/root/autodl-fs/onehot")
    
    if factors_data is None or labels_data is None:
        print("数据加载失败")
        exit(1)
    
    valid_positions, invalid_positions, valid_mask = find_common_valid_positions_class(labels_data)

    coordinates = sample_positions(valid_positions, 10000)

    print(f"valid_mask的形状: {valid_mask.shape}")
