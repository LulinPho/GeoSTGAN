import torch
import torch.utils.data
import torch.nn.functional as F
import logging
import random

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class PatchDataset(torch.utils.data.Dataset):
    """用于动态生成patch数据的Dataset类"""
    
    def __init__(self, features: torch.Tensor, labels: torch.Tensor, masks:torch.Tensor, coordinates: torch.Tensor, patch_size: int = 450, corner_sampling: bool = False, enhancement: bool = True):
        """
        初始化PatchDataset
        
        Args:
            features: 输入的TIF张量数据，形状为(T, C, H, W)
            labels: 输入的TIF张量数据，形状为(T, C, H, W)
            masks: 输入的TIF张量数据，形状为(T, C, H, W)
            coordinates: 采样坐标点，形状为(N, 2)，每行为(y, x)坐标
            patch_size: patch的大小，默认为450
            enhancement: 是否进行数据增强，默认为True
        """
        self.features = features
        self.labels = labels
        self.masks = masks
        self.coordinates = coordinates
        self.patch_size = patch_size
        self.corner_sampling = corner_sampling
        self.enhancement = enhancement
        self.half_patch = patch_size // 2
        _, _, self.H, self.W = features.shape
        logger.info(f"初始化PatchDataset: features形状={features.shape}, labels形状={labels.shape}, masks形状={masks.shape}, 坐标数量={len(coordinates)}")
            
    def __len__(self) -> int:
        """返回数据集中patch的总数"""
        return len(self.coordinates)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        获取指定索引位置的patch
        
        Args:
            idx: 坐标的索引
            
        Returns:
            对应位置的patch张量，形状为(T, C, patch_size, patch_size)
        """
        if self.corner_sampling:
            # 角采样：坐标为左上角
            y, x = self.coordinates[idx].tolist()
            position = torch.tensor([y, x])

            # 计算patch的边界（左上角采样）
            y_start = y
            y_end = y + self.patch_size
            x_start = x
            x_end = x + self.patch_size

            # 检查是否需要padding
            pad_y_start = max(0, -y_start)
            pad_y_end = max(0, y_end - self.H)
            pad_x_start = max(0, -x_start)
            pad_x_end = max(0, x_end - self.W)

            # 调整实际提取的边界
            y_start = max(0, y_start)
            y_end = min(self.H, y_end)
            x_start = max(0, x_start)
            x_end = min(self.W, x_end)

            # 提取patch
            feature_patch = self.features[:, :, y_start:y_end, x_start:x_end]
            label_patch = self.labels[:, :, y_start:y_end, x_start:x_end]
            mask_patch = self.masks[:, :, y_start:y_end, x_start:x_end]

            # 如果需要padding，进行填充
            if pad_y_start > 0 or pad_y_end > 0 or pad_x_start > 0 or pad_x_end > 0:
                feature_patch = F.pad(feature_patch, (pad_x_start, pad_x_end, pad_y_start, pad_y_end), 
                            mode='constant', value=0)
                label_patch = F.pad(label_patch, (pad_x_start, pad_x_end, pad_y_start, pad_y_end), 
                            mode='constant', value=0)
                mask_patch = F.pad(mask_patch, (pad_x_start, pad_x_end, pad_y_start, pad_y_end), 
                            mode='constant', value=0)
        else:
            # 获取坐标
            y, x = self.coordinates[idx].tolist()
            position = torch.tensor([y, x])
            
            # 计算patch的边界
            y_start = y - self.half_patch
            y_end = y + self.half_patch
            x_start = x - self.half_patch
            x_end = x + self.half_patch
            
            # 检查是否需要padding
            pad_y_start = max(0, -y_start)
            pad_y_end = max(0, y_end - self.H)
            pad_x_start = max(0, -x_start)
            pad_x_end = max(0, x_end - self.W)
            
            # 调整实际提取的边界
            y_start = max(0, y_start)
            y_end = min(self.H, y_end)
            x_start = max(0, x_start)
            x_end = min(self.W, x_end)
            
            # 提取patch
            feature_patch = self.features[:, :, y_start:y_end, x_start:x_end]
            label_patch = self.labels[:, :, y_start:y_end, x_start:x_end]
            mask_patch = self.masks[:, :, y_start:y_end, x_start:x_end]
            
            # 如果需要padding，进行填充
            if pad_y_start > 0 or pad_y_end > 0 or pad_x_start > 0 or pad_x_end > 0:
                feature_patch = F.pad(feature_patch, (pad_x_start, pad_x_end, pad_y_start, pad_y_end), 
                            mode='constant', value=0)
                label_patch = F.pad(label_patch, (pad_x_start, pad_x_end, pad_y_start, pad_y_end), 
                            mode='constant', value=0)
                mask_patch = F.pad(mask_patch, (pad_x_start, pad_x_end, pad_y_start, pad_y_end), 
                            mode='constant', value=0)
        # 数据增强措施
        if self.enhancement:
            # 随机旋转patch（以90°为单位）
            k = random.randint(0, 3)
            if k > 0:
                feature_patch = torch.rot90(feature_patch, k, dims=[-2, -1])
                label_patch = torch.rot90(label_patch, k, dims=[-2, -1])
                mask_patch = torch.rot90(mask_patch, k, dims=[-2, -1])
            # 随机水平或垂直翻转patch
            if random.random() < 0.5:
                feature_patch = torch.flip(feature_patch, dims=[-1])  # 水平翻转
                label_patch = torch.flip(label_patch, dims=[-1])  # 水平翻转
                mask_patch = torch.flip(mask_patch, dims=[-1])  # 水平翻转
            if random.random() < 0.5:
                feature_patch = torch.flip(feature_patch, dims=[-2])  # 垂直翻转
                label_patch = torch.flip(label_patch, dims=[-2])  # 垂直翻转
                mask_patch = torch.flip(mask_patch, dims=[-2])  # 垂直翻转
            
        return feature_patch, label_patch, mask_patch, position


