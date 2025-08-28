import torch
from collections import OrderedDict
from utils import get_data, create_positions_sequence, normalize_tensor_with_params
from models.gan import Generator
from data.dataset import PatchDataset
from torch.utils.data import DataLoader
import tifffile
import numpy as np
import os
import logging
import time
from tqdm import tqdm
from visual import visualize_sequence_tensor, visualize_tensor
import pandas as pd

def augment_with_invalid_by_valid_mask(prob_maps: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    """使用确定性的有效性掩膜(valid_mask)将 [C,H,W] 扩展为含无效类 [C+1,H,W]。
    - 若 valid_mask=1（有效）：保留六类概率，并将第七类置0
    - 若 valid_mask=0（无效）：六类置0，第七类置1
    valid_mask 形状可为 [1,H,W] 或 [H,W]
    """
    with torch.no_grad():
        # 对齐device与dtype，避免CPU/GPU混用
        valid_mask = valid_mask.to(device=prob_maps.device, dtype=prob_maps.dtype)
        C, H, W = prob_maps.shape
        out = torch.zeros(C + 1, H, W, device=prob_maps.device, dtype=prob_maps.dtype)
        out[:C] = prob_maps * valid_mask
        out[C:C+1] = 1.0 - valid_mask
    return out

def infer_sequence_aligned_step(generator: torch.nn.Module,
                               xseq: torch.Tensor,
                               mask: torch.Tensor,
                               coordinates: torch.Tensor,
                               num_classes: int,
                               feature_channels: int,
                               device: torch.device,
                               img_size: tuple,
                               patch_size: int = 512,
                               batch_size: int = 16,
                               show_progress: bool = False):
    """
    对齐infer_sequence.py的推理过程
    输入：
        xseq: [4, C, H, W] - 滑动窗口的特征序列
        mask: [4, N, H, W] - 对应的掩膜
        coordinates: 坐标序列
        
    返回：
        next_label_full: [num_classes, H, W] - 预测的标签
        next_feature_full: [feature_channels, H, W] - 预测的特征
    """
    generator.eval()
    
    labels = torch.zeros(4, num_classes, img_size[0], img_size[1])  # dummy labels
    dataset = PatchDataset(xseq.cpu(), labels, mask.cpu(), coordinates.cpu(), 
                          patch_size, corner_sampling=True, enhancement=False)
    
    next_label_full = torch.zeros((num_classes, img_size[0], img_size[1]), device=device)
    next_feature_full = torch.zeros((feature_channels, img_size[0], img_size[1]), device=device)
    next_label_count = torch.zeros((num_classes, img_size[0], img_size[1]), device=device)
    next_feature_count = torch.zeros((feature_channels, img_size[0], img_size[1]), device=device)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # 构造高斯窗，中心为1，边缘为0.5
    with torch.no_grad():
        gaussian_window = torch.ones((patch_size, patch_size), dtype=torch.float32).to(device)
        center = torch.tensor(patch_size // 2).to(torch.float32)
        sigma = torch.tensor(patch_size / 4.0).to(torch.float32)  # 控制高斯分布宽度
        for i in range(patch_size):
            for j in range(patch_size):
                dist = ((i - center) ** 2 + (j - center) ** 2) ** 0.5
                # 归一化到[0,1]，中心为1，边缘为0.5
                value = 0.5 + 0.5 * torch.exp(-dist ** 2 / (2 * sigma ** 2))
                gaussian_window[i, j] = value
    
    with torch.no_grad():
        for batch_idx, (x, _, mask_patch, coordinates_patch) in tqdm(
            enumerate(dataloader), total=len(dataloader), desc="Inference", ncols=80, disable=(not show_progress)
        ):
            x = x.to(device)
            mask_patch = mask_patch.to(device)
            coordinates_patch = coordinates_patch.to(device)
            
            x_gen = torch.cat([x, mask_patch], dim=2)
            output, next_feature = generator(x_gen)
            
            norm_params = torch.load("checkpoints/features_normalize_params.pth", weights_only=True)
            next_feature = normalize_tensor_with_params(next_feature, norm_params)
            
            B, T, C, patch_h, patch_w = output.shape
            
            output = output[:, -1]  # [B, C, patch_h, patch_w]
            next_feature = next_feature[:, -1]  # [B, F, patch_h, patch_w]
            
            for i in range(B):
                y_pos, x_pos = coordinates_patch[i]
                y_pos = int(y_pos.item())
                x_pos = int(x_pos.item())
                patch_label = output[i]  # [num_classes, patch_h, patch_w]
                patch_feat = next_feature[i]  # [feature_channels, patch_h, patch_w]
                
                next_label_full[:, y_pos:y_pos+patch_h, x_pos:x_pos+patch_w] += patch_label
                next_label_count[:, y_pos:y_pos+patch_h, x_pos:x_pos+patch_w] += gaussian_window
                next_feature_full[:, y_pos:y_pos+patch_h, x_pos:x_pos+patch_w] += patch_feat
                next_feature_count[:, y_pos:y_pos+patch_h, x_pos:x_pos+patch_w] += gaussian_window
    
    next_label_count = torch.where(next_label_count == 0, torch.ones_like(next_label_count), next_label_count)
    next_label_full = next_label_full / next_label_count
    next_label_full = torch.argmax(next_label_full, dim=0)
    
    next_feature_count = torch.where(next_feature_count == 0, torch.ones_like(next_feature_count), next_feature_count)
    next_feature_full = next_feature_full / next_feature_count
    
    mask_last = mask[-1]  # [N, H, W]
    # 无效像元：所有类别掩膜之和为0的位置
    zero_mask = (mask_last.sum(dim=0) < 0.5)
    next_feature_full[:, zero_mask] = 0
    
    return next_label_full, next_feature_full

def infer_sequence(time_steps):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 设置日志
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())
    log_path = f"logs/infer_2020_{time.strftime('%Y%m%d_%H%M%S')}.log"
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger.addHandler(logging.FileHandler(log_path))

    # 加载2020年因子和标签数据
    factors_data_path = "/root/autodl-fs/factors"
    labels_data_path = "/root/autodl-fs/label_floats"
    num_patches = 1  # 这里只是为了get_data接口，实际我们会用整图
    features, labels, valid_mask, _, _, num_classes = get_data(factors_data_path, labels_data_path, num_patches)
    features = features.to('cpu')
    valid_mask = valid_mask.to('cpu')

    norm_params = torch.load("checkpoints/features_normalize_params.pth", weights_only=True)
    features = normalize_tensor_with_params(features, norm_params)

    # 加载生成器
    generator = Generator(
        hidden_channels=32,
        num_classes=num_classes,
        feature_decoder=True,
        features_channels=features.shape[1] - valid_mask.shape[1]
    ).to(device)

    patch_size = 512
    overlap = 256  # 与constraint.py保持一致
    batch_size = 32  # 与constraint.py保持一致

    coordinates, _ = create_positions_sequence(features, patch_size, overlap, 1)

    # 触发conv1初始化
    dummy_input = torch.randn(1, 1, features.shape[1] + valid_mask.shape[1], patch_size, patch_size).to(device)
    with torch.no_grad():
        _ = generator(dummy_input)

    # 加载预训练权重
    checkpoint = torch.load("checkpoints/stgan/iter1500.pth", weights_only=True)
    generator_state_dict = OrderedDict()
    for k, v in checkpoint['generator_state_dict'].items():
        # 移除module前缀
        if k.startswith('module.'):
            new_key = k[7:]  # 移除 'module.' 前缀
        else:
            new_key = k
        generator_state_dict[new_key] = v

    try:
        generator.load_state_dict(generator_state_dict, strict=True)
        print("生成器加载成功！")
    except Exception as e:
        print(f"生成器加载失败: {e}")
        # 打印当前模型的键
        current_keys = set(generator.state_dict().keys())
        saved_keys = set(generator_state_dict.keys())
        print(f"当前模型键数量: {len(current_keys)}")
        print(f"保存的键数量: {len(saved_keys)}")
        missing_keys = saved_keys - current_keys
        unexpected_keys = current_keys - saved_keys
        if missing_keys:
            print(f"缺失的键: {list(missing_keys)[:10]}...")
        if unexpected_keys:
            print(f"意外的键: {list(unexpected_keys)[:10]}...")

    # 获取原始图像尺寸
    img_size = (features.shape[2], features.shape[3])  # (H, W)
    feature_channels = features.shape[1] - valid_mask.shape[1]

    # 创建可视化目录
    vis_dir = os.path.join("visual", f"infer_2020")
    os.makedirs(vis_dir, exist_ok=True)

    # 初始化滑动窗口xseq，取前4个时刻的特征
    xseq = features[1:].clone()  # [4, C, H, W]
    mask = valid_mask[-1].unsqueeze(0).repeat(4, 1, 1, 1)  # [4, N, H, W]

    # 用于保存每一步的预测结果
    pred_imgs = []
    counts = {}

    for t in range(time_steps):
        logger.info(f"开始时序步 {t+1} 推理...")
        
        # 使用对齐的推理函数
        next_label_full, next_feature_full = infer_sequence_aligned_step(
            generator=generator,
            xseq=xseq,
            mask=mask,
            coordinates=coordinates,
            num_classes=num_classes,
            feature_channels=feature_channels,
            device=device,
            img_size=img_size,
            patch_size=patch_size,
            batch_size=batch_size,
            show_progress=True
        )
        
        # 统计当前时刻各用地类型的像元数量（仅限有效掩膜区域）
        land_type_map = next_label_full  # [H, W]
        # valid_mask: [H, W]，有效区域为1
        valid_mask_current = (mask[-1].sum(dim=0) > 0.5).to(device)
        class_counts = []
        for cls in range(num_classes):
            count = ((land_type_map == cls) & valid_mask_current).sum().item()
            class_counts.append(count)
        print(f"时序步 {t+1} 各用地类型数量（仅统计有效区域）：")
        for cls in range(num_classes):
            print(f"  类别 {cls}: {class_counts[cls]} 个像元")

        counts[t] = class_counts

        # 更新xseq，滑动窗口：去掉最早的，加上新预测的
        # 将next_label_full转换为one-hot编码
        next_label_onehot = torch.zeros(num_classes, img_size[0], img_size[1], device=device)
        next_label_onehot.scatter_(0, next_label_full.unsqueeze(0), 1.0)
        
        next_feature_cat = torch.cat([next_label_onehot, next_feature_full], dim=0)  # [C, H, W]
        xseq = torch.cat([xseq[1:], next_feature_cat.unsqueeze(0).cpu()], dim=0)
        
        # 准备可视化数据
        probs = augment_with_invalid_by_valid_mask(next_label_onehot, valid_mask_current)
        pred_imgs.append(probs.cpu())
    
    # 可视化
    vis_path = os.path.join(vis_dir, f"sequence.png")
    pred_imgs = torch.stack(pred_imgs, dim=0)
    visualize_sequence_tensor(pred_imgs, vis_path, titles=["2025", "2030", "2035", "2040"][:time_steps])
    print(f"已保存预测序列可视化到: {vis_path}")

    # 保存结果到pth文件
    save_path = os.path.join(vis_dir, f"sequence.pth")
    torch.save(pred_imgs, save_path)
    print(f"已保存预测序列到: {save_path}")

    

    # 将counts转换为DataFrame
    counts_array = [counts[i] for i in range(time_steps)]  # 只取实际预测的时间步
    df = pd.DataFrame(
        data=counts_array,  # 行：时间点，列：类别
        index=[f"时刻{step+1}" for step in range(time_steps)],
        columns=[f"类别{cls}" for cls in range(num_classes)]
    )

    # 保存为Excel文件
    excel_save_path = os.path.join(vis_dir, "land_type_counts.xlsx")
    df.to_excel(excel_save_path)
    print(f"用地类型数量统计已保存为Excel: {excel_save_path}")

    logger.info("推理完成！")

if __name__ == "__main__":
    infer_sequence(time_steps=1)

