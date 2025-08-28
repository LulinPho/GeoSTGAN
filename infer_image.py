import torch
from collections import OrderedDict
from utils import get_data, create_positions_sequence, visualize_temporal_comparison, normalize_tensor_with_params
from models.gan import Generator
from data.dataset import PatchDataset
from torch.utils.data import DataLoader
import tifffile
import numpy as np
import os
import logging
import time
from tqdm import tqdm

def infer_2020():
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
        features_channels=features.shape[1]- valid_mask.shape[1]
    ).to(device)

    patch_size = 512
    overlap = 256
    batch_size = 1

    coordinates,_ = create_positions_sequence(features, patch_size, overlap, batch_size)

    # 触发conv1初始化
    dummy_input = torch.randn(1, 1, features.shape[1]+ valid_mask.shape[1], patch_size, patch_size).to(device)
    with torch.no_grad():
        _ = generator(dummy_input)

    # 加载预训练权重
    checkpoint = torch.load("checkpoints/stgan_20250813/iter1500.pth", weights_only=True)
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

    dataset = PatchDataset(features, labels, valid_mask, coordinates, patch_size, corner_sampling=True, enhancement=False)
    dataloader = DataLoader(dataset, batch_size=24, shuffle=False)

    # 获取原始图像尺寸
    img_size = (features.shape[2], features.shape[3])  # (H, W)
    
    # 创建所有时点的全图张量用于存储预测结果
    full_image = torch.zeros((labels.shape[0], num_classes, img_size[0], img_size[1]), device=device)
    # 新增：用于统计每个像素被赋值的次数
    full_image_weight = torch.zeros((labels.shape[0], num_classes, img_size[0], img_size[1]), device=device)

    # 创建可视化目录
    vis_dir = os.path.join("visual", f"infer_2020")
    os.makedirs(vis_dir, exist_ok=True)

    # 整图推理
    generator.eval()
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
            enumerate(dataloader), total=len(dataloader), desc="Inference", ncols=80
        ):
            x = x[:,:4].to(device)
            mask_patch = mask_patch.to(device)
            coordinates_patch = coordinates_patch.to(device)
            
            x_gen = torch.cat([x, mask_patch], dim=2)
            output, next_feature = generator(x_gen)
            
            norm_params = torch.load("checkpoints/features_normalize_params.pth", weights_only=True)
            next_feature = normalize_tensor_with_params(next_feature, norm_params)
            
            B, T, C, patch_h, patch_w = output.shape
            
            for i in range(B):
                y_pos, x_pos = coordinates_patch[i]
                y_pos = int(y_pos.item())
                x_pos = int(x_pos.item())
                patch_label = output[i]
                
                full_image[:,:,y_pos:y_pos+patch_h, x_pos:x_pos+patch_w] += patch_label
                full_image_weight[:,:,y_pos:y_pos+patch_h, x_pos:x_pos+patch_w] += gaussian_window

    # 用赋值次数做平均
    full_image = full_image / full_image_weight
    
    valid_mask = torch.sum(labels, dim=1) > 0.5

    # 真实标签全图
    real_full = labels.argmax(dim=1).cpu()  # [T, H, W]
    # 预测结果全图
    fake_full = full_image.argmax(dim=1).cpu()  # [T, H, W]

    # 后处理
    fake_full[~valid_mask] = 6

    real_full[~valid_mask] = 6

    # 可视化
    vis_path = os.path.join(vis_dir, f"full.png")
    visualize_temporal_comparison(
        tensor1=real_full,
        tensor2=fake_full,
        save_path=vis_path,
        title1="True",
        title2="Fake"
    )

    # 保存最终结果
    os.makedirs("outputs", exist_ok=True)

    # 保存每一个时间点的result和real文件
    T = fake_full.shape[0]
    for t in range(T):
        result = fake_full[t].cpu().numpy()
        real = real_full[t].cpu().numpy()
        # 保存预测结果
        save_tif_path = f"outputs/infer_{2000 + (t+1)*5}.tif"
        tifffile.imwrite(save_tif_path, result.astype(np.uint8))
        logger.info(f"第{t}个时间点的预测结果已保存为: {save_tif_path}")
        # 保存真实标签
        save_real_tif_path = f"outputs/real_{2000 + (t+1)*5}.tif"
        tifffile.imwrite(save_real_tif_path, real.astype(np.uint8))
        logger.info(f"第{t}个时间点的真实标签已保存为: {save_real_tif_path}")

    logger.info("推理完成！")

if __name__ == "__main__":
    infer_2020()
