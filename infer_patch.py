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

def infer():
    """
    单张图片推理函数：使用get_data采样10个坐标，进行预测，将结果保存至visual/predicition中
    """
    import torch
    import numpy as np
    import tifffile
    import os
    import logging
    from models.gan import Generator
    from utils import get_data,visualize_temporal_comparison

    # 日志设置
    logger = logging.getLogger("infer_2020")
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    factors_data_path = "/root/autodl-fs/factors"
    labels_data_path = "/root/autodl-fs/label_floats"
    num_patches = 10

    # 采样10个patch
    batch_size = 10
    # get_data返回: x, y, mask, coordinates, labels
    features, labels, valid_mask, coordinates, invalid_coordinates, num_classes = get_data(factors_data_path, labels_data_path, num_patches)
    features = features.to('cpu')
    valid_mask = valid_mask.to('cpu')
    labels = labels.to('cpu')
    coordinates = coordinates.to('cpu')

    norm_params = torch.load("checkpoints/features_normalize_params.pth", weights_only=True)
    features = normalize_tensor_with_params(features, norm_params)

    dataset = PatchDataset(features, labels, valid_mask, coordinates, 512, corner_sampling=True, enhancement=False)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # 加载生成器
    generator = Generator(
        hidden_channels=48,
        num_classes=num_classes,
        feature_decoder=True,
        features_channels=features.shape[1]- valid_mask.shape[1]
    ).to(device)

    # 触发conv1初始化
    dummy_input = torch.randn(1, 1, features.shape[1]+ valid_mask.shape[1], 512, 512).to(device)
    with torch.no_grad():
        _ = generator(dummy_input)

    # 加载预训练权重
    checkpoint = torch.load("checkpoints/stgan/iter900.pth", weights_only=True)
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

    outputs = []
    reals = []

    # 推理
    with torch.no_grad():
        for batch_idx, (x, y, mask, coordinates) in enumerate(dataloader):
            x = x[:, :4].to(device)
            y = y.to(device)
            mask = mask.to(device)
            coordinates = coordinates.to(device)
            x_gen = torch.cat([x, mask], dim=2)
            output, _ = generator(x_gen)  # [B, T, C, H, W]
            output = output.detach().cpu()  # [B, T, C, H, W]
            outputs.append(output)
            reals.append(y)

    outputs = torch.cat(outputs, dim=0)
    reals = torch.cat(reals, dim=0)

    # 保存结果
    save_dir = "visual/predicition"
    os.makedirs(save_dir, exist_ok=True)

    for i in range(batch_size):
        real_full = reals[i].argmax(dim=1).cpu()  # [T, H, W]
        fake_full = outputs[i].argmax(dim=1).cpu() 
        vis_path = os.path.join(save_dir, f"sample_{i}_compare.png")
        visualize_temporal_comparison(
            tensor1=real_full,
            tensor2=fake_full,
            save_path=vis_path,
            title1="True",
            title2="Fake"
        )
        logger.info(f"第{i}个patch推理可视化已保存到: {vis_path}")

    logger.info("10个patch单张图片推理完成！")

    