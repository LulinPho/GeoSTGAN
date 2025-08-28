"""
计算预训练阶段的EWC状态
在step_train预训练完成后，计算Fisher信息矩阵以保护预训练学到的知识
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
import logging
import os
import time
from collections import OrderedDict
import numpy as np
from tqdm import tqdm

from models.gan import Generator, DualDiscriminator
from losses import mask_penalty, generator_loss, discriminator_loss
from data.dataset import PatchDataset
from utils import get_data, load_mask, create_positions_sequence, normalize_tensor_with_params
from data.utils import load_hierarchical_tif
from ewc import EWC, create_ewc
from adaptive_training_scheduler import AdaptiveTrainingScheduler, calculate_pixel_accuracy

def compute_pretrain_fisher_information(generator, discriminator, dataloader, device, num_samples=1000):
    """
    计算预训练阶段的Fisher信息矩阵，使用与step_train.py相同的损失定义和加权方案
    
    Args:
        generator: 生成器模型
        discriminator: 判别器模型
        dataloader: 预训练数据加载器
        device: 设备
        num_samples: 用于计算Fisher信息的样本数量
    
    Returns:
        fisher_info: Fisher信息矩阵
    """
    generator.eval()
    discriminator.eval()
    # 冻结判别器参数，避免在反传时为判别器计算梯度
    for p in discriminator.parameters():
        p.requires_grad_(False)
    
    # 初始化Fisher信息
    fisher_info = {}
    for name, param in generator.named_parameters():
        if param.requires_grad:
            fisher_info[name] = torch.zeros_like(param.data)
    
    logging.getLogger(__name__).info(f"开始计算Fisher信息矩阵，使用 {num_samples} 个样本...")
    
    # 初始化自适应训练调度器（与step_train.py保持一致）
    scheduler = AdaptiveTrainingScheduler()
    
    # 计算Fisher信息
    sample_count = 0
    batch_count = 0

    pbar = tqdm(dataloader, desc="计算Fisher信息")
    # 预先加载特征归一化参数，避免循环内重复IO
    try:
        norm_params = torch.load("checkpoints/features_normalize_params.pth", weights_only=True)
    except Exception:
        norm_params = None
    for batch_idx, batch in enumerate(pbar):
        if sample_count >= num_samples:
            break
            
        # 获取数据（与step_train.py保持一致）
        x = batch[0].to(device)
        x = x[:,:-1] # 去掉最后一期
        x_l = batch[0][:,-1,6:,:,:].to(device) # 与step_train一致：最后一期的特征通道从6开始
        y = batch[1].to(device)
        y_l = y[:,-1,:,:,:]
        mask = batch[2].to(device)
        mask_l = mask[:,-1,:,:,:]

        # 创建有效掩膜（与step_train.py保持一致）：按像元是否存在任一有效通道来确定
        valid_sum = mask_l.sum(dim=1, keepdim=True)  # [B,1,H,W]
        valid_set_mask = (valid_sum > 0.5).float().expand_as(mask_l)
        
        batch_size = x.size(0)
        current_samples = min(batch_size, num_samples - sample_count)
        
        x = x[:current_samples]
        y = y[:current_samples]
        mask = mask[:current_samples]
        x_l = x_l[:current_samples]
        y_l = y_l[:current_samples]
        mask_l = mask_l[:current_samples]
        valid_set_mask = valid_set_mask[:current_samples]
        
        x_gen = torch.cat([x, mask], dim=2)
        fake_samples, fake_features = generator(x_gen)
        fake_samples_l = fake_samples[:,-1,:,:,:]
        fake_features_l = fake_features[:,-1,:,:,:]
        
        res = discriminator(x_gen, fake_samples_l, mode='both')
        patch_fake_validity, _ = res['patch']
        global_fake_validity, _ = res['global']
        
        # 掩膜化分类损失（与step_train一致）
        valid_pixel_mask = (y_l.sum(dim=1, keepdim=True) > 0.5).float()
        focal_loss, loss_bce_patch, loss_bce_global = generator_loss(
            fake_samples_l, y_l, patch_fake_validity, global_fake_validity, valid_pixel_mask=valid_pixel_mask
        )
        
        if norm_params is not None:
            fake_features_l = normalize_tensor_with_params(fake_features_l, norm_params)
        # 掩膜化特征MSE（与step_train一致）
        mse_per_pixel = (fake_features_l - x_l) ** 2
        mse_pixel = mse_per_pixel.mean(dim=1, keepdim=True)
        denom = valid_pixel_mask.sum().clamp_min(1.0)
        feature_loss = (mse_pixel * valid_pixel_mask).sum() / denom
        
        mask_loss = mask_penalty(fake_samples_l, valid_set_mask, alpha=0.5)
        
        current_pixel_accuracy = calculate_pixel_accuracy(fake_samples_l, y_l, valid_set_mask)
        
        adaptive_weights = scheduler.get_loss_weights(0, current_pixel_accuracy)
        
        loss_dict = {
            'focal': focal_loss,
            'patch_bce': loss_bce_patch,
            'global_bce': loss_bce_global, 
            'mask': mask_loss,
            'feature': feature_loss
        }
        
        # 计算加权总损失（与step_train.py保持一致）
        total_loss = (
            adaptive_weights['focal'] * loss_dict['focal'] +
            adaptive_weights['patch_bce'] * loss_dict['patch_bce'] +
            adaptive_weights['global_bce'] * loss_dict['global_bce'] +
            adaptive_weights['mask'] * loss_dict['mask'] +
            adaptive_weights['feature'] * loss_dict['feature']
        )
        
        # 计算梯度
        generator.zero_grad()
        total_loss.backward()
        
        # 累积Fisher信息
        for name, param in generator.named_parameters():
            if param.requires_grad and param.grad is not None:
                fisher_info[name] += param.grad.data ** 2
        
        sample_count += current_samples
        batch_count += 1
        
        # 更新进度条
        pbar.set_postfix({
            "Samples": f"{sample_count}/{num_samples}",
            "Batches": batch_count,
            "Loss": f"{total_loss.item():.4f}",
            "PixAcc": f"{current_pixel_accuracy:.3f}",
            "Focal": f"{adaptive_weights['focal']:.3f}",
            "PatchBCE": f"{adaptive_weights['patch_bce']:.3f}",
            "GlobalBCE": f"{adaptive_weights['global_bce']:.3f}",
            "Mask": f"{adaptive_weights['mask']:.3f}",
            "Feature": f"{adaptive_weights['feature']:.3f}"
        })

    # 平均Fisher信息
    for name in fisher_info:
        fisher_info[name] /= sample_count
    
    print(f"Fisher信息矩阵计算完成，使用了 {sample_count} 个样本")
    
    return fisher_info


def _setup_dist(rank: int, world_size: int):
    os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
    os.environ.setdefault('MASTER_PORT', '29588')
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)


def _cleanup_dist():
    if dist.is_initialized():
        dist.destroy_process_group()


def _distributed_fisher_worker(
    rank: int,
    world_size: int,
    dataset: PatchDataset,
    batch_size: int,
    feature_channels: int,
    num_classes: int,
    checkpoint_path: str,
    num_samples: int,
    fisher_tmp_path: str,
):
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    _setup_dist(rank, world_size)
    try:
        # 构建本地DataLoader
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
        dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=2, pin_memory=True)

        # 实例化模型并加载权重
        generator = Generator(
            hidden_channels=32,
            num_classes=num_classes,
            feature_decoder=True,
            features_channels=feature_channels,
        ).to(device)
        discriminator = DualDiscriminator(
            hidden_channels=64, 
            block_num=512,
            use_soft_input=True,  
            temperature=1.0       
        ).to(device)

        # 加载检查点
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)

        # 获取数据集信息
        features = dataset.features
        labels = dataset.labels
        patch_size = dataset.patch_size
            
        # 触发conv1初始化
        dummy_input = torch.randn(1, 1, features.shape[1] + labels.shape[1], patch_size, patch_size).to(device)
        with torch.no_grad():
            _ = generator(dummy_input)

        dummy_input_1 = torch.randn(1,4, num_classes * 2 + feature_channels, patch_size, patch_size).to(device)
        dummy_input_2 = torch.randn(1,num_classes, patch_size, patch_size).to(device)
        with torch.no_grad():
            _ = discriminator(dummy_input_1, dummy_input_2, mode='both')

        from collections import OrderedDict
        generator_state_dict = OrderedDict()
        for k, v in checkpoint['generator_state_dict'].items():
            if k.startswith('module.'):
                k = k[7:]
            if k.startswith('terraflow.'):
                k = 'gru.' + k
            generator_state_dict[k] = v
        generator.load_state_dict(generator_state_dict, strict=True)

        discriminator_state_dict = OrderedDict()
        for k, v in checkpoint['dual_discriminator_state_dict'].items():
            if k.startswith('module.'):
                k = k[7:]
            discriminator_state_dict[k] = v
        discriminator.load_state_dict(discriminator_state_dict, strict=True)

        # 计算本rank的Fisher并规约
        generator.eval()
        discriminator.eval()
        for p in discriminator.parameters():
            p.requires_grad_(False)

        # 初始化本地Fisher
        fisher_local = {}
        for name, param in generator.named_parameters():
            if param.requires_grad:
                fisher_local[name] = torch.zeros_like(param.data, device=device)

        # 初始化自适应训练调度器（与step_train.py保持一致）
        scheduler = AdaptiveTrainingScheduler()

        sample_count_local = 0
        pbar = tqdm(dataloader, desc=f"[Rank {rank}] 计算Fisher信息", disable=(rank != 0))
        # 预先加载特征归一化参数
        try:
            norm_params = torch.load("checkpoints/features_normalize_params.pth", weights_only=True)
        except Exception:
            norm_params = None
        for batch_idx, batch in enumerate(pbar):
            if sample_count_local >= num_samples:
                break

            # 获取数据（与step_train.py保持一致）
            x = batch[0].to(device, non_blocking=True)
            x = x[:,:-1] # 去掉最后一期
            x_l = batch[0][:,-1,6:,:,:].to(device, non_blocking=True) # 与step_train一致：最后一期特征通道从6开始
            y = batch[1].to(device, non_blocking=True)
            y_l = y[:,-1,:,:,:]
            mask = batch[2].to(device, non_blocking=True)
            mask_l = mask[:,-1,:,:,:]

            # 创建有效掩膜（与step_train.py保持一致）：像元层面的有效性判断
            valid_sum = mask_l.sum(dim=1, keepdim=True)  # [B,1,H,W]
            valid_set_mask = (valid_sum > 0.5).float().expand_as(mask_l)

            batch_size_actual = x.size(0)
            remaining = max(0, num_samples - sample_count_local)
            current = min(batch_size_actual, remaining)
            if current <= 0:
                break
            x = x[:current]
            y = y[:current]
            mask = mask[:current]
            x_l = x_l[:current]
            y_l = y_l[:current]
            mask_l = mask_l[:current]
            valid_set_mask = valid_set_mask[:current]

            # 前向传播（与step_train.py保持一致）
            x_gen = torch.cat([x, mask], dim=2)
            fake_samples, fake_features = generator(x_gen)
            fake_samples_l = fake_samples[:,-1,:,:,:]
            fake_features_l = fake_features[:,-1,:,:,:]
            
            # 计算判别器输出（与step_train.py保持一致）
            res = discriminator(x_gen, fake_samples_l, mode='both')
            patch_fake_validity, _ = res['patch']
            global_fake_validity, _ = res['global']
            
            # 计算生成器损失（与step_train.py保持一致）
            valid_pixel_mask = (y_l.sum(dim=1, keepdim=True) > 0.5).float()
            focal_loss, loss_bce_patch, loss_bce_global = generator_loss(
                fake_samples_l, y_l, patch_fake_validity, global_fake_validity, valid_pixel_mask=valid_pixel_mask
            )
            if norm_params is not None:
                fake_features_l = normalize_tensor_with_params(fake_features_l, norm_params)
            # 与step_train一致：仅在有效像元上计算特征MSE
            mse_per_pixel = (fake_features_l - x_l) ** 2  # [B,C,H,W]
            mse_pixel = mse_per_pixel.mean(dim=1, keepdim=True)  # [B,1,H,W]
            denom = valid_pixel_mask.sum().clamp_min(1.0)
            feature_loss = (mse_pixel * valid_pixel_mask).sum() / denom
            
            # 计算掩膜约束损失（与step_train.py保持一致）
            mask_loss = mask_penalty(fake_samples_l, valid_set_mask, alpha=0.5)
            
            # 计算性能指标（与step_train.py保持一致）
            current_pixel_accuracy = calculate_pixel_accuracy(fake_samples_l, y_l, valid_set_mask)
            
            # 获取自适应权重（与step_train.py保持一致）
            adaptive_weights = scheduler.get_loss_weights(0, current_pixel_accuracy)  # epoch=0 for pretrain
            
            # 构造损失字典（与step_train.py保持一致）
            loss_dict = {
                'focal': focal_loss,
                'patch_bce': loss_bce_patch,
                'global_bce': loss_bce_global, 
                'mask': mask_loss,
                'feature': feature_loss
            }
            
            # 计算加权总损失（与step_train.py保持一致）
            total_loss = (
                adaptive_weights['focal'] * loss_dict['focal'] +
                adaptive_weights['patch_bce'] * loss_dict['patch_bce'] +
                adaptive_weights['global_bce'] * loss_dict['global_bce'] +
                adaptive_weights['mask'] * loss_dict['mask'] +
                adaptive_weights['feature'] * loss_dict['feature']
            )

            generator.zero_grad(set_to_none=True)
            total_loss.backward()

            for name, param in generator.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher_local[name] += (param.grad.detach() ** 2)

            sample_count_local += current
            if rank == 0:
                pbar.set_postfix({
                    "Samples": sample_count_local, 
                    "Loss": f"{float(total_loss.detach().cpu().item()):.4f}",
                    "PixAcc": f"{current_pixel_accuracy:.3f}"
                })

        # 将Fisher与样本数跨进程求和
        for name in fisher_local:
            dist.all_reduce(fisher_local[name], op=dist.ReduceOp.SUM)
        sample_count_tensor = torch.tensor([sample_count_local], device=device, dtype=torch.long)
        dist.all_reduce(sample_count_tensor, op=dist.ReduceOp.SUM)
        total_samples = int(sample_count_tensor.item())

        # 归一化
        for name in fisher_local:
            fisher_local[name] = fisher_local[name] / max(1, total_samples)

        # 仅rank0保存到临时文件（转CPU）
        if rank == 0:
            fisher_cpu = {k: v.detach().cpu() for k, v in fisher_local.items()}
            os.makedirs(os.path.dirname(fisher_tmp_path), exist_ok=True)
            torch.save(fisher_cpu, fisher_tmp_path)
    finally:
        _cleanup_dist()

def save_pretrain_ewc_state(generator, fisher_info, save_path, task_name="pretrain"):
    """
    保存预训练EWC状态
    
    Args:
        generator: 生成器模型
        fisher_info: Fisher信息矩阵
        save_path: 保存路径
        task_name: 任务名称
    """
    # 创建EWC实例
    device = next(generator.parameters()).device
    ewc_instance = create_ewc(generator, device, ewc_type="standard", lambda_ewc=1000.0)
    
    # 保存重要参数
    important_params = {}
    for name, param in generator.named_parameters():
        if param.requires_grad and name in fisher_info:
            important_params[name] = {
                'mean': param.data.clone(),
                'precision': fisher_info[name].clone().to(device)
            }
    
    # 存储到EWC实例
    ewc_instance.important_params[task_name] = important_params
    ewc_instance.task_count = 1
    
    # 保存EWC状态
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    ewc_instance.save_ewc_state(save_path)
    
    print(f"预训练EWC状态已保存到: {save_path}")
    
    # 打印参数重要性统计
    ewc_instance.print_parameter_importance(top_k=15)
    
    return ewc_instance

def analyze_fisher_information(fisher_info, top_k=20):
    """
    分析Fisher信息矩阵
    
    Args:
        fisher_info: Fisher信息矩阵
        top_k: 显示前k个最重要的参数
    """
    print("\n=== Fisher信息分析 ===")
    
    # 计算每个参数的重要性
    importance_dict = {}
    for name, fisher in fisher_info.items():
        importance = torch.mean(fisher).item()
        importance_dict[name] = importance
    
    # 按重要性排序
    sorted_params = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    
    print(f"最重要的 {top_k} 个参数:")
    for i, (name, importance) in enumerate(sorted_params[:top_k]):
        print(f"{i+1:2d}. {name}: {importance:.6f}")
    
    # 统计信息
    all_importances = list(importance_dict.values())
    print(f"\n统计信息:")
    print(f"平均重要性: {np.mean(all_importances):.6f}")
    print(f"重要性标准差: {np.std(all_importances):.6f}")
    print(f"最大重要性: {np.max(all_importances):.6f}")
    print(f"最小重要性: {np.min(all_importances):.6f}")
    
    return importance_dict

def main():
    """主函数"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.info(f"使用设备: {device}")
    
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    patch_size = 512
    batch_size = 2
    num_classes = 6

    factors_data_path = "/root/autodl-fs/factors"
    labels_data_path = "/root/autodl-fs/label_floats"
    num_patches = 1000
    features, labels, valid_mask, coordinates, invalid_coordinates, num_classes = get_data(factors_data_path, labels_data_path, num_patches)
    features = features.to('cpu')
    labels = labels.to('cpu')
    valid_mask = valid_mask.to('cpu')
    coordinates = coordinates.to('cpu')
    invalid_coordinates = invalid_coordinates.to('cpu')

    coordinates = torch.cat([coordinates, invalid_coordinates], dim=0)
    
    # 数据归一化
    norm_params = torch.load("checkpoints/features_normalize_params.pth", weights_only=True)
    features = normalize_tensor_with_params(features, norm_params)
    
    # 加载预训练模型
    logger.info("加载预训练模型...")
    feature_channels = int(features.shape[1] - labels.shape[1])
    generator = Generator(
        hidden_channels=32,
        num_classes=num_classes,
        feature_decoder=True,
        features_channels=feature_channels
    ).to(device)
    
    discriminator = DualDiscriminator(
        hidden_channels=64,
        block_num=512,
        use_soft_input=True,  # 与step_train.py保持一致
        temperature=1.0       # 与step_train.py保持一致
    ).to(device)

    # 触发conv1初始化
    dummy_input = torch.randn(1, 1, features.shape[1] + valid_mask.shape[1], patch_size, patch_size).to(device)
    with torch.no_grad():
        _ = generator(dummy_input)

    dummy_input_1 = torch.randn(1,4, num_classes * 2 + feature_channels, patch_size, patch_size).to(device)
    dummy_input_2 = torch.randn(1,num_classes, patch_size, patch_size).to(device)
    with torch.no_grad():
        _ = discriminator(dummy_input_1, dummy_input_2, mode='both')

    # 加载预训练权重
    # 自动选择最新的stgan_*目录下最大iter的检查点
    checkpoint_path = None
    latest_iter = -1
    if os.path.exists("checkpoints"):
        for d in os.listdir("checkpoints"):
            dir_path = os.path.join("checkpoints", d)
            if os.path.isdir(dir_path) and d.startswith("stgan_"):
                for fname in os.listdir(dir_path):
                    if fname.startswith("iter") and fname.endswith(".pth"):
                        try:
                            it = int(fname[4:-4])
                        except Exception:
                            continue
                        if it > latest_iter:
                            latest_iter = it
                            checkpoint_path = os.path.join(dir_path, fname)
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        
        # 处理生成器状态字典
        generator_state_dict = OrderedDict()
        for k, v in checkpoint['generator_state_dict'].items():
            if k.startswith('module.'):
                k = k[7:]
            if k.startswith('terraflow.'):
                k = 'gru.' + k
            generator_state_dict[k] = v
        
        generator.load_state_dict(generator_state_dict, strict=True)
        logger.info("生成器加载成功！")
        
        # 处理判别器状态字典
        discriminator_state_dict = OrderedDict()
        for k, v in checkpoint['dual_discriminator_state_dict'].items():
            if k.startswith('module.'):
                k = k[7:]
            discriminator_state_dict[k] = v
        
        discriminator.load_state_dict(discriminator_state_dict, strict=True)
        logger.info("判别器加载成功！")
    else:
        raise FileNotFoundError("未找到可用的预训练检查点，请先训练生成器或提供自定义路径")
    
    # 创建数据集
    dataset = PatchDataset(
        features, 
        labels, 
        valid_mask, 
        coordinates, 
        patch_size, 
        corner_sampling=False, 
        enhancement=True
    )
    
    world_size = torch.cuda.device_count()
    if world_size > 1:
        print(f"启动分布式Fisher计算，使用 {world_size} 张GPU...")
        tmp_path = "checkpoints/tmp_fisher_pretrain_rank0.pth"
        # 每个rank尝试处理近似均分的样本量
        per_rank_samples = int((int(coordinates.shape[0]) + world_size - 1) // world_size)
        mp.spawn(
            _distributed_fisher_worker,
            args=(
                world_size,
                dataset,
                batch_size,
                feature_channels,
                num_classes,
                checkpoint_path,
                per_rank_samples,
                tmp_path,
            ),
            nprocs=world_size,
            join=True,
        )
        assert os.path.exists(tmp_path), "分布式Fisher计算未生成结果文件"
        fisher_info = torch.load(tmp_path, map_location='cpu')
        # 清理临时文件
        try:
            os.remove(tmp_path)
        except Exception:
            pass
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        logger.info(f"数据集大小: {len(dataset)}, 批次数量: {len(dataloader)}")
        logger.info("开始计算Fisher信息矩阵...")
        fisher_info = compute_pretrain_fisher_information(
            generator,
            discriminator,
            dataloader,
            device,
            num_samples=int(coordinates.shape[0])
        )
    
    # 分析Fisher信息
    importance_dict = analyze_fisher_information(fisher_info, top_k=20)
    
    # 保存预训练EWC状态
    ewc_save_path = "checkpoints/ewc_pretrain_state.pkl"
    ewc_instance = save_pretrain_ewc_state(
        generator, 
        fisher_info, 
        ewc_save_path, 
        task_name="pretrain"
    )
    
    # 保存Fisher信息矩阵（可选，用于进一步分析）
    fisher_save_path = "checkpoints/fisher_info_pretrain.pth"
    torch.save(fisher_info, fisher_save_path)
    logger.info(f"Fisher信息矩阵已保存到: {fisher_save_path}")
    
    # 验证EWC状态
    logger.info("\n验证EWC状态...")
    test_ewc = create_ewc(generator, device, ewc_type="standard", lambda_ewc=1000.0)
    test_ewc.load_ewc_state(ewc_save_path)
    
    # 计算EWC损失
    ewc_loss = test_ewc.compute_ewc_loss()
    logger.info(f"EWC损失值: {ewc_loss.item():.6f}")
    
    logger.info("\n预训练EWC状态计算完成！")
    logger.info("现在可以在constraint.py中使用这个EWC状态进行增量学习。")

if __name__ == "__main__":
    main()
