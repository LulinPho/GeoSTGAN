"""
分布式约束训练优化模块
将VJP重建过程改为分布式并行计算，显著提升训练效率
"""

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import torch.multiprocessing as mp
import logging
import os
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
import time
from concurrent.futures import ThreadPoolExecutor
import threading
from contextlib import nullcontext

from models.gan import Generator
from losses import mask_penalty, inverse_quantity_constraint, efficient_constraint, water_morphological_constraint
from sharpening_losses import entropy_sharpening_loss, adaptive_temperature_annealing
from data.dataset import PatchDataset
from utils import get_data, create_positions_sequence, normalize_tensor_with_params
from ewc import create_ewc

logger = logging.getLogger(__name__)

def infer_step(
    generator: torch.nn.Module,
    xseq: torch.Tensor,
    mask: torch.Tensor,
    coordinates: torch.Tensor,
    num_classes: int,
    feature_channels: int,
    device: torch.device,
    img_size: tuple,
    patch_size: int = 512,
    batch_size: int = 24,
    rank: int = 0,
    world_size: int = 1
):
    """
    分布式推理步骤，将patches分配给不同GPU并行处理
    """
    generator.eval()
    
    # 创建数据集
    labels = torch.zeros(4, num_classes, img_size[0], img_size[1])
    dataset = PatchDataset(xseq.cpu(), labels, mask.cpu(), coordinates.cpu(), 
                          patch_size, corner_sampling=True, enhancement=False)
    
    # 初始化结果张量
    next_label_full = torch.zeros((num_classes, img_size[0], img_size[1]), device=device)
    next_feature_full = torch.zeros((feature_channels, img_size[0], img_size[1]), device=device)
    next_label_count = torch.zeros((num_classes, img_size[0], img_size[1]), device=device)
    next_feature_count = torch.zeros((feature_channels, img_size[0], img_size[1]), device=device)
    
    # 分布式数据采样器
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    
    # 构造高斯窗
    with torch.no_grad():
        gaussian_window = torch.ones((patch_size, patch_size), dtype=torch.float32).to(device)
        center = torch.tensor(patch_size // 2).to(torch.float32)
        sigma = torch.tensor(patch_size / 4.0).to(torch.float32)
        for i in range(patch_size):
            for j in range(patch_size):
                distribution = ((i - center) ** 2 + (j - center) ** 2) ** 0.5
                value = 0.5 + 0.5 * torch.exp(-distribution ** 2 / (2 * sigma ** 2))
                gaussian_window[i, j] = value
    
    # 并行推理
    norm_params = torch.load("checkpoints/features_normalize_params.pth", weights_only=True)
    
    import sys
    from tqdm import tqdm

    with torch.no_grad():
        if rank == 0:
            dataloader_iter = tqdm(enumerate(dataloader), total=len(dataloader), desc="Inference", file=sys.stdout)
        else:
            dataloader_iter = enumerate(dataloader)
        for batch_idx, (x, _, mask_patch, coordinates_patch) in dataloader_iter:
            x = x.to(device)
            mask_patch = mask_patch.to(device)
            coordinates_patch = coordinates_patch.to(device)
            
            x_gen = torch.cat([x, mask_patch], dim=2)
            output, next_feature = generator(x_gen)
            
            next_feature = normalize_tensor_with_params(next_feature, norm_params)
            
            B, T, C, patch_h, patch_w = output.shape
            output = output[:, -1]  # [B, C, patch_h, patch_w]
            next_feature = next_feature[:, -1]  # [B, F, patch_h, patch_w]
            
            # 并行处理batch中的每个patch
            for i in range(B):
                y_pos, x_pos = coordinates_patch[i]
                y_pos = int(y_pos.item())
                x_pos = int(x_pos.item())
                patch_label = output[i]
                patch_feat = next_feature[i]
                
                next_label_full[:, y_pos:y_pos+patch_h, x_pos:x_pos+patch_w] += patch_label * gaussian_window
                next_label_count[:, y_pos:y_pos+patch_h, x_pos:x_pos+patch_w] += gaussian_window
                next_feature_full[:, y_pos:y_pos+patch_h, x_pos:x_pos+patch_w] += patch_feat * gaussian_window
                next_feature_count[:, y_pos:y_pos+patch_h, x_pos:x_pos+patch_w] += gaussian_window
    
    # 汇聚所有GPU的结果
    if world_size > 1:
        dist.all_reduce(next_label_full, op=dist.ReduceOp.SUM)
        dist.all_reduce(next_label_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(next_feature_full, op=dist.ReduceOp.SUM)
        dist.all_reduce(next_feature_count, op=dist.ReduceOp.SUM)
    
    # 归一化
    next_label_count = torch.where(next_label_count == 0, torch.ones_like(next_label_count), next_label_count)
    next_label_full = next_label_full / next_label_count
    
    next_feature_count = torch.where(next_feature_count == 0, torch.ones_like(next_feature_count), next_feature_count)
    next_feature_full = next_feature_full / next_feature_count
    
    # 应用掩膜
    mask_last = mask[-1]
    zero_mask = (mask_last.sum(dim=0) < 0.5)
    next_feature_full[:, zero_mask] = 0
    
    return next_label_full, next_feature_full

class DistributedVJPProcessor:
    """分布式VJP处理器，实现高效的并行VJP计算"""
    
    def __init__(self, rank: int, world_size: int, device: torch.device):
        self.rank = rank
        self.world_size = world_size
        self.device = device
    
    def process_vjp_batch(self, 
                         generator,
                         dataloader,
                         g_logits: torch.Tensor,
                         gaussian_window: torch.Tensor,
                         inv_label_count: torch.Tensor,
                         norm_params: dict
                         ):
        """
        并行处理VJP计算批次
        """
        # 不累积VJP损失，避免保持计算图导致显存溢出
        processed_patches = 0
        
        # 为当前rank创建数据采样器
        sampler = DistributedSampler(
            dataloader.dataset, 
            num_replicas=self.world_size, 
            rank=self.rank, 
            shuffle=False
        )
        
        distributed_dataloader = DataLoader(
            dataloader.dataset,
            batch_size=dataloader.batch_size,
            sampler=sampler
        )
        
        import sys
        from tqdm import tqdm

        if self.rank == 0:
            dataloader_iter = tqdm(enumerate(distributed_dataloader), total=len(distributed_dataloader), desc="VJP Processing", file=sys.stdout)
        else:
            dataloader_iter = enumerate(distributed_dataloader)

        sync_ctx = generator.no_sync() if isinstance(generator, DDP) and self.world_size > 1 else nullcontext()
        with sync_ctx:
            for patch_idx, (x, _, mask_patch, coordinates_patch) in dataloader_iter:
                try:
                    x = x.to(self.device)
                    mask_patch = mask_patch.to(self.device)
                    coordinates_patch = coordinates_patch.to(self.device)
                    
                    x_gen = torch.cat([x, mask_patch], dim=2)
                    
                    with torch.enable_grad():
                        output, next_feature = generator(x_gen)
                        next_feature = normalize_tensor_with_params(next_feature, norm_params)
                        
                        output = output[:, -1]  # [B, C, patch_h, patch_w]
                        
                        B = output.shape[0]
                        for i in range(B):
                            # 计算VJP
                            y_pos, x_pos = coordinates_patch[i]
                            y_pos, x_pos = int(y_pos.item()), int(x_pos.item())
                            
                            patch_label = output[i]  # [C, patch_h, patch_w]
                            patch_h, patch_w = patch_label.shape[1], patch_label.shape[2]
                            
                            # 提取对应区域的梯度和权重
                            g_label_region = g_logits[:, y_pos:y_pos+patch_h, x_pos:x_pos+patch_w]
                            inv_label_region = inv_label_count[:, y_pos:y_pos+patch_h, x_pos:x_pos+patch_w]
                            
                            # VJP计算 - 添加数值稳定性检查
                            eps = 1e-8
                            inv_label_region_stable = torch.clamp(inv_label_region, min=eps)
                            vjp_contribution = (g_label_region * patch_label * gaussian_window / inv_label_region_stable).sum()
                            
                            # 立即进行反向传播，避免累积计算图
                            vjp_contribution.backward(retain_graph=True)
                            processed_patches += 1
                    
                    # 清理临时变量，释放计算图
                    del x, mask_patch, coordinates_patch, x_gen, output, next_feature
                    del g_label_region, inv_label_region, vjp_contribution
                    
                    # 每处理几个patches就清理一次显存
                    if patch_idx % 5 == 0:
                        torch.cuda.empty_cache()
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        logger.warning(f"Patch {patch_idx} on rank {self.rank} 显存不足，跳过")
                        torch.cuda.empty_cache()
                        continue
                    else:
                        raise e
        
        return processed_patches

def distributed_constraint_train(
    generator,
    xseq: torch.Tensor,
    mask: torch.Tensor,
    coordinates: torch.Tensor,
    num_classes: int,
    feature_channels: int,
    device: torch.device,
    constraint_matrix: torch.Tensor,
    transfer_constraint: torch.Tensor,
    img_size: tuple,
    ewc_instance=None,
    config=None,
    current_iteration: int = 0,
    learning_rate: float = 1e-5,
    rank: int = 0,
    world_size: int = 1,
):
    """
    分布式VJP约束训练
    """
    generator.train()

    def freeze_unused_branches(model):
        """冻结特征解码器参数"""
        logger.info("开始冻结特征解码器参数...")
        
        if hasattr(model, 'gru') and hasattr(model.gru, 'feature_decoder'):
            for param in model.gru.feature_decoder.parameters():
                param.requires_grad = False
        return model

    generator = freeze_unused_branches(generator)
    
    # 第一阶段：分布式推理
    with torch.no_grad():
        full_probs, full_features = infer_step(
            generator=generator,
            xseq=xseq,
            mask=mask,
            coordinates=coordinates,
            num_classes=num_classes,
            feature_channels=feature_channels,
            device=device,
            img_size=img_size,
            patch_size=config.patch_size,
            batch_size=config.batch_size,
            rank=rank,
            world_size=world_size
        )
        torch.cuda.empty_cache()

    valid_mask = (mask[-1].sum(dim=0) > 0.5).to(device)

    # 第二阶段：计算约束损失
    full_probs_detached = full_probs.detach().requires_grad_(True)
    old_probs = xseq[-1,:6]
    
    # 计算各种约束损失
    if constraint_matrix is not None and constraint_matrix.sum() > 0:
        constraint_loss, _ = inverse_quantity_constraint(
            probs=full_probs_detached,
            constraint_matrix=constraint_matrix,
            valid_pixel_mask=valid_mask,
            xi=config.constraint_xi
        )
        constraint_loss = constraint_loss * config.constraint_weight
    else:
        constraint_loss = torch.tensor(0.0, device=device)

    if transfer_constraint is not None and transfer_constraint.sum() > 0:
        transfer_loss, _ = efficient_constraint(
            probs=full_probs_detached,
            current_img=old_probs,
            efficient_matrix=transfer_constraint,
            valid_pixel_mask=valid_mask,
        )

        transfer_loss = transfer_loss * config.transfer_weight
    else:
        transfer_loss = torch.tensor(0.0, device=device)
    
    mask_last = mask[-1].to(device)
    mask_loss = mask_penalty(
        full_probs_detached.unsqueeze(0), 
        mask_last.unsqueeze(0),
        alpha=config.mask_alpha
    )

    mask_loss = mask_loss * config.mask_weight

    water_morphological_loss, violation_map = water_morphological_constraint(
        probs=full_probs_detached,
        previous_probs=old_probs,
        water_class_idx=config.water_class_idx,
        dilation_kernel_size=config.water_dilation_kernel_size,
        penalty_strength=config.water_penalty_strength,
        valid_pixel_mask=valid_mask
    )

    water_morphological_loss = water_morphological_loss * config.water_constraint_weight

    # 第2.5阶段：计算熵锐化损失（鼓励单一类型最大化）
    sharpening_loss = torch.tensor(0.0, device=device)
    avg_entropy = torch.tensor(0.0, device=device)
    
    if config.enable_sharpening and config.sharpening_weight > 0:
        # 设置自适应温度退火
        current_temp = config.entropy_temperature
        if config.adaptive_temperature:
            max_iterations = 100  # 使用默认值或从config获取
            current_temp = adaptive_temperature_annealing(
                current_iteration, max_iterations,
                initial_temp=config.entropy_temperature * 2.0,
                final_temp=config.entropy_temperature * 0.5
            )
        
        # 计算熵锐化损失
        sharpening_loss, avg_entropy = entropy_sharpening_loss(
            probs=full_probs_detached,
            temperature=current_temp,
            target_entropy_ratio=config.target_entropy_ratio,
            valid_pixel_mask=valid_mask
        )

        sharpening_loss = sharpening_loss * config.sharpening_weight
        

    total_constraint_loss = (
        constraint_loss + 
        mask_loss +
        transfer_loss +
        water_morphological_loss +
        sharpening_loss
    )
    
    # 反向传播获得梯度
    total_constraint_loss.backward()
    g_logits = full_probs_detached.grad.detach()
    
    torch.cuda.empty_cache()
    
    # 第三阶段：分布式VJP重建
    optimizer = torch.optim.AdamW(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    optimizer.zero_grad()
    
    # 构造数据集
    labels = torch.zeros(4, num_classes, img_size[0], img_size[1])
    dataset = PatchDataset(xseq.cpu(), labels, mask.cpu(), coordinates.cpu(), 
                          config.patch_size, corner_sampling=True, enhancement=False)
    dataloader = DataLoader(dataset, batch_size=config.vjp_batch_size, shuffle=False)

    # 构造高斯窗
    with torch.no_grad():
        gaussian_window = torch.ones((config.patch_size, config.patch_size), dtype=torch.float32).to(device)
        center = torch.tensor(config.patch_size // 2).to(torch.float32)
        sigma = torch.tensor(config.patch_size / 4.0).to(torch.float32)
        for i in range(config.patch_size):
            for j in range(config.patch_size):
                distance = ((i - center) ** 2 + (j - center) ** 2) ** 0.5
                value = 0.5 + 0.5 * torch.exp(-distance ** 2 / (2 * sigma ** 2))
                gaussian_window[i, j] = value
    
    # 预计算归一化权重
    with torch.no_grad():
        inv_label_count = torch.zeros((num_classes, img_size[0], img_size[1]), dtype=torch.float32).to(device)
        for batch_idx, (_, _, _, coordinates_patch) in enumerate(dataloader):
            # coordinates_patch 形状: [B, 2]
            B_coords = coordinates_patch.shape[0]
            for i in range(B_coords):
                y_pos, x_pos = coordinates_patch[i]
                y_pos, x_pos = int(y_pos.item()), int(x_pos.item())
                inv_label_count[:, y_pos:y_pos+config.patch_size, x_pos:x_pos+config.patch_size] += gaussian_window
    
    # 分布式VJP处理
    vjp_processor = DistributedVJPProcessor(rank, world_size, device)
    
    norm_params = torch.load("checkpoints/features_normalize_params.pth", weights_only=True)
    
    # 并行计算VJP
    _ = vjp_processor.process_vjp_batch(
        generator=generator,
        dataloader=dataloader,
        g_logits=g_logits,
        gaussian_window=gaussian_window,
        inv_label_count=inv_label_count,
        norm_params=norm_params
    )

    # 第四阶段：EWC损失计算
    ewc_loss = torch.tensor(0.0, device=device)
    if ewc_instance is not None:
        ewc_loss = ewc_instance.compute_ewc_loss() * config.ewc_weight
        ewc_loss.backward()

    total_constraint_loss = total_constraint_loss + ewc_loss
    
    # 第五阶段：参数更新
    optimizer.step()
    optimizer.zero_grad()
    torch.cuda.empty_cache()
    
    loss_dict = {
        'total_loss': total_constraint_loss.detach().cpu().item(),
        'constraint_loss': constraint_loss.detach().cpu().item(),
        'transfer_loss': transfer_loss.detach().cpu().item(),
        'mask_loss': mask_loss.detach().cpu().item(),
        'water_constraint_loss': water_morphological_loss.detach().cpu().item(),
        'ewc_loss': ewc_loss.detach().cpu().item(),
        'sharpening_loss': sharpening_loss.detach().cpu().item(),
        'avg_entropy': avg_entropy.detach().cpu().item(),
    }

    torch.cuda.empty_cache()
    
    return full_probs.detach(), full_features.detach(), loss_dict
