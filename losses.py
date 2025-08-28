"""
GAN损失函数模块

包含所有GAN训练中使用的损失函数：
- 判别器损失函数
- 生成器损失函数  
- 约束损失函数
- 掩膜损失函数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss
import cv2
import numpy as np


def discriminator_loss(true_validity, fake_validity):
    """
    判别器损失函数（适用于全局判别器和patch判别器）

    Args:
        true_validity: 真实样本的判别器输出，形状可为 (b, ...) 或 (b, t, ...)
        fake_validity: 生成样本的判别器输出，形状同上
    Returns:
        disc_loss: 判别器总损失
        true_loss: 真实样本损失
        fake_loss: 生成样本损失
    """
    bce = BCEWithLogitsLoss()
    true_loss = bce(true_validity, torch.full_like(true_validity, 0.95))
    fake_loss = bce(fake_validity, torch.full_like(fake_validity, 0.05))
    disc_loss = (true_loss + fake_loss) * 0.5
    return disc_loss, true_loss, fake_loss


def generator_loss(probs, y, patch_fake_validity, global_fake_validity, gamma=2.0, eps=1e-8, valid_pixel_mask=None):
    """
    生成器损失函数（亚元素级别占比预测），返回focal loss和判别器loss。

    Args:
        probs: 预测概率（已为softmax输出），形状 (b, c, h, w)
        y: 真实标签，形状 (b, c, h, w)，独热编码
        patch_fake_validity: patch判别器输出
        global_fake_validity: global判别器输出
        gamma: focal loss的gamma参数
        eps: 数值稳定项
    Returns:
        loss_focal: focal loss
        loss_bce_patch: patch判别器loss
        loss_bce_global: global判别器loss
    """
    # focal loss（掩膜版，probs已为softmax输出）
    b, c, h, w = probs.shape
    probs_flat = probs.permute(0, 2, 3, 1).reshape(-1, c)  # (N, c)
    y_flat = y.permute(0, 2, 3, 1).reshape(-1, c)          # (N, c)
    y_label = y_flat.argmax(dim=1)                         # (N,)
    pt = probs_flat[torch.arange(probs_flat.size(0)), y_label].clamp(min=eps, max=1.0-eps)  # (N,)
    focal_loss = -((1 - pt) ** gamma) * torch.log(pt)      # (N,)
    if valid_pixel_mask is not None:
        m = valid_pixel_mask.reshape(-1)                   # (N,)
        denom = m.sum().clamp_min(1.0)
        loss_focal = (focal_loss * m).sum() / denom
    else:
        loss_focal = focal_loss.mean()

    # 判别器loss修正
    patch_fake_validity = patch_fake_validity.to(torch.float32)
    global_fake_validity = global_fake_validity.to(torch.float32)
    loss_bce_patch = BCEWithLogitsLoss()(patch_fake_validity, torch.full_like(patch_fake_validity, 1.0))
    loss_bce_global = BCEWithLogitsLoss()(global_fake_validity, torch.full_like(global_fake_validity, 1.0))

    return loss_focal, loss_bce_patch, loss_bce_global

def mask_penalty(probs, mask, alpha=0.5, eps=1e-6):
    """
    掩膜约束损失（避免对每个有效类都推向1的结构性偏差）。

    思路：
    - 只要求"有效类的总概率质量"大（接近1），而非每个有效类各自接近1；
    - 同时抑制"无效类的总概率质量"。

    Args:
        probs: 概率图，形状 (b, c, h, w)
        mask: 有效性掩膜，形状 (b, c, h, w)。1表示该像素可被分为该类，0表示不可
        alpha: 无效项权重（0~1）。loss = (1-alpha)*valid_term + alpha*invalid_term
        eps: 数值稳定项

    Returns:
        标量损失
    """
    # 掩膜集合（按类）
    valid_mask = (mask > 0.5).to(probs.dtype)           # [b, c, h, w]
    invalid_mask = 1.0 - valid_mask                     # [b, c, h, w]

    # 像元有效区域（只要该像元有任一有效类）
    pixel_valid = (valid_mask.sum(dim=1, keepdim=True) > 0).to(probs.dtype)  # [b, 1, h, w]
    pixel_valid_s = pixel_valid.squeeze(1)                                      # [b, h, w]
    denom = pixel_valid.sum().clamp_min(1.0)

    # 每像元的有效/无效总概率
    sum_valid = (probs * valid_mask).sum(dim=1)          # [b, h, w]
    sum_invalid = (probs * invalid_mask).sum(dim=1)      # [b, h, w]

    # 仅在像元有效区域内计算两项
    valid_term = -(torch.log(sum_valid + eps) * pixel_valid_s).sum() / denom
    invalid_term = (sum_invalid * pixel_valid_s).sum() / denom

    loss = (1.0 - alpha) * valid_term + alpha * invalid_term
    return loss


def inverse_quantity_constraint(probs, constraint_matrix, xi=(1.0, 1.0, 1.0), valid_pixel_mask: torch.Tensor | None = None, logger=None):
    """
    逆向数量约束损失函数（整图版本）。

    支持目标类约束、上限约束、下限约束三种类型，约束信息以(3, N)的张量E输入。
    公式参考：
        L_t = i^T * abs(l_c - e_1)
        L_u = i^T * max(l_c - e_2, 0)
        L_l = i^T * max(e_3 - l_c, 0)
        L_num = (ξ_1 L_t + ξ_2 L_u + ξ_3 L_l) / (H*W)
    其中i^T为单位向量转置，e_1为目标类约束，e_2为上限，e_3为下限。

    Args:
        probs: 生成器输出概率，形状为 (C, H, W)
        constraint_matrix: 约束提示矩阵E，形状为 (3, C)，
            E[0]为目标类约束(target)，E[1]为上限(upper)，E[2]为下限(lower)
        xi: 损失权重参数 (ξ_1, ξ_2, ξ_3)，默认全为1.0
        valid_pixel_mask: 可选，有效像元掩膜 (H, W) 或 (1, H, W)。若提供，仅在有效像元上统计与归一化

    Returns:
        loss: 数量约束损失（标量）
        pred_counts: 当前整图每类的预测数量（概率和）
    """
    device = probs.device
    C = probs.shape[0]
    H = probs.shape[1]
    W = probs.shape[2]
    # 计算每个类别的概率和（有效区域）
    if valid_pixel_mask is not None:
        if valid_pixel_mask.dim() == 3:
            valid_pixel_mask = valid_pixel_mask.squeeze(0)
        valid_pixel_mask = valid_pixel_mask.to(device=device, dtype=probs.dtype)  # [H, W]
        pred_counts = (probs * valid_pixel_mask.unsqueeze(0)).sum(dim=(1, 2))   # [C]
        denom = valid_pixel_mask.sum()
    else:
        pred_counts = probs.sum(dim=(1, 2))   # [C]
        denom = torch.tensor(float(H * W), device=device, dtype=probs.dtype)

    # 约束向量
    e_1 = constraint_matrix[0].to(device)  # 目标类约束
    e_2 = constraint_matrix[1].to(device)  # 上限
    e_3 = constraint_matrix[2].to(device)  # 下限

    # 仅对非零约束项计算损失，零表示无约束
    mask_t = (e_1 != 0).float()
    mask_u = (e_2 != 0).float()
    mask_l = (e_3 != 0).float()

    # 目标约束损失
    L_t = torch.sum(mask_t * torch.abs(pred_counts - e_1))
    # 上限约束损失
    L_u = torch.sum(mask_u * torch.clamp(pred_counts - e_2, min=0))
    # 下限约束损失
    L_l = torch.sum(mask_l * torch.clamp(e_3 - pred_counts, min=0))

    xi_1, xi_2, xi_3 = xi

    # 总损失
    L_num = (xi_1 * L_t + xi_2 * L_u + xi_3 * L_l) / (denom + 1e-8)

    if logger is not None:
        # 汇报累计像元数量和约束目标
        logger.info("类别\t下限\t目标值\t上限")
        for i in range(C):
            lower = e_3[i].item()
            target = e_1[i].item()
            upper = e_2[i].item()
            acc = pred_counts[i].item()
            logger.info(f"{i}:\t{lower:.1f} ≤ ({acc:.1f}) ≈ ({target:.1f}) ≤ {upper:.1f}")

    return L_num, pred_counts


def efficient_constraint(probs, current_img, efficient_matrix, valid_pixel_mask: torch.Tensor | None = None, logger=None):
    """
    可微效应约束损失：根据 current_img（概率）与 probs（概率），
    以softmax概率形成“软转换矩阵”，并仅在有效像元上约束（可选）。

    Args:
        probs: 生成器输出概率，形状 (C, H, W)
        current_img: 真实标签概率，形状 (C, H, W)
        efficient_matrix: 效应矩阵 (C, C)，E[i, j] 为 i->j 的代价
        valid_pixel_mask: 可选，有效像元掩膜，形状 (H, W) 或 (1, H, W)。
            若提供，仅在有效像元（=1）上统计转换；若为None，则在全图统计。

    Returns:
        loss: 效应约束损失（标量）
        transition_counts: 软转换矩阵 (C, C)
    """
    device = probs.device
    C, H, W = probs.shape

    # 展平至像素维度
    N = H * W
    probs_flat = probs.view(C, N).clone().to(device)      # [C, N]
    y_flat = current_img.view(C, N).clone().to(device)    # [C, N]

    # 有效像元权重，仅对像元维施加（避免平方权重）
    if valid_pixel_mask is not None:
        if valid_pixel_mask.dim() == 3 and valid_pixel_mask.shape[0] == 1:
            valid_pixel_mask = valid_pixel_mask.squeeze(0)
        # [H, W] -> [N]
        w_flat = valid_pixel_mask.to(dtype=probs.dtype, device=device).view(N)
        y_flat = y_flat * w_flat.unsqueeze(0)  # [C, N] * [1, N]
    
    # 软转换矩阵：对每个像素累积 y ⊗ p
    # transition_counts[c_true, c_pred] = sum_n y[c_true, n] * p[c_pred, n]
    transition_counts = torch.einsum('cn,dn->cd', y_flat, probs_flat)  # [C, C]

    # 代价与归一化
    E = efficient_matrix.to(device=device, dtype=probs.dtype)
    effect_loss = (transition_counts * E).sum()

    # 归一化像素数：若有mask，用有效像元数，否则用总像素数
    if valid_pixel_mask is not None:
        valid_count = valid_pixel_mask.sum().to(dtype=probs.dtype)
        denom = torch.clamp(valid_count, min=torch.tensor(1.0, device=device, dtype=probs.dtype))
    else:
        denom = torch.tensor(float(H * W), device=device, dtype=probs.dtype)
    loss = effect_loss / (denom + 1e-8)

    return loss, transition_counts


def water_morphological_constraint(probs: torch.Tensor, 
                                 previous_probs: torch.Tensor, 
                                 water_class_idx: int = 4,
                                 dilation_kernel_size: int = 3,
                                 penalty_strength: float = 10.0,
                                 valid_pixel_mask: torch.Tensor = None,
                                 logger=None):
    """
    水域形态学约束损失函数：确保水域类型只能从已有用地类型膨胀产生，不能突然出现。
    
    约束原理：
    - 水域（类型4）不能在空白区域突然出现
    - 水域只能在前一时刻已有用地类型的邻域内扩展
    - 通过形态学膨胀操作检查约束满足情况
    
    Args:
        probs: 当前时刻的概率预测，形状 (C, H, W)
        previous_probs: 前一时刻的概率分布，形状 (C, H, W)  
        water_class_idx: 水域类型的索引，默认为4
        dilation_kernel_size: 膨胀操作的核大小，控制允许的扩展距离
        penalty_strength: 违反约束时的惩罚强度
        valid_pixel_mask: 可选，有效像元掩膜 (H, W)
        
    Returns:
        loss: 形态学约束损失（标量）
        violation_map: 违反约束的区域图 (H, W)
    """
    device = probs.device
    C, H, W = probs.shape
    
    # 获取当前时刻水域的概率分布
    current_water_prob = probs[water_class_idx]  # [H, W]
    
    # 获取前一时刻所有非水域类型的概率分布
    # 假设类型0-5分别对应不同的用地类型，水域是类型4
    non_water_indices = [i for i in range(C) if i != water_class_idx]
    previous_non_water = previous_probs[non_water_indices].sum(dim=0)  # [H, W]
    
    # 创建前一时刻已有用地的二值掩膜（概率阈值化）
    existing_land_mask = (previous_non_water > 0.1).float()  # [H, W]
    
    # 使用形态学膨胀扩展允许水域出现的区域
    # 转换为numpy进行opencv操作
    existing_land_np = existing_land_mask.detach().cpu().numpy().astype(np.uint8)
    
    # 创建膨胀核
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                     (dilation_kernel_size, dilation_kernel_size))
    
    # 膨胀操作：扩展允许水域出现的区域
    dilated_mask_np = cv2.dilate(existing_land_np, kernel, iterations=1)
    
    # 转换回tensor
    dilated_mask = torch.from_numpy(dilated_mask_np).float().to(device)  # [H, W]
    
    # 计算违反约束的区域：水域出现在膨胀区域之外
    # 只有当水域概率较高时才认为是违反约束
    water_threshold = 0.3  # 水域概率阈值
    water_presence = (current_water_prob > water_threshold).float()  # [H, W]
    
    # 违反约束区域：水域出现但不在允许区域内
    violation_map = water_presence * (1.0 - dilated_mask)  # [H, W]
    
    # 计算损失：违反约束的程度 × 水域概率
    violation_intensity = violation_map * current_water_prob  # [H, W]
    
    # 应用有效像元掩膜（如果提供）
    if valid_pixel_mask is not None:
        if valid_pixel_mask.dim() == 3:
            valid_pixel_mask = valid_pixel_mask.squeeze(0)
        valid_pixel_mask = valid_pixel_mask.to(device=device, dtype=probs.dtype)
        violation_intensity = violation_intensity * valid_pixel_mask
        normalizer = valid_pixel_mask.sum() + 1e-8
    else:
        normalizer = float(H * W)
    
    # 计算平均违反损失
    loss = penalty_strength * violation_intensity.sum() / normalizer
    
    # 记录违反情况
    violation_count = violation_map.sum().item()
    total_water_pixels = water_presence.sum().item()
    
    if violation_count > 0 and logger is not None:
        logger.warning(f"水域形态学约束违反: {violation_count:.1f} 个像素违反约束, "
                      f"占水域像素的 {violation_count/max(total_water_pixels, 1)*100:.2f}%")
    
    return loss, violation_map


def adaptive_water_morphological_constraint(probs: torch.Tensor,
                                          previous_probs: torch.Tensor,
                                          water_class_idx: int = 4,
                                          base_dilation_size: int = 3,
                                          adaptive_factor: float = 1.5,
                                          penalty_strength: float = 10.0,
                                          valid_pixel_mask: torch.Tensor = None,
                                          logger=None):
    """
    自适应水域形态学约束：根据前一时刻水域密度动态调整膨胀范围。
    
    在水域密集区域允许更大的扩展范围，在稀疏区域更严格限制。
    
    Args:
        probs: 当前时刻的概率预测，形状 (C, H, W)
        previous_probs: 前一时刻的概率分布，形状 (C, H, W)
        water_class_idx: 水域类型的索引
        base_dilation_size: 基础膨胀核大小
        adaptive_factor: 自适应因子，用于根据局部水域密度调整膨胀范围
        penalty_strength: 惩罚强度
        valid_pixel_mask: 可选，有效像元掩膜
        
    Returns:
        loss: 自适应形态学约束损失
        violation_map: 违反约束的区域图
    """
    device = probs.device
    C, H, W = probs.shape
    
    # 获取当前和前一时刻的水域概率
    current_water_prob = probs[water_class_idx]  # [H, W]
    previous_water_prob = previous_probs[water_class_idx]  # [H, W]
    
    # 计算前一时刻所有已有用地（包括水域）的分布
    previous_total_land = previous_probs.sum(dim=0)  # [H, W]
    existing_land_mask = (previous_total_land > 0.1).float()
    
    # 计算局部水域密度以确定自适应膨胀范围
    kernel_size = 5
    water_density = F.avg_pool2d(
        previous_water_prob.unsqueeze(0).unsqueeze(0),
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size//2
    ).squeeze()  # [H, W]
    
    # 根据水域密度计算自适应膨胀核大小
    # 密度高的区域允许更大扩展，密度低的区域更严格
    adaptive_dilation_size = base_dilation_size + (water_density * adaptive_factor).int()
    adaptive_dilation_size = torch.clamp(adaptive_dilation_size, min=1, max=base_dilation_size * 3)
    
    # 转换为numpy进行空间变化的膨胀操作
    existing_land_np = existing_land_mask.detach().cpu().numpy().astype(np.uint8)
    adaptive_size_np = adaptive_dilation_size.detach().cpu().numpy()
    
    # 创建自适应膨胀掩膜
    dilated_mask = torch.zeros_like(existing_land_mask)
    
    # 对不同区域应用不同大小的膨胀核
    unique_sizes = np.unique(adaptive_size_np)
    for size in unique_sizes:
        if size > 0:
            region_mask = (adaptive_size_np == size).astype(np.uint8)
            if region_mask.sum() > 0:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(size), int(size)))
                local_existing = existing_land_np * region_mask
                local_dilated = cv2.dilate(local_existing, kernel, iterations=1)
                dilated_mask += torch.from_numpy(local_dilated * region_mask).float().to(device)
    
    # 确保膨胀掩膜为二值
    dilated_mask = torch.clamp(dilated_mask, 0, 1)
    
    # 计算违反约束的区域
    water_threshold = 0.3
    water_presence = (current_water_prob > water_threshold).float()
    violation_map = water_presence * (1.0 - dilated_mask)
    
    # 计算损失
    violation_intensity = violation_map * current_water_prob
    
    if valid_pixel_mask is not None:
        if valid_pixel_mask.dim() == 3:
            valid_pixel_mask = valid_pixel_mask.squeeze(0)
        valid_pixel_mask = valid_pixel_mask.to(device=device, dtype=probs.dtype)
        violation_intensity = violation_intensity * valid_pixel_mask
        normalizer = valid_pixel_mask.sum() + 1e-8
    else:
        normalizer = float(H * W)
    
    loss = penalty_strength * violation_intensity.sum() / normalizer
    
    return loss, violation_map