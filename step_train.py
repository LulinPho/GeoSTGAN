import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import logging
import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from models.gan import Generator, DualDiscriminator
from losses import mask_penalty, generator_loss, discriminator_loss
from data.dataset import PatchDataset
from utils import get_data, visualize_temporal_comparison, normalize_tensor_and_save_params, normalize_tensor_with_params
from adaptive_training_scheduler import (
    AdaptiveTrainingScheduler, PerformanceTracker, 
    calculate_pixel_accuracy
)
import time
import logging
from collections import OrderedDict
from models.losses_weighted import SoftWeightedLoss
import numpy as np


def augment_with_invalid_by_valid_mask(prob_maps: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    """使用确定性的有效性掩膜(valid_mask)将 [B,C,H,W] 扩展为含无效类 [B,C+1,H,W]。
    - 若 valid_mask=1（有效）：保留六类概率，并将第七类置0
    - 若 valid_mask=0（无效）：六类置0，第七类置1
    valid_mask 形状可为 [B,1,H,W] 或 [B,H,W]
    """
    with torch.no_grad():
        if valid_mask.dim() == 3:
            valid_mask = valid_mask.unsqueeze(1)
        valid_mask = valid_mask.to(dtype=prob_maps.dtype)
        B, C, H, W = prob_maps.shape
        out = torch.zeros(B, C + 1, H, W, device=prob_maps.device, dtype=prob_maps.dtype)
        out[:, :C] = prob_maps * valid_mask
        out[:, C:C+1] = 1.0 - valid_mask
    return out

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def adjust_learning_rate(optimizer, epoch, initial_lr, warmup_epochs, total_epochs):
    """
    学习率调度器，采用warmup+cosine衰减
    """
    if epoch < warmup_epochs:
        lr = initial_lr * (epoch + 1) / warmup_epochs
    else:
        lr = initial_lr * (1 - (epoch - warmup_epochs) / (total_epochs - warmup_epochs)) ** 0.9
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def generator_train(
    rank,
    world_size,
    dataset,
    num_classes,
    feature_channels,
    model_save_path: str = "./checkpoints/x2y",
    num_epochs: int = 10,
    batch_size: int = 2,
    learning_rate: float = 0.0002,
    log_path: str = None,
    retrain_path: str = None
):
    """
    多卡训练GAN模型
    
    Args:
        features: 特征
        labels: 标签
        coordinates: 坐标
        model_save_path: 模型保存路径
        num_epochs: 训练轮数
        batch_size: 批次大小
        learning_rate: 学习率
        loss_version: 损失函数版本，"v1"或"v2"
        retrain_path: 重新训练的模型路径
    """
    if log_path is not None:
        logger = logging.getLogger(f'[Rank{rank}]')
        logger.setLevel(logging.INFO)
        logger.addHandler(logging.StreamHandler())
        fh = logging.FileHandler(log_path)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    # 设置当前进程的设备
    setup(rank, world_size)

    device = f"cuda:{rank}"
    
    logger.info(f"[Rank {rank}] 使用设备: {device}")
    # 构建保存路径
    vis_dir = os.path.join("visual", f"train_stgan")
    os.makedirs(vis_dir, exist_ok=True)

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        sampler=sampler, 
        num_workers=6,  
        persistent_workers=True, 
    )

    logger.info(f"[Rank {rank}] 数据集大小: {len(dataset)}")
    logger.info(f"[Rank {rank}] 批次数量: {len(dataloader)}")

    generator = Generator(
        hidden_channels=32,
        num_classes=6,
        feature_decoder=True,
        features_channels=feature_channels
    ).to(device)
    
    dual_discriminator = DualDiscriminator(
        hidden_channels=64,
        block_num=512,
        use_soft_input=True,
        temperature=1.0
    ).to(device)

    # 辅助有效像元判别网络（不参与优化，仅用于输出后处理）
    # 不再使用辅助有效性网络

    # 包装为DDP模型
    generator = DDP(generator, device_ids=[rank])
    dual_discriminator = DDP(dual_discriminator, device_ids=[rank])
    # 无辅助网络DDP
    # 优化器 - 现在只需要两个优化器
    generator_optimizer = torch.optim.AdamW(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    discriminator_optimizer = torch.optim.AdamW(dual_discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    
    # 加载检查点
    if retrain_path is not None:
        checkpoint = torch.load(retrain_path, weights_only=True)

        # 创建一个虚拟输入来触发generator的初始化
        dummy_input = torch.randn(1,1, num_classes * 2 + feature_channels, 512, 512).to(device)
        with torch.no_grad():
            _ = generator(dummy_input)

        # 创建一个虚拟输入来触发discriminator的初始化
        dummy_input_1 = torch.randn(1,4, num_classes * 2 + feature_channels, 512, 512).to(device)
        dummy_input_2 = torch.randn(1,num_classes, 512, 512).to(device)
        with torch.no_grad():
            _ = dual_discriminator(dummy_input_1, dummy_input_2, mode='both')
        
        # 处理生成器状态字典
        generator_state_dict = OrderedDict()
        for k, v in checkpoint['generator_state_dict'].items():
            generator_state_dict[k] = v
        generator.load_state_dict(generator_state_dict, strict=True)
        

        # 处理双重判别器状态字典
        # 如果是从旧版本的独立判别器检查点加载，需要特殊处理
        if 'dual_discriminator_state_dict' in checkpoint:
            # 新版本的检查点，直接加载
            dual_discriminator_state_dict = OrderedDict()
            for k, v in checkpoint['dual_discriminator_state_dict'].items():
                dual_discriminator_state_dict[k] = v
            dual_discriminator.load_state_dict(dual_discriminator_state_dict, strict=True)
            
            discriminator_optimizer.load_state_dict(checkpoint['discriminator_optimizer_state_dict'])
        else:
            # 旧版本的检查点，需要合并patch和global判别器状态
            logger.info(f"[Rank {rank}] 检测到旧版本检查点，将跳过判别器加载，重新训练判别器")
            logger.info(f"[Rank {rank}] 如需完整恢复，请使用兼容的检查点")
        
        generator_optimizer.load_state_dict(checkpoint['generator_optimizer_state_dict'])
        iter_num = checkpoint['iter_num']
        patch_true_losses = checkpoint['patch_true_losses']
        patch_fake_losses = checkpoint['patch_fake_losses']
        patch_disc_losses = checkpoint['patch_disc_losses']
        global_true_losses = checkpoint['global_true_losses']
        global_fake_losses = checkpoint['global_fake_losses']
        global_disc_losses = checkpoint['global_disc_losses']
        gen_losses = checkpoint['gen_losses']
        focal_losses = checkpoint['focal_losses']
        bce_patch_losses = checkpoint['bce_patch_losses']
        bce_global_losses = checkpoint['bce_global_losses']
        mask_losses = checkpoint['mask_losses']
        feature_losses = checkpoint['feature_losses']
        current_epoch = checkpoint['epoch']+1

        logger.info(f"[Rank {rank}] load model from {retrain_path}")
    else:
        iter_num = 0
        patch_true_losses = []
        patch_fake_losses = []
        patch_disc_losses = []
        global_true_losses = []
        global_fake_losses = []
        global_disc_losses = []
        gen_losses = []
        focal_losses = []
        bce_patch_losses = []
        bce_global_losses = []
        mask_losses = []
        feature_losses = []
        current_epoch = 0

    
    try:
        features_normalize_params = torch.load("checkpoints/features_normalize_params.pth", weights_only=True)
    except:
        features_normalize_params = None

    # 初始化自适应训练调度器
    scheduler = AdaptiveTrainingScheduler()
    
    # 性能跟踪器
    performance_tracker = PerformanceTracker(window_size=10)

    logger.info(f"[Rank {rank}] 开始自适应训练...")

    for epoch in range(current_epoch, num_epochs):

        sampler.set_epoch(epoch)

        dual_discriminator.train()
        generator.train()
        progress_bar = tqdm(dataloader, desc=f"[Rank {rank}] Epoch {epoch+1}/{num_epochs}", disable=(rank != 0))
        
        # 统一的余弦退火学习率调度 - 生成器和判别器同步
        adjust_learning_rate(generator_optimizer, epoch, learning_rate, 2, num_epochs)
        adjust_learning_rate(discriminator_optimizer, epoch, learning_rate, 2, num_epochs)

        if rank == 0:
            logger.info(f"[Rank {rank}] Epoch: {epoch+1}, Learning Rate: {generator_optimizer.param_groups[0]['lr']}")

        for batch_idx, batch in enumerate(progress_bar):

            # 取数据
            x = batch[0].to(device)
            x = x[:,:-1]  # 历史T-1帧作为条件
            x_l = batch[0][:,-1,6:,:,:].to(device)  # 历史最后一帧的特征通道（6:）
            y = batch[1].to(device)
            y_l = y[:,-1,:,:,:]  # 目标最后一帧
            mask = batch[2].to(device)
            mask_l = mask[:,-1,:,:,:]

            valid_sum = mask_l.sum(dim=1, keepdim=True)  # [B, 1, H, W]
            valid_set_mask = (valid_sum > 0.5).float().expand_as(mask_l)

            import math
            def create_random_region_mask_and_replace(valid_mask, y_l, region_ratio=0.25, min_size=32):
                """
                随机使用矩形、圆形或椭圆形区域进行mask选择，并用y_l的对应区域替换mask_l
                """
                B, C, H, W = valid_mask.shape
                replaced_mask = torch.zeros_like(valid_mask)
                for b in range(B):
                    if valid_mask[b, 0].sum() == 0:  # 跳过全无效的样本
                        continue

                    # 在有效区域内随机选择一个中心点
                    valid_coords = torch.nonzero(valid_mask[b, 0])
                    if len(valid_coords) == 0:
                        continue

                    center_idx = torch.randint(0, len(valid_coords), (1,))
                    center_y, center_x = valid_coords[center_idx][0]

                    # 计算区域大小（基于总有效区域的比例）
                    total_valid = valid_mask[b, 0].sum()
                    target_area = int(total_valid * region_ratio)
                    region_size = max(min_size, int(np.sqrt(target_area)))

                    # 随机选择形状：0-矩形，1-圆形，2-椭圆
                    shape_type = torch.randint(0, 3, (1,)).item()

                    if shape_type == 0:
                        # 矩形
                        y1 = max(0, center_y - region_size // 2)
                        y2 = min(H, center_y + region_size // 2)
                        x1 = max(0, center_x - region_size // 2)
                        x2 = min(W, center_x + region_size // 2)
                        # 用y_l的对应区域替换
                        replaced_mask[b:b+1, :, y1:y2, x1:x2] = y_l[b:b+1, :, y1:y2, x1:x2]

                    elif shape_type == 1:
                        # 圆形
                        radius = max(min_size // 2, int(region_size // 2))
                        Y, X = torch.meshgrid(torch.arange(H, device=valid_mask.device), torch.arange(W, device=valid_mask.device), indexing='ij')
                        dist = (Y - center_y) ** 2 + (X - center_x) ** 2
                        circle_mask = (dist <= radius ** 2).unsqueeze(0).repeat(C, 1, 1)
                        # 只替换圆形区域
                        replaced_mask[b, :, :, :] = torch.where(circle_mask > 0, y_l[b, :, :, :], replaced_mask[b, :, :, :])

                    else:
                        # 椭圆形（随机离心率）
                        a = max(min_size // 2, int(region_size // 2))
                        b_ellipse = max(min_size // 2, int(region_size // 2 * torch.rand(1).item() * 0.8 + 0.6))  # 0.6~1.4倍
                        angle = torch.rand(1).item() * 2 * math.pi  # 随机旋转角度
                        Y, X = torch.meshgrid(torch.arange(H, device=valid_mask.device), torch.arange(W, device=valid_mask.device), indexing='ij')
                        y_shift = Y - center_y
                        x_shift = X - center_x
                        # 旋转坐标轴
                        x_rot = x_shift * math.cos(angle) + y_shift * math.sin(angle)
                        y_rot = -x_shift * math.sin(angle) + y_shift * math.cos(angle)
                        ellipse_mask = ((x_rot / a) ** 2 + (y_rot / b_ellipse) ** 2 <= 1).unsqueeze(0).repeat(C, 1, 1)
                        # 只替换椭圆区域
                        replaced_mask[b, :, :, :] = torch.where(ellipse_mask > 0, y_l[b, :, :, :], replaced_mask[b, :, :, :])

                return replaced_mask

            # 用y_l的最后一帧信息替换mask_l的随机区域
            mask_l = create_random_region_mask_and_replace(valid_set_mask, y_l)

            x_gen = torch.cat([x, mask], dim=2)
            
            res = dual_discriminator(x_gen, y_l, mode='both')
            patch_true_validity, _ = res['patch']
            global_true_validity, _ = res['global']

            # 判别器步骤使用detach的假样本
            with torch.no_grad():
                fake_samples, fake_features = generator(x_gen)
            fake_samples = fake_samples.detach()
            fake_features = fake_features.detach()

            fake_samples_l = fake_samples[:, -1, :, :, :]
            res = dual_discriminator(x_gen, fake_samples_l, mode='both')
            patch_fake_validity, _ = res['patch']
            global_fake_validity, _ = res['global']
            patch_disc_loss, patch_true_loss, patch_fake_loss = discriminator_loss(patch_true_validity, patch_fake_validity)
            global_disc_loss, global_true_loss, global_fake_loss = discriminator_loss(global_true_validity, global_fake_validity)

            # 组合判别器损失
            total_disc_loss = (patch_disc_loss + global_disc_loss) /2

            discriminator_optimizer.zero_grad()
            total_disc_loss.backward()
            discriminator_optimizer.step()
    
            dy_1 = patch_true_validity.mean()
            dy_2 = global_true_validity.mean()
            dg_11 = patch_fake_validity.mean()
            dg_12 = global_fake_validity.mean()

            # 生成器步骤
            fake_samples, fake_features= generator(x_gen)
            fake_samples_l = fake_samples[:,-1,:,:,:]
            fake_features_l = fake_features[:,-1,:,:,:]

            fake_features_l = normalize_tensor_with_params(fake_features_l, features_normalize_params)
            res = dual_discriminator(x_gen, fake_samples_l, mode='both')
            patch_fake_validity,_ = res['patch']
            global_fake_validity,_ = res['global']

            dg_21 = patch_fake_validity.mean()
            dg_22 = global_fake_validity.mean()

            # 仅在有效像元上计算分类损失
            valid_pixel_mask = (y_l.sum(dim=1, keepdim=True) > 0.5).float()  # [B,1,H,W]
            focal_loss, loss_bce_patch, loss_bce_global = generator_loss(
                fake_samples_l, y_l, patch_fake_validity, global_fake_validity, valid_pixel_mask=valid_pixel_mask
            )

            # 计算特征损失
            # 特征MSE只在有效像元上计算
            pred_features = fake_features_l
            target_x = x_l
            mse_per_pixel = (pred_features - target_x) ** 2  # [B,C,H,W]
            # 聚合到像元级：对通道求均值
            mse_pixel = mse_per_pixel.mean(dim=1, keepdim=True)  # [B,1,H,W]
            denom = valid_pixel_mask.sum().clamp_min(1.0)
            mse_loss = (mse_pixel * valid_pixel_mask).sum() / denom

            mask_loss = mask_penalty(fake_samples_l, valid_set_mask, alpha=0.5)

            # 计算性能指标
            current_pixel_accuracy = calculate_pixel_accuracy(fake_samples_l, y_l, valid_set_mask)
            adaptive_weights = scheduler.get_loss_weights(epoch, current_pixel_accuracy)
            
            # 构造损失字典
            loss_dict = {
                'focal': focal_loss,
                'patch_bce': loss_bce_patch,
                'global_bce': loss_bce_global, 
                'mask': mask_loss,
                'feature': mse_loss
            }
            
            # 计算加权总损失
            gen_loss = (
                adaptive_weights['focal'] * loss_dict['focal'] +
                adaptive_weights['patch_bce'] * loss_dict['patch_bce'] +
                adaptive_weights['global_bce'] * loss_dict['global_bce'] +
                adaptive_weights['mask'] * loss_dict['mask'] +
                adaptive_weights['feature'] * loss_dict['feature']
            )

            generator_optimizer.zero_grad()
            gen_loss.backward()
            generator_optimizer.step()

            # 不再训练辅助网络
            
            iter_num += 1

            # 更新性能跟踪器
            performance_tracker.update(
                pixel_accuracy=current_pixel_accuracy,
                focal_loss=focal_loss.item(),
                discriminator_loss=(patch_disc_loss.item() + global_disc_loss.item()) / 2
            )
            
            patch_true_losses.append(patch_true_loss.item())
            patch_fake_losses.append(patch_fake_loss.item())    
            patch_disc_losses.append(patch_disc_loss.item())
            global_true_losses.append(global_true_loss.item())
            global_fake_losses.append(global_fake_loss.item())    
            global_disc_losses.append(global_disc_loss.item())
            gen_losses.append(gen_loss.item())
            focal_losses.append(focal_loss.item())
            bce_patch_losses.append(loss_bce_patch.item())
            bce_global_losses.append(loss_bce_global.item())
            mask_losses.append(mask_loss.item())
            feature_losses.append(mse_loss.item())

            # 构造无效类输出
            with torch.no_grad():
                B, T, C, H, W = fake_samples.shape
                fake_aug = torch.zeros(B, T, C+1, H, W, device=fake_samples.device, dtype=fake_samples.dtype)
                for t in range(T):
                    valid_mask_t = (y[:, t].sum(dim=1, keepdim=True) > 0.5).float()  # [B,1,H,W]
                    fake_aug[:, t] = augment_with_invalid_by_valid_mask(fake_samples[:, t], valid_mask_t)
                fake_aug_l = fake_aug[0]

            with torch.no_grad():
                B, T, C, H, W = y.shape
                y_aug = torch.zeros(B, T, C+1, H, W, device=y.device, dtype=y.dtype)
                y_aug[:, :, :C] = y
                y_valid_sum = y.sum(dim=2, keepdim=True)
                y_aug[:, :, C:C+1] = (y_valid_sum <= 0.5).float()
                real_tensor = y_aug[0]

            fake_tensor = fake_aug_l.argmax(dim=1)  # [T, H, W]
            real_tensor = real_tensor.argmax(dim=1)  # [T, H, W]


            # 处理进度条，显示像素精度驱动的训练状态
            current_lr = generator_optimizer.param_groups[0]['lr']
            progress_bar.set_postfix({
                "PixAcc": f"{current_pixel_accuracy:.3f}",
                "FocalW": f"{adaptive_weights['focal']:.2f}",
                "DiscW": f"{adaptive_weights['patch_bce']:.2f}",
                "G_loss": f"{gen_losses[-1]:.4f}",
                "D_loss": f"{patch_disc_losses[-1]:.3f}",
                "LR": f"{current_lr:.5f}",
                "Iter": iter_num
            })
            
            # 更新进度条
            if rank == 0 and batch_idx % 25 == 0:

                # 使用日志记录损失
                logger.info(
                    f"[Rank {rank}] Epoch: {epoch+1}, Batch: {batch_idx+1}, "
                    f"D_loss: {patch_disc_losses[-1]:.4f}/{global_disc_losses[-1]:.4f}, Tr_loss: {patch_true_losses[-1]:.4f}/{global_true_losses[-1]:.4f}, "
                    f"Fa_loss: {patch_fake_losses[-1]:.4f}/{global_fake_losses[-1]:.4f}, G_loss: {gen_losses[-1]:.4f}, "
                    f"Fl_loss: {focal_losses[-1]:.4f}, Bc_loss: {bce_patch_losses[-1]:.4f}/{bce_global_losses[-1]:.4f}, "
                    f"D(x,y): {dy_1:.4f}/{dy_2:.4f}, D(x,G(x)): P:{dg_11:.4f}/{dg_21:.4f}, G:{dg_12:.4f}/{dg_22:.4f}, M_loss: {mask_losses[-1]:.4f}, Feature_loss: {feature_losses[-1]:.4f}, Iter: {iter_num}"
                )
                # 保存对比图
                vis_path = os.path.join(vis_dir, f"iter{iter_num}.png")
                visualize_temporal_comparison(
                    tensor1=real_tensor,
                    tensor2=fake_tensor,
                    save_path=vis_path,
                    title1="True",
                    title2="Fake"
                )
            
            # 只在主进程保存模型
            if rank == 0 and iter_num % 500 == 0:
                save_path = os.path.join(model_save_path, f"iter{iter_num}.pth")
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                logger.info(f"[Rank {rank}] 保存模型到: {save_path}")
                torch.save({
                    'epoch': epoch,
                    'generator_state_dict': generator.state_dict(),
                    'dual_discriminator_state_dict': dual_discriminator.state_dict(),
                    'generator_optimizer_state_dict': generator_optimizer.state_dict(),
                    'discriminator_optimizer_state_dict': discriminator_optimizer.state_dict(),
                    'patch_true_losses': patch_true_losses,
                    'patch_fake_losses': patch_fake_losses,
                    'patch_disc_losses': patch_disc_losses,
                    'global_true_losses': global_true_losses,
                    'global_fake_losses': global_fake_losses,
                    'global_disc_losses': global_disc_losses,
                    'gen_losses': gen_losses,
                    'focal_losses': focal_losses,
                    'bce_patch_losses': bce_patch_losses,
                    'bce_global_losses': bce_global_losses,
                    'mask_losses': mask_losses,
                    'feature_losses': feature_losses,
                    'iter_num': iter_num,
                }, save_path)
                logger.info(f"[Rank {rank}] 模型已保存到: {save_path}") 

        # 只在主进程保存模型
        if rank == 0:
            save_path = os.path.join(model_save_path, f"iter{iter_num}.pth")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            logger.info(f"[Rank {rank}] 保存模型到: {save_path}")
            torch.save({
                'epoch': epoch,
                'generator_state_dict': generator.state_dict(),
                'dual_discriminator_state_dict': dual_discriminator.state_dict(),
                'generator_optimizer_state_dict': generator_optimizer.state_dict(),
                'discriminator_optimizer_state_dict': discriminator_optimizer.state_dict(),
                'patch_true_losses': patch_true_losses,
                'patch_fake_losses': patch_fake_losses,
                'patch_disc_losses': patch_disc_losses,
                'global_true_losses': global_true_losses,
                'global_fake_losses': global_fake_losses,
                'global_disc_losses': global_disc_losses,
                'gen_losses': gen_losses,
                'focal_losses': focal_losses,
                'bce_patch_losses': bce_patch_losses,
                'bce_global_losses': bce_global_losses,
                'mask_losses': mask_losses,
                'feature_losses': feature_losses,
                'iter_num': iter_num,
            }, save_path)
            logger.info(f"[Rank {rank}] 模型已保存到: {save_path}")
        
    cleanup() 
    if rank == 0:
        logger.info("多卡训练完成!")


def train():
    # 设置日志
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())
    log_path = f"logs/stgan_{time.strftime('%Y%m%d_%H%M%S')}.log"
    logger.addHandler(logging.FileHandler(log_path))

    # 训练参数
    world_size = torch.cuda.device_count()
    factors_data_path = "/root/autodl-fs/factors"
    labels_data_path = "/root/autodl-fs/label_floats"
    model_save_path = "checkpoints/stgan_20250813"
    patch_size = 512
    num_patches = 4000  # 减少patch数量以加速训练
    num_epochs = 5
    batch_size = 4  # 进一步增加batch size以提高GPU利用率
    learning_rate = 0.0002  # 进一步降低学习率以提高稳定性
    retrain_path = None # "checkpoints/stgan/iter720.pth" # 继续训练的检查点 - 使用新架构时请设为None

    features, labels, valid_mask, coordinates, invalid_coordinates, _ = get_data(factors_data_path, labels_data_path, num_patches)
    features = features.to('cpu')
    labels = labels.to('cpu')
    valid_mask = valid_mask.to('cpu')
    coordinates = coordinates.to('cpu')
    invalid_coordinates = invalid_coordinates.to('cpu')
    if len(coordinates) == 0:
        raise ValueError("没有找到有效的训练位置")

    # # 历史掩膜tif路径（2020）
    # history_mask_tif = "/root/autodl-fs/mask/2020/2020.tif"
    # # 长线掩膜tif路径（2035）
    # longterm_mask_tif = "/root/autodl-fs/mask/2035/2035.tif"

    # valid_mask = load_mask(history_mask_tif, longterm_mask_tif, 4, 0)

    coordinates = torch.cat([coordinates, invalid_coordinates], dim=0)

    # 对特征进行归一化
    features, features_normalize_params = normalize_tensor_and_save_params(features)

    # 将归一化参数保存到checkpoints文件夹中
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(features_normalize_params, os.path.join("checkpoints", "features_normalize_params.pth"))
    logger.info("特征归一化参数已保存到 checkpoints/features_normalize_params.pth")

    num_classes = labels.shape[1]
    feature_channels = features.shape[1]-labels.shape[1]

    print(features.shape, labels.shape, valid_mask.shape, coordinates.shape, invalid_coordinates.shape)

    dataset = PatchDataset(features, labels, valid_mask, coordinates, patch_size, enhancement=True)  # 关闭数据增强以提高速度

    torch.multiprocessing.spawn(   # type: ignore
        generator_train,  # 函数名
        args=(
            world_size,            # world_size: int
            dataset,
            num_classes,
            feature_channels,
            model_save_path,         # model_save_path: str
            num_epochs,                    # num_epochs: int
            batch_size,                    # batch_size: int
            learning_rate,                # learning_rate: float
            log_path,
            retrain_path
        ),
        nprocs=world_size
    )


if __name__ == "__main__":
    train()