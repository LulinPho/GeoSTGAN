"""
约束推理模块
基于constraint_train.py的逻辑，去除训练部分，专门用于推理预测
每一步都使用infer-step，保存可视化和预测结果数据
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
import logging
import os
import numpy as np
from tqdm import tqdm
import time
from collections import OrderedDict
import pandas as pd

from models.gan import Generator
from distributed_constraint import infer_step
from data.dataset import PatchDataset
from utils import get_data, create_positions_sequence, normalize_tensor_with_params
from visual import visualize_sequence_tensor, visualize_tensor

logger = logging.getLogger(__name__)

class ConstraintInferConfig:
    """约束推理配置类 - 支持分布式和非分布式推理"""
    
    def __init__(self):
        # ============ 基础推理参数 ============
        self.patch_size = 512
        self.overlap = 256
        self.batch_size = 24
        
        # ============ 分布式参数 ============
        self.distributed_backend = 'nccl'
        self.master_addr = '127.0.0.1'
        self.master_port = '29599'
        
        # ============ 性能优化参数 ============
        self.num_workers = 2
        self.pin_memory = True
        self.prefetch_factor = 2
        self.memory_efficient_mode = True
        
        # ============ 结果保存参数 ============
        self.save_interval = 1           # 结果保存间隔
        self.visualize_interval = 1     # 可视化保存间隔
        
        # ============ 推理输出标识 ============
        self.result_components = [
            'pred_images',
            'land_type_counts', 
            'entropy_values',
            'prediction_confidence'
        ]
        
    def setup_distributed_env(self):
        """设置分布式环境变量"""
        os.environ.setdefault('MASTER_ADDR', self.master_addr)
        os.environ.setdefault('MASTER_PORT', self.master_port)
    
    def determine_inference_mode(self, world_size: int = None):
        """根据world_size确定推理模式"""
        if world_size is None:
            world_size = torch.cuda.device_count()
        return world_size > 1
    
    def log_config_summary(self, logger, rank=0):
        """记录配置摘要信息"""
        if rank == 0:
            logger.info("=" * 60)
            logger.info("约束推理配置摘要")
            logger.info("=" * 60)
            logger.info(f"基础参数: patch_size={self.patch_size}, batch_size={self.batch_size}")
            logger.info(f"推理参数: overlap={self.overlap}, memory_efficient={self.memory_efficient_mode}")
            logger.info(f"保存参数: save_interval={self.save_interval}, visualize_interval={self.visualize_interval}")
            logger.info("=" * 60)
    
    def get_output_dir(self, experiment_name=None):
        """获取输出目录路径"""
        if experiment_name is None:
            experiment_name = f"constraint_infer_{time.strftime('%Y%m%d_%H%M%S')}"
        return os.path.join("results", "constraint_infer", experiment_name)

# ============ 核心工具函数 ============

def _setup_dist(rank: int, world_size: int):
    """设置分布式环境"""
    os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
    os.environ.setdefault('MASTER_PORT', '29599')
    torch.cuda.set_device(rank)
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)

def _cleanup_dist():
    """清理分布式环境"""
    if dist.is_initialized():
        dist.destroy_process_group()

def augment_with_invalid_by_valid_mask(prob_maps: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    """使用确定性的有效性掩膜将 [C,H,W] 扩展为含无效类 [C+1,H,W]"""
    with torch.no_grad():
        valid_mask = valid_mask.to(device=prob_maps.device, dtype=prob_maps.dtype)
        C, H, W = prob_maps.shape
        out = torch.zeros(C + 1, H, W, device=prob_maps.device, dtype=prob_maps.dtype)
        out[:C] = prob_maps * valid_mask
        out[C:C+1] = 1.0 - valid_mask
    return out

def calculate_prediction_metrics(prob_maps: torch.Tensor, valid_mask: torch.Tensor):
    """计算预测相关的指标"""
    with torch.no_grad():
        # 计算平均熵
        epsilon = 1e-8
        entropy = -torch.sum(prob_maps * torch.log(prob_maps + epsilon), dim=0)
        avg_entropy = entropy[valid_mask > 0.5].mean().item()
        
        # 计算预测置信度（最大概率）
        max_probs = torch.max(prob_maps, dim=0)[0]
        avg_confidence = max_probs[valid_mask > 0.5].mean().item()
        
        # 统计各类别像元数量
        land_type_map = torch.argmax(prob_maps, dim=0)
        class_counts = []
        for cls in range(prob_maps.shape[0]):
            count = ((land_type_map == cls) & (valid_mask > 0.5)).sum().item()
            class_counts.append(count)
            
    return {
        'avg_entropy': avg_entropy,
        'avg_confidence': avg_confidence,
        'class_counts': class_counts
    }

def preload_inference_data(config):
    """预加载推理数据到CPU内存"""
    logger.info("开始预加载推理数据...")
    
    factors_data_path = "/root/autodl-fs/factors"
    labels_data_path = "/root/autodl-fs/label_floats"
    num_patches = 1
    
    features, labels, valid_mask, _, _, num_classes = get_data(
        factors_data_path, labels_data_path, num_patches
    )
    features = features.to('cpu')
    valid_mask = valid_mask.to('cpu')
    
    # 数据归一化
    norm_params = torch.load("checkpoints/features_normalize_params.pth", weights_only=True)
    features = normalize_tensor_with_params(features, norm_params)
    
    # 创建坐标序列
    coordinates, _ = create_positions_sequence(features, config.patch_size, config.overlap, 1)
    
    preloaded_data = {
        'features': features,
        'labels': labels,
        'valid_mask': valid_mask,
        'num_classes': num_classes,
        'norm_params': norm_params,
        'coordinates': coordinates,
    }
    
    logger.info(f"数据预加载完成 - 特征形状: {features.shape}, 类别数: {num_classes}")
    logger.info(f"内存使用: {features.numel() * features.element_size() / 1024**3:.2f} GB")
    
    return preloaded_data

# ============ 工作进程函数 ============
def distributed_constraint_infer_worker(rank, world_size, step, iter_num, final_mask, 
                                       config, preloaded_data, checkpoint_path, 
                                       experiment_name):
    """分布式约束推理工作进程"""
    if config is None:
        config = ConstraintInferConfig()
    
    # 设置分布式环境
    _setup_dist(rank, world_size)
    
    try:
        device = torch.device(f"cuda:{rank}")
        logger_name = f"distributed-infer-rank{rank}"
        worker_logger = logging.getLogger(logger_name)
        worker_logger.setLevel(logging.INFO)
        
        if rank == 0:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(
                f'[Rank{rank}] %(levelname)s: %(message)s'
            ))
            worker_logger.addHandler(handler)
            worker_logger.info(f"启动分布式约束推理 - Rank {rank}/{world_size}")
        
        # 使用预加载的数据
        features = preloaded_data['features']
        valid_mask = preloaded_data['valid_mask']
        num_classes = preloaded_data['num_classes']
        coordinates = preloaded_data['coordinates']

        features = features[1:]
        valid_mask = torch.cat([valid_mask[1:], final_mask.unsqueeze(0)], dim=0)

        pixel_mask = torch.sum(final_mask, dim=0) > 0.5

        # 统计pixel_mask中True和False的数量
        num_true = pixel_mask.sum().item()
        num_false = pixel_mask.numel() - num_true
        worker_logger.info(f"pixel_mask True count: {num_true}, False count: {num_false}")
        
        # 创建生成器
        generator = Generator(
            hidden_channels=32,
            num_classes=num_classes,
            feature_decoder=True,
            features_channels=features.shape[1] - valid_mask.shape[1]
        ).to(device)

        # 触发conv1初始化
        dummy_input = torch.randn(1, 1, features.shape[1] + valid_mask.shape[1], 
                                config.patch_size, config.patch_size).to(device)
        with torch.no_grad():
            _ = generator(dummy_input)

        # 加载检查点
        if checkpoint_path and os.path.exists(checkpoint_path):
            state_dict = torch.load(checkpoint_path, weights_only=True)
            
            # 加载模型参数
            if 'generator_state_dict' in state_dict:
                generator_state_dict = OrderedDict()
                for k, v in state_dict['generator_state_dict'].items():
                    if k.startswith('module.'):
                        new_key = k[7:]
                    else:
                        new_key = k
                    generator_state_dict[new_key] = v
                try:
                    generator.load_state_dict(generator_state_dict, strict=True)
                    worker_logger.info(f"生成器加载成功！加载自文件: {checkpoint_path}")
                except Exception as e:
                    worker_logger.warning(f"生成器加载失败: {e}")
            elif 'model_state_dict' in state_dict:
                generator_state_dict = OrderedDict()
                for k, v in state_dict['model_state_dict'].items():
                    if k.startswith('module.'):
                        new_key = k[7:]
                    else:
                        new_key = k
                    generator_state_dict[new_key] = v
                try:
                    generator.load_state_dict(generator_state_dict, strict=True)
                    worker_logger.info(f"生成器加载成功！加载自文件: {checkpoint_path}")
                except Exception as e:
                    worker_logger.warning(f"生成器加载失败: {e}")
        else:
            # 使用默认检查点
            default_path = os.path.join("checkpoints", "stgan", f"iter1500.pth")
            if os.path.exists(default_path):
                stgan_checkpoint = torch.load(default_path, weights_only=True)
                if 'generator_state_dict' in stgan_checkpoint:
                    stgan_generator_state_dict = OrderedDict()
                    for k, v in stgan_checkpoint['generator_state_dict'].items():
                        if k.startswith('module.'):
                            new_key = k[7:]
                        else:
                            new_key = k
                        stgan_generator_state_dict[new_key] = v
                    try:
                        generator.load_state_dict(stgan_generator_state_dict, strict=True)
                        worker_logger.info(f"生成器加载成功！加载自默认文件: {default_path}")
                    except Exception as e:
                        worker_logger.warning(f"生成器加载失败: {e}")

        generator = DDP(generator, device_ids=[rank])

        # 设置生成器为评估模式
        generator.eval()

        # 创建输出目录
        output_dir = config.get_output_dir(experiment_name)
        if rank == 0:
            os.makedirs(output_dir, exist_ok=True)

        # 推理结果存储
        inference_results = {
            'prob_imgs': [],
            'metrics_history': [],
            'step_info': []
        }

        for iter in range(iter_num):
            prob_imgs = []
            step_metrics = []
            
            if rank == 0:
                worker_logger.info(f"推理迭代次数: {iter+1}/{iter_num}")

            for _step in range(step):
                # 每一步都进行推理
                with torch.no_grad():
                    next_label_full, next_feature_full = infer_step(
                        generator=generator, xseq=features, mask=valid_mask, coordinates=coordinates,
                        num_classes=num_classes, feature_channels=features.shape[1] - valid_mask.shape[1],
                        device=device, img_size=(features.shape[2], features.shape[3]),
                        rank=rank, world_size=world_size
                    )
                    next_label_full = next_label_full.cpu()
                    next_feature_full = next_feature_full.cpu()

                    # 计算预测指标
                    metrics = calculate_prediction_metrics(next_label_full, pixel_mask)
                    step_metrics.append(metrics)

                    prob_img = augment_with_invalid_by_valid_mask(next_label_full.clone(), pixel_mask)
                    prob_imgs.append(prob_img)

                    next_x = torch.cat([next_label_full, next_feature_full], dim=0)
                    features = torch.cat([features[1:], next_x.unsqueeze(0)], dim=0)
                    valid_mask = torch.cat([valid_mask[1:], final_mask.unsqueeze(0)], dim=0)

                    if rank == 0:
                        worker_logger.info(f"步骤 {_step+1}/{step} - 平均熵: {metrics['avg_entropy']:.4f}, "
                                         f"平均置信度: {metrics['avg_confidence']:.4f}")
                        # 输出各类别统计
                        class_info = ", ".join([f"类别{i}:{count}" for i, count in enumerate(metrics['class_counts'])])
                        worker_logger.info(f"类别统计: {class_info}")

            # 保存当前迭代的结果
            inference_results['prob_imgs'].append(torch.stack(prob_imgs, dim=0))
            inference_results['metrics_history'].append(step_metrics)
            inference_results['step_info'].append({
                'iteration': iter + 1,
                'total_steps': step,
                'timestamp': time.time()
            })
            
            # 定期保存可视化结果
            if rank == 0 and (iter + 1) % config.visualize_interval == 0:
                worker_logger.info(f"保存第 {iter+1} 迭代的可视化结果...")
                
                # 保存可视化图像
                current_prob_imgs = torch.stack(prob_imgs, dim=0)
                visualize_sequence_tensor(
                    current_prob_imgs, 
                    save_path=os.path.join(output_dir, f"inference_iter_{iter+1}.png"),
                    titles=[f"Step {i+1}" for i in range(step)]
                )
                
                # 保存指标数据
                metrics_df_list = []
                for step_idx, step_metric in enumerate(step_metrics):
                    row_data = {
                        'iteration': iter + 1,
                        'step': step_idx + 1,
                        'avg_entropy': step_metric['avg_entropy'],
                        'avg_confidence': step_metric['avg_confidence']
                    }
                    # 添加各类别计数
                    for cls_idx, count in enumerate(step_metric['class_counts']):
                        row_data[f'class_{cls_idx}_count'] = count
                    metrics_df_list.append(row_data)
                
                metrics_df = pd.DataFrame(metrics_df_list)
                metrics_df.to_csv(os.path.join(output_dir, f"metrics_iter_{iter+1}.csv"), index=False)
                
                worker_logger.info(f"第 {iter+1} 迭代结果已保存至 {output_dir}")

        # 保存最终的完整结果
        if rank == 0:
            worker_logger.info("保存最终推理结果...")
            
            # 保存所有迭代的概率图像
            all_prob_imgs = torch.stack([imgs for imgs in inference_results['prob_imgs']], dim=0)
            all_type_imgs = torch.argmax(all_prob_imgs, dim=2).to(torch.uint8)
            torch.save(all_type_imgs, os.path.join(output_dir, "all_type_imgs.pth"))
            
            # 创建完整的指标统计表
            all_metrics_data = []
            for iter_idx, (step_metrics, step_info) in enumerate(zip(inference_results['metrics_history'], inference_results['step_info'])):
                for step_idx, step_metric in enumerate(step_metrics):
                    row_data = {
                        'iteration': iter_idx + 1,
                        'step': step_idx + 1,
                        'avg_entropy': step_metric['avg_entropy'],
                        'avg_confidence': step_metric['avg_confidence'],
                        'timestamp': step_info['timestamp']
                    }
                    # 添加各类别计数
                    for cls_idx, count in enumerate(step_metric['class_counts']):
                        row_data[f'class_{cls_idx}_count'] = count
                    all_metrics_data.append(row_data)
            
            final_metrics_df = pd.DataFrame(all_metrics_data)
            
            # 保存为Excel文件，包含多个工作表
            excel_path = os.path.join(output_dir, "complete_inference_results.xlsx")
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                final_metrics_df.to_excel(writer, sheet_name="Complete_Metrics", index=False)
                
                # 创建按迭代分组的统计
                iter_summary = final_metrics_df.groupby('iteration').agg({
                    'avg_entropy': ['mean', 'std'],
                    'avg_confidence': ['mean', 'std']
                }).round(4)
                iter_summary.to_excel(writer, sheet_name="Iteration_Summary")
                
                # 创建类别变化趋势
                class_cols = [col for col in final_metrics_df.columns if col.startswith('class_') and col.endswith('_count')]
                class_trends = final_metrics_df[['iteration', 'step'] + class_cols]
                class_trends.to_excel(writer, sheet_name="Class_Trends", index=False)
            
            worker_logger.info(f"完整推理结果已保存至 {excel_path}")
            worker_logger.info("约束推理完成！")

    finally:
        _cleanup_dist()

# ============ 统一推理入口 ============

def unified_constraint_inference(step, iter_num, final_mask, config=None, 
                                world_size=None, checkpoint_path=None, 
                                experiment_name=None):
    """
    统一约束推理入口函数
    根据world_size自动选择分布式或非分布式推理
    """
    
    # 使用默认配置
    if config is None:
        config = ConstraintInferConfig()
    
    # 自动检测GPU数量
    if world_size is None:
        world_size = torch.cuda.device_count()
    
    # 设置环境变量
    config.setup_distributed_env()
    
    # 配置日志
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        logger.addHandler(logging.StreamHandler())
    
    if experiment_name is None:
        experiment_name = f"constraint_infer_{time.strftime('%Y%m%d_%H%M%S')}"
    
    log_path = f"logs/constraint_infer_{experiment_name}.log"
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger.addHandler(logging.FileHandler(log_path))
    
    # 记录配置信息
    config.log_config_summary(logger, rank=0)

    logger.info(f"启动分布式约束推理，使用 {world_size} 个GPU")
    logger.info(f"实验名称: {experiment_name}")
    if checkpoint_path:
        logger.info(f"使用检查点: {checkpoint_path}")
    
    # 预加载推理数据到CPU内存
    logger.info("预加载推理数据...")
    preloaded_data = preload_inference_data(config)
    
    # 启动多进程分布式推理
    mp.spawn(
        distributed_constraint_infer_worker,
        args=(
            world_size, step, iter_num, final_mask, config, preloaded_data,
            checkpoint_path, experiment_name
        ),
        nprocs=world_size,
        join=True,
    )

if __name__ == "__main__":
    # 示例使用
    print("=== 约束推理模块测试 ===")
    
    # 加载数据
    try:
        import rasterio
        with rasterio.open("/root/autodl-fs/mask/2035/2035b.tif") as src:
            final_mask = src.read()
            final_mask = torch.from_numpy(final_mask[:6])
    except:
        final_mask = torch.ones(6, 1024, 1024)
        print("警告：使用模拟终期掩膜")

    print("配置完成，开始约束推理...")

    # INSERT_YOUR_CODE
    # 生成所有tw和ewc组合的checkpoint路径
    tw_values = [0.1, 0.05]
    ewc_values = [10.0, 5.0, 0]
    checkpoint_list = []
    for tw in tw_values:
        for ewc in ewc_values:
            ckpt = f"checkpoints/best_models/cw1.0_mw1.0_tw{tw}_ewc{ewc}_ww1.0_sw0.0.pth"
            checkpoint_list.append(ckpt)
    
    for checkpoint in checkpoint_list:
        import re
        match = re.search(r'tw([0-9.]+)_ewc([0-9.]+)', checkpoint)
        if match:
            tw = float(match.group(1))
            ewc = float(match.group(2))
        else:
            tw = None
            ewc = None
        # 启动统一推理（自动检测GPU数量）
        unified_constraint_inference(
            step=3, 
            iter_num=1,  # 推理迭代次数
            final_mask=final_mask, 
            config=None,
                checkpoint_path=checkpoint,  # 可指定特定检查点
                experiment_name=f"test_constraint_inference_tw{tw}_ewc{ewc}"
            )
