"""
统一约束训练模块
整合所有约束训练功能，支持自动选择分布式/单GPU训练
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

from models.gan import Generator
from distributed_constraint import distributed_constraint_train, infer_step
from data.dataset import PatchDataset
from utils import get_data, create_positions_sequence, normalize_tensor_with_params
from ewc import create_ewc

logger = logging.getLogger(__name__)

class ConstraintConfig:
    """统一约束训练配置类 - 同时支持分布式和非分布式训练"""
    
    def __init__(self):
        # ============ 基础训练参数 ============
        self.patch_size = 512
        self.overlap = 256
        self.batch_size = 24
        self.learning_rate = 0.0001
        self.learning_rate_decay = 0.8
        
        # ============ 分布式参数 ============
        self.distributed_backend = 'nccl'
        self.master_addr = '127.0.0.1'
        self.master_port = '29599'
        
        # ============ 性能优化参数 ============
        self.num_workers = 2
        self.pin_memory = True
        self.prefetch_factor = 2
        self.gradient_accumulation_steps = 1
        self.vjp_batch_size = 2
        self.memory_efficient_mode = True
        self.async_gradient_sync = True
        
        # ============ 损失权重参数 ============
        self.constraint_weight = 1.0        # 数量约束损失权重
        self.mask_weight = 1.0              # 掩膜约束损失权重
        self.transfer_weight = 0.1          # 转移约束损失权重
        self.ewc_weight = 5.0              # EWC损失权重
        self.water_constraint_weight = 1.0  # 水域形态学约束损失权重
        
        # ============ 熵锐化损失参数 ============
        self.enable_sharpening = False
        self.sharpening_weight = 0.0
        self.entropy_temperature = 1.0
        self.target_entropy_ratio = 0.2
        self.adaptive_temperature = True
        self.sharpening_schedule = {
            'start_iteration': 10,
            'ramp_up_iterations': 10,
            'max_weight': 1.0
        }
        
        # ============ 水域约束参数 ============
        self.water_class_idx = 3            # 水域类型索引
        self.water_dilation_kernel_size = 5 # 水域膨胀核大小
        self.water_penalty_strength = 1.0   # 水域约束惩罚强度
        
        # ============ 早停参数 ============
        self.early_stop_patience = 10
        self.loss_composition_delta = 1e-2  # 损失组成比例L1偏移阈值
        self.total_loss_delta = 5e-2        # 总损失L1变化阈值
        
        # ============ 约束参数 ============ 
        self.constraint_xi = (0.0, 1.0, 1.0)  # 数量约束权重参数
        self.mask_alpha = 0.5                 # 掩膜损失alpha参数
        
        # ============ 监控和记录参数 ============
        self.save_checkpoint_interval = 10    # 检查点保存间隔
        self.log_loss_interval = 1           # 损失记录间隔
        self.visualize_interval = 10         # 可视化保存间隔
        
        # ============ 损失组成标识 ============
        self.loss_components = [
            'constraint_loss',
            'mask_loss', 
            'transfer_loss',
            'water_constraint_loss',
            'sharpening_loss',
            'ewc_loss'
        ]
        
        # ============ 损失显示名称映射 ============
        self.loss_display_names = {
            'total_loss': '总计',
            'constraint_loss': '数量约束',
            'mask_loss': '掩膜约束',
            'transfer_loss': '转移约束',
            'water_constraint_loss': '水域约束',
            'sharpening_loss': '熵锐化',
            'ewc_loss': 'EWC',
            'avg_entropy': '平均熵'
        }
        
    def setup_distributed_env(self):
        """设置分布式环境变量"""
        os.environ.setdefault('MASTER_ADDR', self.master_addr)
        os.environ.setdefault('MASTER_PORT', self.master_port)
    
    def get_weight_string(self):
        """生成权重字符串用于文件命名"""
        return (f"cw{self.constraint_weight}_mw{self.mask_weight}_"
                f"tw{self.transfer_weight}_ewc{self.ewc_weight}_"
                f"ww{self.water_constraint_weight}_sw{self.sharpening_weight}")
    
    def determine_training_mode(self, world_size: int = None):
        """根据world_size确定训练模式"""
        if world_size is None:
            world_size = torch.cuda.device_count()
        
        return world_size > 1
    
    def log_config_summary(self, logger, rank=0):
        """记录配置摘要信息"""
        if rank == 0:
            logger.info("=" * 60)
            logger.info("统一约束训练配置摘要")
            logger.info("=" * 60)
            logger.info(f"基础参数: patch_size={self.patch_size}, batch_size={self.batch_size}, lr={self.learning_rate}")
            logger.info(f"损失权重: 数量={self.constraint_weight}, 掩膜={self.mask_weight}, 转移={self.transfer_weight}")
            logger.info(f"         水域={self.water_constraint_weight}, 熵锐化={self.sharpening_weight}, EWC={self.ewc_weight}")
            logger.info(f"水域约束: 类别索引={self.water_class_idx}, 膨胀核={self.water_dilation_kernel_size}, 惩罚强度={self.water_penalty_strength}")
            logger.info(f"熵锐化: 启用={self.enable_sharpening}, 温度={self.entropy_temperature}, 目标熵比={self.target_entropy_ratio}")
            logger.info(f"早停参数: 耐心={self.early_stop_patience}, 组成比例阈值={self.loss_composition_delta}, 总损失阈值={self.total_loss_delta}")
            logger.info("=" * 60)
    
    def format_loss_report(self, loss_dict, iteration=0, step_time=0.0):
        """格式化损失报告"""
        # 主要损失报告
        main_losses = []
        for loss_name in ['total_loss', 'constraint_loss', 'mask_loss', 'transfer_loss', 'water_constraint_loss', 'sharpening_loss', 'ewc_loss']:
            if loss_name in loss_dict:
                display_name = self.loss_display_names.get(loss_name, loss_name)
                value = loss_dict[loss_name]
                main_losses.append(f"{display_name}: {value:.4f}")
        
        main_report = f"[迭代 {iteration+1}] 损失 - " + ", ".join(main_losses)
        
        # 详细信息报告
        details = []
        if step_time > 0:
            details.append(f"计算时间: {step_time:.2f}s")
        
        detail_report = "附加信息 - " + ", ".join(details) if details else ""
        
        return main_report, detail_report
    
    def calculate_loss_composition(self, loss_dict):
        """计算损失组成比例"""
        # 提取主要损失分量
        main_losses = {}
        for component in self.loss_components:
            main_losses[component] = max(float(loss_dict.get(component, 0.0)), 0.0)
        
        # 计算总和和比例
        total = sum(main_losses.values())
        denom = max(total, 1e-12)
        
        composition = {}
        for component, value in main_losses.items():
            composition[component] = value / denom
        
        return composition, total
    
    def format_composition_report(self, composition):
        """格式化损失组成报告"""
        comp_items = []
        for component in self.loss_components:
            if component in composition:
                display_name = self.loss_display_names.get(component, component).replace('约束', '').replace('损失', '')
                comp_items.append(f"{display_name}={composition[component]:.4f}")
        
        return "损失组成: " + ", ".join(comp_items)
    
    def get_checkpoint_dir(self):
        """获取检查点目录路径"""
        weight_str = self.get_weight_string()
        return os.path.join("checkpoints", "constraint", weight_str)
    
    def should_early_stop(self, current_loss_dict, loss_history):
        """
        检查是否应该早停
        
        Args:
            current_loss_dict: 当前迭代的损失字典
            loss_history: 历史损失记录列表，每个元素是loss_dict
            
        Returns:
            bool: 是否应该早停
            str: 早停原因
        """
        if len(loss_history) < self.early_stop_patience:
            return False, "历史记录不足"
        
        # 计算当前损失组成比例
        current_composition, _ = self.calculate_loss_composition(current_loss_dict)
        current_total_loss = current_loss_dict.get('total_loss', 0.0)
        
        # 检查最近patience步的损失变化
        recent_history = loss_history[-self.early_stop_patience:]
        
        # 条件1: 检查损失组成比例的L1偏移
        composition_stable = True
        for i in range(1, len(recent_history)):
            prev_composition, _ = self.calculate_loss_composition(recent_history[i-1])
            curr_composition, _ = self.calculate_loss_composition(recent_history[i])
            
            # 计算L1偏移
            l1_offset = 0.0
            for component in self.loss_components:
                if component in prev_composition and component in curr_composition:
                    l1_offset += abs(curr_composition[component] - prev_composition[component])
            
            if l1_offset > self.loss_composition_delta:
                composition_stable = False
                break
        
        # 条件2: 检查总损失的L1变化
        total_loss_stable = True
        for i in range(1, len(recent_history)):
            prev_total = recent_history[i-1].get('total_loss', 0.0)
            curr_total = recent_history[i].get('total_loss', 0.0)
            
            loss_change = abs(curr_total - prev_total)
            if loss_change > self.total_loss_delta:
                total_loss_stable = False
                break
        
        # 检查当前迭代与最近历史的变化
        if len(recent_history) > 0:
            last_composition, _ = self.calculate_loss_composition(recent_history[-1])
            current_l1_offset = 0.0
            for component in self.loss_components:
                if component in last_composition and component in current_composition:
                    current_l1_offset += abs(current_composition[component] - last_composition[component])
            
            last_total = recent_history[-1].get('total_loss', 0.0)
            current_total_change = abs(current_total_loss - last_total)
            
            if current_l1_offset > self.loss_composition_delta:
                composition_stable = False
            if current_total_change > self.total_loss_delta:
                total_loss_stable = False
        
        # 只要有一个条件满足就早停
        if composition_stable and total_loss_stable:
            reason = (f"损失组成比例稳定(L1偏移 < {self.loss_composition_delta}) 且 "
                     f"总损失变化稳定(L1变化 < {self.total_loss_delta})")
            return True, reason
        
        return False, f"组成稳定: {composition_stable}, 总损失稳定: {total_loss_stable}"

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

def freeze_unused_branches(model):
    """冻结特征解码器参数"""
    logger.info("开始冻结特征解码器参数...")
    
    if hasattr(model, 'gru') and hasattr(model.gru, 'feature_decoder'):
        for param in model.gru.feature_decoder.parameters():
            param.requires_grad = False
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    return model

def augment_with_invalid_by_valid_mask(prob_maps: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    """使用确定性的有效性掩膜将 [C,H,W] 扩展为含无效类 [C+1,H,W]"""
    with torch.no_grad():
        valid_mask = valid_mask.to(device=prob_maps.device, dtype=prob_maps.dtype)
        C, H, W = prob_maps.shape
        out = torch.zeros(C + 1, H, W, device=prob_maps.device, dtype=prob_maps.dtype)
        out[:C] = prob_maps * valid_mask
        out[C:C+1] = 1.0 - valid_mask
    return out

def preload_training_data(config):
    """预加载训练数据到CPU内存，避免在各个worker中重复加载"""
    logger.info("开始预加载训练数据...")
    
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
def distributed_constraint_worker(rank, world_size, step, iter_num, final_mask, 
                                 num_constraint, transfer_constraint, config, 
                                 preloaded_data):
    """分布式约束训练工作进程"""
    if config is None:
        config = ConstraintConfig()
    
    # 设置分布式环境
    _setup_dist(rank, world_size)
    
    try:
        device = torch.device(f"cuda:{rank}")
        logger_name = f"distributed-constraint-rank{rank}"
        worker_logger = logging.getLogger(logger_name)
        worker_logger.setLevel(logging.INFO)
        
        if rank == 0:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(
                f'[Rank{rank}] %(levelname)s: %(message)s'
            ))
            worker_logger.addHandler(handler)
            worker_logger.info(f"启动分布式约束训练 - Rank {rank}/{world_size}")
        
        # 使用预加载的数据
        features = preloaded_data['features']
        valid_mask = preloaded_data['valid_mask']
        num_classes = preloaded_data['num_classes']
        coordinates = preloaded_data['coordinates']

        features = features[1:]
        valid_mask = torch.cat([valid_mask[1:], final_mask.unsqueeze(0)], dim=0)

        pixel_mask = torch.sum(final_mask, dim=0)>0.5

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

        constraint_dir = config.get_checkpoint_dir()

        # 查找最新的检查点文件
        current_iteration = 0
        loaded = False

        # 查找constraint目录下最新的数字命名检查点（保存格式是 {iter_num+1}.pth）
        if os.path.exists(constraint_dir):
            checkpoint_files = [f for f in os.listdir(constraint_dir) if f.endswith('.pth') and f.startswith('iter')]
            if checkpoint_files:
                # 按迭代次数排序，取最新的
                checkpoint_files.sort(key=lambda x: int(x.replace('iter', '').replace('.pth', '')))
                latest_checkpoint = checkpoint_files[-1]
                constraint_ckpt_path = os.path.join(constraint_dir, latest_checkpoint)
                
                state_dict = torch.load(constraint_ckpt_path, weights_only=True)
                
                # 加载模型参数，方式与stgan加载一致
                if 'generator_state_dict' in state_dict:
                    generator_state_dict = OrderedDict()
                    for k, v in state_dict['generator_state_dict'].items():
                        # 移除module前缀
                        if k.startswith('module.'):
                            new_key = k[7:]
                        else:
                            new_key = k
                        generator_state_dict[new_key] = v
                    try:
                        generator.load_state_dict(generator_state_dict, strict=True)
                        worker_logger.info(f"生成器加载成功！加载自文件: {constraint_ckpt_path}")
                    except Exception as e:
                        worker_logger.info(f"生成器加载失败: {e}")
                        current_keys = set(generator.state_dict().keys())
                        saved_keys = set(generator_state_dict.keys())
                        worker_logger.info(f"当前模型键数量: {len(current_keys)}")
                        worker_logger.info(f"保存的键数量: {len(saved_keys)}")
                        missing_keys = saved_keys - current_keys
                        unexpected_keys = current_keys - saved_keys
                        if missing_keys:
                            worker_logger.info(f"缺失的键: {list(missing_keys)[:10]}...")
                        if unexpected_keys:
                            worker_logger.info(f"意外的键: {list(unexpected_keys)[:10]}...")
                elif 'model_state_dict' in state_dict:
                    generator_state_dict = OrderedDict()
                    for k, v in state_dict['model_state_dict'].items():
                        # 移除module前缀
                        if k.startswith('module.'):
                            new_key = k[7:]
                        else:
                            new_key = k
                        generator_state_dict[new_key] = v
                    try:
                        generator.load_state_dict(generator_state_dict, strict=True)
                        worker_logger.info(f"生成器加载成功！加载自文件: {constraint_ckpt_path}")
                    except Exception as e:
                        worker_logger.info(f"生成器加载失败: {e}")
                        current_keys = set(generator.state_dict().keys())
                        saved_keys = set(generator_state_dict.keys())
                        worker_logger.info(f"当前模型键数量: {len(current_keys)}")
                        worker_logger.info(f"保存的键数量: {len(saved_keys)}")
                        missing_keys = saved_keys - current_keys
                        unexpected_keys = current_keys - saved_keys
                        if missing_keys:
                            worker_logger.info(f"缺失的键: {list(missing_keys)[:10]}...")
                        if unexpected_keys:
                            worker_logger.info(f"意外的键: {list(unexpected_keys)[:10]}...")
                else:
                    worker_logger.info("未找到可用的生成器权重字典，跳过加载。")
                
                # 加载训练状态
                current_iteration = state_dict.get("iter_num", 0)
                
                if rank == 0:
                    worker_logger.info(f"已加载constraint检查点: {constraint_ckpt_path}, 从迭代 {current_iteration} 继续训练")
                loaded = True

        # 如果没有找到constraint检查点，尝试加载stgan检查点或generator_state_dict
        if not loaded:
            stgan_path = os.path.join("checkpoints", "stgan", f"iter1500.pth")
            stgan_checkpoint = torch.load(stgan_path, weights_only=True)
            stgan_generator_state_dict = None
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
                    worker_logger.info(f"生成器加载成功！加载自文件: {stgan_path}")
                except Exception as e:
                    worker_logger.info(f"生成器加载失败: {e}")
                    # 打印当前模型的键
                    current_keys = set(generator.state_dict().keys())
                    saved_keys = set(stgan_generator_state_dict.keys())
                    worker_logger.info(f"当前模型键数量: {len(current_keys)}")
                    worker_logger.info(f"保存的键数量: {len(saved_keys)}")
                    missing_keys = saved_keys - current_keys
                    unexpected_keys = current_keys - saved_keys
                    if missing_keys:
                        worker_logger.info(f"缺失的键: {list(missing_keys)[:10]}...")
                    if unexpected_keys:
                        worker_logger.info(f"意外的键: {list(unexpected_keys)[:10]}...")

        ewc_instance = create_ewc(generator, device, ewc_type="standard", lambda_ewc=1000.0)
        ewc_pretrain_path = "checkpoints/ewc_pretrain_state.pkl"
        if os.path.exists(ewc_pretrain_path):
            ewc_instance.load_ewc_state(ewc_pretrain_path)
            worker_logger.info("已加载预训练EWC状态")

        generator = DDP(generator, device_ids=[rank])

        # 统一的损失历史字典结构
        if not loaded:
            loss_history = {
                'total_loss': [],
                'constraint_loss': [],
                'mask_loss': [],
                'transfer_loss': [],
                'water_constraint_loss': [],
                'sharpening_loss': [],
                'ewc_loss': []
            }
        else:
            # 从检查点加载损失历史，兼容新旧格式
            loss_history = {
                'total_loss': state_dict.get("total_loss", state_dict.get("total_losses", [])),
                'constraint_loss': state_dict.get("constraint_loss", state_dict.get("constraint_losses", [])),
                'mask_loss': state_dict.get("mask_loss", state_dict.get("mask_losses", [])),
                'transfer_loss': state_dict.get("transfer_loss", state_dict.get("transfer_losses", [])),
                'water_constraint_loss': state_dict.get("water_constraint_loss", state_dict.get("water_constraint_losses", [])),
                'sharpening_loss': state_dict.get("sharpening_loss", state_dict.get("sharpening_losses", [])),
                'ewc_loss': state_dict.get("ewc_loss", state_dict.get("ewc_losses", []))
            }

        learning_rate = config.learning_rate

        for iter in range(current_iteration, iter_num):
            prob_imgs = []
            
            # 每10个iter，学习率衰减一次
            learning_rate = config.learning_rate * (config.learning_rate_decay ** (iter // 10))


            if rank==0:
                logger.info(f"迭代次数: {iter}/{iter_num}")

            for _step in range(step):
                if _step < step - 1:
                    # 只做推理，不进行训练
                    with torch.no_grad():
                        next_label_full, next_feature_full = infer_step(
                            generator=generator, xseq=features, mask=valid_mask, coordinates=coordinates,
                            num_classes=num_classes, feature_channels=features.shape[1] - valid_mask.shape[1],
                            device=device, img_size=(features.shape[2], features.shape[3]),
                            rank=rank, world_size=world_size
                        )
                        next_label_full = next_label_full.cpu()
                        next_feature_full = next_feature_full.cpu()

                        prob_img = augment_with_invalid_by_valid_mask(next_label_full.clone(), pixel_mask)
                        prob_imgs.append(prob_img)

                        next_x = torch.cat([next_label_full, next_feature_full], dim=0)

                        features = torch.cat([features[1:], next_x.unsqueeze(0)], dim=0)

                        valid_mask = torch.cat([valid_mask[1:], final_mask.unsqueeze(0)], dim=0)
                else:
                    # 最后一步才执行训练
                    probs, _, loss_dict = distributed_constraint_train(
                        generator=generator, xseq=features, mask=valid_mask, coordinates=coordinates,
                        num_classes=num_classes, feature_channels=features.shape[1] - valid_mask.shape[1],
                        device=device, constraint_matrix=num_constraint,
                        transfer_constraint=transfer_constraint, img_size=(features.shape[2], features.shape[3]),
                        ewc_instance=ewc_instance, config=config,
                        current_iteration=iter, learning_rate=learning_rate,
                        rank=rank, world_size=world_size
                    )

                    probs = probs.detach().cpu()

                    prob_img = augment_with_invalid_by_valid_mask(probs.clone(), pixel_mask)
                    prob_imgs.append(prob_img)

                    # 统一记录损失到字典结构
                    for loss_name in loss_history.keys():
                        if loss_name in loss_dict:
                            loss_history[loss_name].append(loss_dict[loss_name])

                    if rank == 0:                    
                        main_report, detail_report = config.format_loss_report(loss_dict, iteration=iter, step_time=0.0)

                        worker_logger.info(main_report)
                        worker_logger.info(detail_report)

                        composition, _ = config.calculate_loss_composition(loss_dict)

                        worker_logger.info(config.format_composition_report(composition))

            # 新的早停策略：基于损失组成比例L1偏移和总损失L1变化
            if rank == 0 and len(loss_history['total_loss']) > 0:
                # 构建历史损失记录列表（转换为early_stop函数期望的格式）
                loss_records = []
                num_records = len(loss_history['total_loss'])
                for i in range(num_records):
                    loss_record = {}
                    for loss_name, loss_values in loss_history.items():
                        if i < len(loss_values):
                            loss_record[loss_name] = loss_values[i]
                        else:
                            loss_record[loss_name] = 0.0
                    loss_records.append(loss_record)
                
                # 检查早停条件
                current_loss_dict = loss_records[-1]
                should_stop, reason = config.should_early_stop(current_loss_dict, loss_records[:-1])
                
                # 打印早停信息
                worker_logger.info(
                    f"[EarlyStopping] Iter: {iter+1}, 总损失: {current_loss_dict['total_loss']:.6f}, "
                    f"历史记录: {len(loss_records)}, 检查结果: {reason}"
                )
                # 只在rank==0下执行
                if not hasattr(config, "auto_lr_silent_counter"):
                    config.auto_lr_silent_counter = 0
                if rank == 0 and len(loss_history['total_loss']) >= 20:
                    import numpy as np
                    # 如果处于静默期，则计数并跳过本轮自动调整
                    if config.auto_lr_silent_counter > 0:
                        config.auto_lr_silent_counter -= 1
                    else:
                        # 维护一个连续15次训练的损失均值
                        last_20_losses = np.array(loss_history['total_loss'][-15:])
                        mean_1 = np.mean(last_20_losses[:5])
                        mean_2 = np.mean(last_20_losses[5:])
                        # 如果损失均值不明显下降（下降幅度未达到5%），就下调学习率
                        if (mean_1 - mean_2) < 0.05 * mean_1:
                            old_lr = learning_rate
                            learning_rate = learning_rate * 0.5
                            worker_logger.info(
                                f"[AutoLR] 检测到最近10次损失均值未明显下降（前10次均值={mean_1:.6f}, 后10次均值={mean_2:.6f}），"
                                f"自动调整学习率: {old_lr:.6e} -> {learning_rate:.6e}，进入静默"
                            )
                            config.auto_lr_silent_counter = 6
                
                if should_stop:
                    worker_logger.info(f"[EarlyStopping] 训练在第 {iter+1} 迭代停止。原因: {reason}")
                    if rank == 0 and iter % config.visualize_interval == 0:
                            from visual import visualize_sequence_tensor
                            import matplotlib.pyplot as plt
                            plt.figure(figsize=(12, 4))
                            vis_dir = config.get_checkpoint_dir()
                            os.makedirs(vis_dir, exist_ok=True)

                            prob_imgs = torch.stack(prob_imgs, dim=0)
                            print(prob_imgs.shape)
                            visualize_sequence_tensor(prob_imgs, save_path=os.path.join(vis_dir, f"prob_imgs_{iter+1}.png"))

                            save_dir = config.get_checkpoint_dir()
                            os.makedirs(save_dir, exist_ok=True)
                            # 将模型、损失和迭代次数保存在同一个文件中
                            save_data = {
                                "model_state_dict": generator.state_dict(),
                                "iter_num": iter + 1,
                                **loss_history  # 直接使用统一的损失历史字典
                            }

                            torch.save(prob_imgs, os.path.join(save_dir, f"prob_imgs.pth"))
                            torch.save(save_data, os.path.join(save_dir, f"iter{iter+1}.pth"))
                            if rank == 0:
                                worker_logger.info(f"模型、损失和迭代次数已保存至 {save_dir}，迭代次数: {iter+1}")
                    break

            
            if rank == 0 and iter % (config.visualize_interval / 2) == 0:
                from visual import visualize_sequence_tensor
                import matplotlib.pyplot as plt
                plt.figure(figsize=(12, 4))
                vis_dir = config.get_checkpoint_dir()
                os.makedirs(vis_dir, exist_ok=True)

                prob_imgs = torch.stack(prob_imgs, dim=0)
                visualize_sequence_tensor(prob_imgs, save_path=os.path.join(vis_dir, f"prob_imgs_{iter+1}.png"))

            if rank == 0 and iter % config.visualize_interval == 0:
                save_dir = config.get_checkpoint_dir()
                os.makedirs(save_dir, exist_ok=True)
                # 将模型、损失和迭代次数保存在同一个文件中
                save_data = {
                    "model_state_dict": generator.state_dict(),
                    "iter_num": iter + 1,
                    **loss_history  # 直接使用统一的损失历史字典
                }

                torch.save(prob_imgs, os.path.join(save_dir, f"prob_imgs.pth"))
                torch.save(save_data, os.path.join(save_dir, f"iter{iter+1}.pth"))
                if rank == 0:
                    worker_logger.info(f"模型、损失和迭代次数已保存至 {save_dir}，迭代次数: {iter+1}")

    finally:
        _cleanup_dist()

# ============ 统一训练入口 ============

def unified_constraint_training(step, iter_num, final_mask, num_constraint, 
                               transfer_constraint, config=None, world_size=None):
    """
    统一约束训练入口函数
    根据world_size自动选择分布式或非分布式训练
    """
    
    # 使用默认配置
    if config is None:
        config = ConstraintConfig()
    
    # 自动检测GPU数量
    if world_size is None:
        world_size = torch.cuda.device_count()
    
    # 设置环境变量
    config.setup_distributed_env()
    
    # 配置日志
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        logger.addHandler(logging.StreamHandler())
    
    log_path = f"logs/unified_constraint_train_{time.strftime('%Y%m%d_%H%M%S')}.log"
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger.addHandler(logging.FileHandler(log_path))
    
    # 记录配置信息
    config.log_config_summary(logger, rank=0)

    logger.info(f"启动分布式约束训练，使用 {world_size} 个GPU")
    # 预加载训练数据到CPU内存
    logger.info("预加载训练数据...")
    preloaded_data = preload_training_data(config)
    
    # 启动多进程分布式训练
    mp.spawn(
        distributed_constraint_worker,
        args=(
            world_size, step, iter_num, final_mask, num_constraint, 
            transfer_constraint, config, preloaded_data
        ),
        nprocs=world_size,
        join=True,
    )

if __name__ == "__main__":
    # 示例使用
    print("=== 统一约束训练模块测试 ===")
    
    # 加载数据
    try:
        import rasterio
        with rasterio.open("/root/autodl-fs/mask/2035/2035b.tif") as src:
            final_mask = src.read()
            final_mask = torch.from_numpy(final_mask[:6])
    except:
        final_mask = torch.ones(6, 1024, 1024)
        print("警告：使用模拟终期掩膜")

    try:
        num_constraint = torch.load("constraint/2035.pth", weights_only=True)
    except:
        num_constraint = torch.zeros(3, 6)
        print("警告：使用模拟数量约束")

    try:
        transfer_constraint = torch.load("constraint/transfer_matrix.pth", weights_only=True)
    except:
        transfer_constraint = torch.zeros(6, 6)
        print("警告：使用模拟转移约束")
    
    
    print("配置完成，开始统一约束训练...")
    
    # 启动统一训练（自动检测GPU数量）
    unified_constraint_training(
        step=3, 
        iter_num=100,  # 减少迭代次数用于测试
        final_mask=final_mask, 
        num_constraint=num_constraint, 
        transfer_constraint=transfer_constraint,
        config=None
    )