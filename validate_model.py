#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证集推理文件 - 使用训练好的模型在验证集上进行推理并计算性能指标
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
import logging
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from collections import OrderedDict
import matplotlib.pyplot as plt
import seaborn as sns

# 导入自定义模块
from models.gan import Generator
from data.dataset import PatchDataset
from utils import get_data

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelValidator:
    """模型验证器"""
    
    def __init__(self, model_path, data_config):
        """
        初始化验证器
        
        Args:
            model_path: 模型权重文件路径
            data_config: 数据配置字典
        """
        self.model_path = model_path
        self.data_config = data_config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = 6  # 6个用地类型
        self.class_names = ['Cropland', 'Woodland', 'Grassland', 'Water', 'Built-up', 'Unused']
        
        # 加载数据
        self.features, self.labels, self.masks, self.valid_coords, self.invalid_coords = self._load_data()
        # 加载模型
        self.model = self._load_model()
        
    # 已在初始化时获取到相关信息，无需再次实现_get_data_info方法
    def _load_model(self):
        """加载训练好的模型"""
        logger.info(f"Loading model from {self.model_path}")
        
        # 从数据中获取实际的特征通道数
        temp_features = self.features
        actual_feature_channels = temp_features.shape[1] - self.num_classes  # 减去标签通道
        
        # 创建模型
        model = Generator(
            hidden_channels=32,
            num_classes=6,
            feature_decoder=True,
            features_channels=actual_feature_channels
        ).to(self.device)
        
        # 加载检查点
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=True)
        
        # 计算总通道数
        total_channels = self.num_classes * 2 + actual_feature_channels
        
        # 创建虚拟输入来初始化模型
        dummy_input = torch.randn(1, 1, total_channels, 512, 512).to(self.device)
        with torch.no_grad():
            _ = model(dummy_input)
        
        # 加载权重
        if 'generator_state_dict' in checkpoint:
            model_state_dict = OrderedDict()
            for k, v in checkpoint['generator_state_dict'].items():
                # 移除DDP的module前缀
                if k.startswith('module.'):
                    k = k[7:]
                model_state_dict[k] = v
            model.load_state_dict(model_state_dict, strict=True)
        else:
            logger.error("No generator_state_dict found in checkpoint")
            raise ValueError("Invalid checkpoint file")
        
        model.eval()
        logger.info("Model loaded successfully")
        return model
    
    def _load_data(self):
        """加载验证数据"""
        logger.info("Loading validation data...")
        
        factors_data, labels_data, valid_mask, valid_coordinates, invalid_coordinates, num_classes = get_data(
            self.data_config['factors_path'],
            self.data_config['labels_path'],
            num_patches=1000  # 临时使用，后面会重新采样
        )
        
        logger.info(f"Data loaded - Features: {factors_data.shape}, Labels: {labels_data.shape}")
        return factors_data, labels_data, valid_mask, valid_coordinates, invalid_coordinates
    
    def sample_validation_coordinates(self, valid_samples=800, invalid_samples=160):
        """
        使用get_data重新采样验证坐标
        
        Args:
            valid_samples: 有效像元采样数量
            invalid_samples: 无效像元采样数量
        """
        logger.info(f"Sampling {valid_samples} valid pixels and {invalid_samples} invalid pixels using get_data")
        
        _, _, _, valid_coordinates, invalid_coordinates, _ = get_data(
            self.data_config['factors_path'],
            self.data_config['labels_path'],
            num_patches=valid_samples
        )
        
        # get_data返回的坐标数量可能不完全等于请求的数量，需要调整
        available_valid = len(valid_coordinates)
        available_invalid = len(invalid_coordinates)
        
        # 调整采样数量
        actual_valid_samples = min(valid_samples, available_valid)
        actual_invalid_samples = min(invalid_samples, available_invalid)
        
        logger.info(f"Available coordinates: {available_valid} valid, {available_invalid} invalid")
        logger.info(f"Using: {actual_valid_samples} valid, {actual_invalid_samples} invalid")
        
        # 选择指定数量的坐标
        selected_valid = valid_coordinates[:actual_valid_samples]
        selected_invalid = invalid_coordinates[:actual_invalid_samples]
        
        # 合并坐标
        all_coords = torch.cat([selected_valid, selected_invalid], dim=0)
        
        # 创建标签：0表示有效像元，1表示无效像元
        pixel_types = torch.cat([
            torch.zeros(actual_valid_samples, dtype=torch.long),
            torch.ones(actual_invalid_samples, dtype=torch.long)
        ])
        
        logger.info(f"Sampled {len(all_coords)} pixels total ({actual_valid_samples} valid + {actual_invalid_samples} invalid)")
        return all_coords, pixel_types
    
    def predict_with_dataset(self, coordinates, pixel_types, patch_size=512, batch_size=24):
        """
        使用Dataset方式进行预测
        
        Args:
            coordinates: 坐标列表
            pixel_types: 像元类型（0=有效，1=无效）
            patch_size: patch大小
            batch_size: 批处理大小
        
        Returns:
            predictions: 预测结果
            true_labels: 真实标签
            valid_masks: 有效性掩膜
            pixel_types: 像元类型
        """
        logger.info(f"Predicting {len(coordinates)} pixels using Dataset...")
        
        # 创建验证数据集（不使用数据增强）
        from data.dataset import PatchDataset
        val_dataset = PatchDataset(
            self.features, 
            self.labels, 
            self.masks, 
            coordinates, 
            patch_size=patch_size,
            corner_sampling=False,  # 使用中心采样
            enhancement=False  # 关闭数据增强
        )
        
        # 创建数据加载器
        from torch.utils.data import DataLoader
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0  # 避免多进程问题
        )
        
        all_predictions = []
        all_true_labels = []
        all_valid_masks = []
        
        # 批量预测
        for batch_idx, batch_data in enumerate(tqdm(val_dataloader, desc="Predicting")):
            # batch_data格式: [features_patch, labels_patch, masks_patch, position]
            features_patch, labels_patch, masks_patch, position = batch_data
            
            # 移动到设备
            features_patch = features_patch[:,:4].to(self.device)
            labels_patch = labels_patch.to(self.device)
            masks_patch = masks_patch.to(self.device)
            
            # 构造模型输入：concatenate features and labels
            # 输入格式：(B, T, C_total, H, W)
            batch_input = torch.cat([features_patch, labels_patch], dim=2)  # (B, T, C_total, H, W)
            
            B, T, C_total, H, W = batch_input.shape
            
            with torch.no_grad():
                # 模型预测
                model_output = self.model(batch_input)
                if isinstance(model_output, tuple):
                    predictions, _ = model_output  # (B, T, C_labels, H, W)
                else:
                    predictions = model_output  # (B, T, C_labels, H, W)
                
                # 应用softmax获得概率
                predictions = predictions[:,-1]
                labels_patch = labels_patch[:,-1]
                masks_patch = masks_patch[:,-1]

                # 根据masks_patch前六个通道的和是否大于0.5定义mask，大于0.5为有效
                # masks_patch形状: (B, C, H, W)，C>=6
                mask_valid = (masks_patch[:, :6, :, :].sum(dim=1) > 0.5)
                
                # 保存整个patch的预测结果，而不仅仅是中心像元
                all_predictions.append(predictions.cpu())
                all_true_labels.append(labels_patch.cpu())
                all_valid_masks.append(mask_valid.cpu())
        
        # 合并所有批次的结果
        predictions = torch.cat(all_predictions, dim=0)
        true_labels = torch.cat(all_true_labels, dim=0)
        valid_masks = torch.cat(all_valid_masks, dim=0)
        
        logger.info("Prediction completed")
        return predictions, true_labels, valid_masks, pixel_types
    
    def calculate_metrics(self, predictions, true_labels, valid_masks):
        """
        计算性能指标，处理无效区域和整个patch（仅最后一期）
        
        Args:
            predictions: 模型预测概率，形状为 (N, C, H, W) - 仅最后一期
            true_labels: 真实标签，形状为 (N, C, H, W) - 仅最后一期
            valid_masks: 有效性掩膜，形状为 (N, H, W) - 前6个通道和>0.5的mask
            pixel_types: 像元类型，形状为 (N,)，0=有效，1=无效
        
        Returns:
            metrics: 包含各种指标的字典
        """
        logger.info("Calculating metrics for entire patches (last time step only)...")
        
        N, C, H, W = predictions.shape
        
        # 使用argmax获得预测类别和真实类别
        pred_classes = torch.argmax(predictions, dim=1)  # (N, H, W)
        true_classes = torch.argmax(true_labels, dim=1)  # (N, H, W)

        print(pred_classes.shape, true_classes.shape, valid_masks.shape)
        
        # 展平为一维，只保留有效的预测点
        # 直接用张量运算实现各项指标，无需转numpy和sklearn
        valid_indices = valid_masks.flatten()
        pred_flat = pred_classes.flatten()[valid_indices]
        true_flat = true_classes.flatten()[valid_indices]

        if pred_flat.numel() == 0:
            logger.warning("No valid predictions found!")
            return {}

        num_classes = self.num_classes
        # 使用cuda算子高效计算混淆矩阵
        with torch.no_grad():
            cm = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=pred_flat.device)
            idx = true_flat * num_classes + pred_flat
            cm_flat = torch.bincount(idx, minlength=num_classes * num_classes)
            cm = cm_flat.reshape(num_classes, num_classes)
            cm = cm.cpu().numpy()

        # 计算每类的TP, FP, FN
        TP = torch.diag(torch.from_numpy(cm)).float()
        FP = torch.from_numpy(cm).sum(dim=0).float() - TP
        FN = torch.from_numpy(cm).sum(dim=1).float() - TP
        support = torch.from_numpy(cm).sum(dim=1).float()

        # 避免除零
        precision = TP / (TP + FP + 1e-8)
        recall = TP / (TP + FN + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        # 计算macro和micro
        macro_precision = precision.mean().item()
        macro_recall = recall.mean().item()
        macro_f1 = f1.mean().item()

        total_TP = TP.sum()
        total_FP = FP.sum()
        total_FN = FN.sum()
        micro_precision = (total_TP / (total_TP + total_FP + 1e-8)).item()
        micro_recall = (total_TP / (total_TP + total_FN + 1e-8)).item()
        micro_f1 = (2 * micro_precision * micro_recall / (micro_precision + micro_recall + 1e-8))

        # 准确率
        accuracy = (TP.sum() / support.sum()).item()

        # 转为numpy
        precision = precision.numpy()
        recall = recall.numpy()
        f1 = f1.numpy()
        support = support.numpy()
        metrics = {
            # 主要指标
            'accuracy': accuracy,
            'precision_per_class': precision,
            'recall_per_class': recall,
            'f1_per_class': f1,
            'support_per_class': support,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1,
            'micro_precision': micro_precision,
            'micro_recall': micro_recall,
            'micro_f1': micro_f1,
            'confusion_matrix': cm,
            
            # 总体统计
            'total_accuracy': accuracy,  # 与accuracy相同，因为只有有效区域
            'total_confusion_matrix': cm,  # 与cm相同
            'valid_samples': len(pred_flat),
            'invalid_samples': 0,  # 无效样本已被过滤
            'total_samples': len(pred_flat)
        }
        
        logger.info(f"Metrics calculated for {len(pred_flat)} valid samples")
        return metrics
    
    def save_results_to_excel(self, metrics, output_path='validation_results.xlsx'):
        """
        将结果保存为Excel文件
        
        Args:
            metrics: 计算得到的指标字典
            output_path: 输出文件路径
        """
        logger.info(f"Saving results to {output_path}")
        
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # 1. 整体指标
            overall_metrics = pd.DataFrame({
                'Metric': ['Accuracy', 'Macro Precision', 'Macro Recall', 'Macro F1', 
                          'Micro Precision', 'Micro Recall', 'Micro F1', 'Total Samples'],
                'Value': [
                    metrics['accuracy'],
                    metrics['macro_precision'],
                    metrics['macro_recall'], 
                    metrics['macro_f1'],
                    metrics['micro_precision'],
                    metrics['micro_recall'],
                    metrics['micro_f1'],
                    metrics['total_samples']
                ]
            })
            overall_metrics.to_excel(writer, sheet_name='Overall Metrics', index=False)
            
            # 2. 各类别指标
            class_metrics = pd.DataFrame({
                'Class': self.class_names,
                'Class_ID': range(self.num_classes),
                'Precision': metrics['precision_per_class'],
                'Recall': metrics['recall_per_class'],
                'F1_Score': metrics['f1_per_class'],
                'Support': metrics['support_per_class']
            })
            class_metrics.to_excel(writer, sheet_name='Class Metrics', index=False)
            
            # 3. 混淆矩阵
            cm_df = pd.DataFrame(
                metrics['confusion_matrix'],
                columns=[f'Pred_{name}' for name in self.class_names],
                index=[f'True_{name}' for name in self.class_names]
            )
            cm_df.to_excel(writer, sheet_name='Confusion Matrix')
            
            # 4. 添加元数据
            metadata = pd.DataFrame({
                'Item': ['Model Path', 'Number of Classes', 'Class Names', 'Device Used'],
                'Value': [
                    self.model_path,
                    self.num_classes,
                    ', '.join(self.class_names),
                    str(self.device)
                ]
            })
            metadata.to_excel(writer, sheet_name='Metadata', index=False)
        
        logger.info(f"Results saved to {output_path}")
    
    def plot_confusion_matrix(self, metrics, save_path='confusion_matrix.png'):
        """
        绘制混淆矩阵图
        
        Args:
            metrics: 指标字典
            save_path: 保存路径
        """
        plt.figure(figsize=(10, 8))
        
        # 计算归一化的混淆矩阵
        cm = metrics['confusion_matrix']
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # 绘制热图
        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt='.3f',
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar_kws={'label': 'Normalized Count'}
        )
        
        plt.title('Confusion Matrix (Normalized) - Last Time Step', fontsize=16)
        plt.xlabel('Predicted Class', fontsize=12)
        plt.ylabel('True Class', fontsize=12)
        plt.tight_layout()
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Confusion matrix plot saved to {save_path}")
        plt.close()
    
    def run_validation(self, valid_samples=800, invalid_samples=160, output_dir='validation_output', patch_size=512, batch_size=24):
        """
        运行完整的验证流程
        
        Args:
            valid_samples: 有效像元采样数量
            invalid_samples: 无效像元采样数量
            output_dir: 输出目录
            patch_size: patch大小
            batch_size: 批处理大小
        """
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. 采样验证坐标
        all_coords, pixel_types = self.sample_validation_coordinates(
            valid_samples, invalid_samples
        )
        
        # 2. 使用Dataset进行预测
        predictions, true_labels, valid_masks, pixel_types = self.predict_with_dataset(
            all_coords, pixel_types, patch_size=patch_size, batch_size=batch_size
        )
        
        # 3. 计算指标
        metrics = self.calculate_metrics(predictions, true_labels, valid_masks)
        
        if not metrics:
            logger.error("No metrics calculated, validation failed")
            return
        
        # 4. 保存结果
        excel_path = os.path.join(output_dir, 'validation_results.xlsx')
        self.save_results_to_excel(metrics, excel_path)
        
        # 5. 绘制混淆矩阵
        cm_path = os.path.join(output_dir, 'confusion_matrix.png')
        self.plot_confusion_matrix(metrics, cm_path)
        
        # 6. 打印摘要
        logger.info("=== Validation Results Summary ===")
        logger.info(f"Total samples: {metrics['total_samples']}")
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Macro F1: {metrics['macro_f1']:.4f}")
        logger.info(f"Micro F1: {metrics['micro_f1']:.4f}")
        
        logger.info("\nPer-class metrics (entire patches, last time step):")
        for i, class_name in enumerate(self.class_names):
            logger.info(f"{class_name}: P={metrics['precision_per_class'][i]:.4f}, "
                       f"R={metrics['recall_per_class'][i]:.4f}, "
                       f"F1={metrics['f1_per_class'][i]:.4f}")
        
        logger.info("Validation completed successfully!")
        return metrics


def main():
    """主函数"""
    # 配置数据路径
    data_config = {
        'factors_path': '/root/autodl-fs/factors',
        'labels_path': '/root/autodl-fs/label_floats',
        'features_channels': 10
    }
    
    # 模型路径
    model_path = 'checkpoints/stgan_20250813/iter1500.pth'
    
    # 创建验证器
    validator = ModelValidator(model_path, data_config)
    
    # 运行验证
    metrics = validator.run_validation(
        valid_samples=400,
        invalid_samples=80,
        output_dir='validation_output'
    )


if __name__ == "__main__":
    main()
