import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from typing import List, Dict, Tuple, Optional
import torch

import matplotlib
matplotlib.rcParams['font.family'] = 'Times New Roman'


def compute_kappa(y_pred, y_true, num_classes=None, invalid_class=6):
    """
    计算Kappa系数，忽略无效值（如第6类，值为5）
    :param y_pred: 预测结果，shape=(H,W)
    :param y_true: 真实标签，shape=(H,W)
    :param num_classes: 类别数，若为None则自动推断
    :param invalid_class: 无效类别的数值
    :return: kappa系数
    """
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()
    # 只保留有效像元
    valid_mask = (y_true != invalid_class) & (y_pred != invalid_class)
    y_pred = y_pred[valid_mask]
    y_true = y_true[valid_mask]
    if y_true.size == 0:
        return 0
    if num_classes is None:
        num_classes = max(y_pred.max(), y_true.max()) + 1
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        confusion[t, p] += 1
    total = confusion.sum()
    if total == 0:
        return 0
    po = np.trace(confusion) / total
    pe = (confusion.sum(axis=0) * confusion.sum(axis=1)).sum() / (total * total)
    kappa = (po - pe) / (1 - pe) if (1 - pe) != 0 else 0
    return kappa

def compute_oa(y_pred, y_true, invalid_class=6):
    """
    计算全局精度（Overall Accuracy, OA），忽略无效值（如第6类，值为5）
    :param y_pred: 预测结果，shape=(H,W)
    :param y_true: 真实标签，shape=(H,W)
    :param invalid_class: 无效类别的数值
    :return: OA
    """
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()
    # 只保留有效像元
    valid_mask = (y_true != invalid_class) & (y_pred != invalid_class)
    y_pred = y_pred[valid_mask]
    y_true = y_true[valid_mask]
    if y_true.size == 0:
        return 0
    correct = (y_pred == y_true).sum()
    total = y_true.size
    oa = correct / total
    return oa

def compute_confusion_matrix(y_pred, y_true, num_classes=None, invalid_class=6):
    """
    计算混淆矩阵（数值和百分比），忽略无效值（如第6类，值为5）
    :param y_pred: 预测结果，shape=(H,W)
    :param y_true: 真实标签，shape=(H,W)
    :param num_classes: 类别数，若为None则自动推断
    :param invalid_class: 无效类别的数值
    :return: confusion_matrix(数值), confusion_matrix_percent(百分比)
    """
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()
    # 只保留有效像元
    valid_mask = (y_true != invalid_class) & (y_pred != invalid_class)
    y_pred = y_pred[valid_mask]
    y_true = y_true[valid_mask]
    if y_true.size == 0:
        if num_classes is None:
            num_classes = 1
        confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
        confusion_percent = np.zeros((num_classes, num_classes), dtype=np.float32)
        return confusion, confusion_percent
    if num_classes is None:
        num_classes = max(y_pred.max(), y_true.max()) + 1
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        confusion[t, p] += 1
    # 百分比矩阵，按真实标签（行）归一化
    with np.errstate(divide='ignore', invalid='ignore'):
        confusion_percent = confusion / confusion.sum(axis=1, keepdims=True)
        confusion_percent = np.nan_to_num(confusion_percent)  # 避免除零
    return confusion, confusion_percent

def compute_fom(y_pred, y_true, y_old, num_classes=None, invalid_class=6):
    """
    计算多分类情况下的FoM（Figure of Merit）指标，结合y_old判断转变过程，适用于土地利用变化模拟精度评价（Pontius et al., 2008）。
    对每个类别分别计算FoM，返回一个字典。

    :param y_pred: 预测结果，shape=(H,W)
    :param y_true: 真实标签，shape=(H,W)
    :param y_old: 变化前的标签，shape=(H,W)
    :param num_classes: 有效类别数（不含无效类别）
    :param invalid_class: 无效类别的数值
    :return: dict，key为类别，value为FoM系数，范围[0,1]
    """
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()
    y_old = y_old.flatten()
    # 只保留有效像元
    valid_mask = (y_true != invalid_class) & (y_pred != invalid_class) & (y_old != invalid_class)
    y_pred = y_pred[valid_mask]
    y_true = y_true[valid_mask]
    y_old = y_old[valid_mask]

    fom_dict = {}
    for target_class in range(num_classes):
        # 真实发生了从y_old到target_class的变化
        true_change = (y_old != target_class) & (y_true == target_class)
        # 预测发生了从y_old到target_class的变化
        pred_change = (y_old != target_class) & (y_pred == target_class)

        # A: 真实发生了变化，但预测未能捕捉到（漏报变化）
        A = np.logical_and(true_change, y_pred != target_class).sum()

        # B: 真实发生了变化，且预测也正确捕捉到（命中）
        B = np.logical_and(true_change, y_pred == target_class).sum()

        # C: 真实发生了变化，但预测为其他错误类别（多分类下，预测为非target_class的其他类别）
        C = np.logical_and(true_change, (y_pred != target_class) & (y_pred != y_old)).sum()

        # D: 真实未发生该变化，但预测为发生了该变化（虚警）
        D = np.logical_and(~true_change, pred_change).sum()

        denom = A + B + C + D
        fom = B / denom if denom != 0 else 0
        fom_dict[target_class] = fom
    return fom_dict

import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, class_names=None, normalize=False, cmap=plt.cm.Blues, save_path=None):
    """
    绘制混淆矩阵（英文标记），不绘制第七类（即最后一类）

    :param cm: 混淆矩阵 (numpy数组)
    :param class_names: 类别名称列表
    :param normalize: 是否归一化显示
    :param cmap: 颜色映射
    :param save_path: 保存路径（如为None则只显示不保存）
    """
    # 去除第七类（假设为最后一行和最后一列）
    if cm.shape[0] > 6 and cm.shape[1] > 6:
        cm = cm[:6, :6]
        if class_names is not None and len(class_names) > 6:
            class_names = class_names[:6]
    elif cm.shape[0] > 6:
        cm = cm[:6, :]
        if class_names is not None and len(class_names) > 6:
            class_names = class_names[:6]
    elif cm.shape[1] > 6:
        cm = cm[:, :6]
        if class_names is not None and len(class_names) > 6:
            class_names = class_names[:6]

    if normalize:
        cm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-8)
        fmt = ".2f"
    else:
        fmt = "d"

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar()
    if class_names is not None:
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45, ha='right', fontname='Times New Roman')
        plt.yticks(tick_marks, class_names, fontname='Times New Roman')
    else:
        tick_marks = np.arange(cm.shape[0])
        plt.xticks(tick_marks, tick_marks, fontname='Times New Roman')
        plt.yticks(tick_marks, tick_marks, fontname='Times New Roman')

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black",
                     fontname='Times New Roman')

    plt.ylabel('True label', fontname='Times New Roman')
    plt.xlabel('Predicted label', fontname='Times New Roman')
    plt.title('Confusion Matrix', fontname='Times New Roman')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")
    else:
        plt.show()

def comprehensive_classification_evaluation(
    y_pred_list: List[torch.Tensor], 
    y_real_list: List[torch.Tensor], 
    y_base_list: List[torch.Tensor],
    num_classes: int = 6,
    invalid_class: int = 5,  # 修改：从6改为5，因为现在只有6类（索引0-5）
    class_names: Optional[List[str]] = None
) -> Dict:
    """
    综合机器学习分类模型评估函数
    
    Args:
        y_pred_list: 预测结果列表，每个元素为(B,C,H,W)张量，B为批次大小
        y_real_list: 真实标签列表，每个元素为(B,C,H,W)张量
        y_base_list: 基准标签列表（用于FoM计算），每个元素为(B,C,H,W)张量
        num_classes: 有效类别数（不包括无效类别）
        invalid_class: 无效类别值
        class_names: 类别名称列表
    
    Returns:
        包含所有评估指标的字典
    """
    if class_names is None:
        class_names = [f"Class_{i}" for i in range(num_classes)]
    
    # 初始化累积变量
    all_y_pred = []
    all_y_real = []
    all_y_base = []
    
    # 处理每个批次的数据
    for y_pred_batch, y_real_batch, y_base_batch in zip(y_pred_list, y_real_list, y_base_list):
        # 将(B,C,H,W)转换为(B,H,W) - 取argmax
        if y_pred_batch.dim() == 4:
            y_pred_batch = torch.argmax(y_pred_batch, dim=1)  # (B,H,W)
        if y_real_batch.dim() == 4:
            y_real_batch = torch.argmax(y_real_batch, dim=1)  # (B,H,W)
        if y_base_batch.dim() == 4:
            y_base_batch = torch.argmax(y_base_batch, dim=1)  # (B,H,W)
        
        # 展平并收集有效像素
        for i in range(y_pred_batch.shape[0]):
            y_pred = y_pred_batch[i].flatten().cpu().numpy()
            y_real = y_real_batch[i].flatten().cpu().numpy()
            y_base = y_base_batch[i].flatten().cpu().numpy()
            
            # 只保留有效像素
            valid_mask = (y_real != invalid_class) & (y_base != invalid_class) & (y_pred != invalid_class)
            y_pred = y_pred[valid_mask]
            y_real = y_real[valid_mask]
            y_base = y_base[valid_mask]
            
            all_y_pred.extend(y_pred)
            all_y_real.extend(y_real)
            all_y_base.extend(y_base)
    
    # 转换为numpy数组
    y_pred = np.array(all_y_pred)
    y_real = np.array(all_y_real)
    y_base = np.array(all_y_base)
    
    # 若无有效像元，直接返回全零结果
    if y_real.size == 0:
        results = {
            'overall_accuracy': 0,
            'total_accuracy': 0,
            'kappa_coefficient': 0,
            'mean_iou': 0,
            'fom_average': 0,
            'precision_macro': 0,
            'recall_macro': 0,
            'f1_macro': 0,
            'precision_weighted': 0,
            'recall_weighted': 0,
            'f1_weighted': 0,
            'class_metrics': {},
            'confusion_matrix': np.zeros((num_classes, num_classes), dtype=np.int64),
            'confusion_matrix_normalized': np.zeros((num_classes, num_classes), dtype=np.float32)
        }
        for i in range(num_classes):
            results['class_metrics'][class_names[i]] = {
                'precision': 0,
                'recall': 0,
                'f1_score': 0,
                'support': 0,
                'iou': 0,
                'producer_accuracy': 0,
                'user_accuracy': 0,
                'fom': 0
            }
        return results

    # 计算基础指标
    oa = accuracy_score(y_real, y_pred)
    
    # 计算精确率、召回率、F1分数
    precision, recall, f1, support = precision_recall_fscore_support(
        y_real, y_pred, average=None, labels=range(num_classes), zero_division=0
    )
    
    # 计算宏平均和加权平均
    precision_macro = precision.mean()
    recall_macro = recall.mean()
    f1_macro = f1.mean()
    
    # 计算加权平均
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_real, y_pred, average='weighted', labels=range(num_classes), zero_division=0
    )
    
    # 计算混淆矩阵
    cm = confusion_matrix(y_real, y_pred, labels=range(num_classes))
    
    # 计算Kappa系数
    kappa = compute_kappa(y_pred, y_real, num_classes, invalid_class)
    
    # 计算FoM指标
    fom_dict = compute_fom(y_pred, y_real, y_base, num_classes, invalid_class)
    fom_avg = np.mean(list(fom_dict.values()))
    
    # 计算每个类别的IoU (Intersection over Union)
    iou_dict = {}
    for i in range(num_classes):
        tp = cm[i, i]  # True Positive
        fp = cm[:, i].sum() - tp  # False Positive
        fn = cm[i, :].sum() - tp  # False Negative
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
        iou_dict[class_names[i]] = iou
    
    # 计算平均IoU
    mean_iou = np.mean(list(iou_dict.values()))
    
    # 计算总体精度 (Total Accuracy)
    total_accuracy = np.trace(cm) / np.sum(cm) if np.sum(cm) > 0 else 0
    
    # 计算生产者精度 (Producer's Accuracy) - 按行计算
    producer_accuracy = np.diag(cm) / np.sum(cm, axis=1)
    producer_accuracy = np.nan_to_num(producer_accuracy)  # 处理除零
    
    # 计算用户精度 (User's Accuracy) - 按列计算
    user_accuracy = np.diag(cm) / np.sum(cm, axis=0)
    user_accuracy = np.nan_to_num(user_accuracy)  # 处理除零
    
    # 构建结果字典
    results = {
        # 总体指标
        'overall_accuracy': oa,
        'total_accuracy': total_accuracy,
        'kappa_coefficient': kappa,
        'mean_iou': mean_iou,
        'fom_average': fom_avg,
        
        # 宏平均指标
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        
        # 加权平均指标
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted,
        
        # 每个类别的详细指标
        'class_metrics': {}
    }
    
    # 为每个类别添加详细指标
    for i in range(num_classes):
        results['class_metrics'][class_names[i]] = {
            'precision': precision[i],
            'recall': recall[i],
            'f1_score': f1[i],
            'support': support[i],
            'iou': iou_dict[class_names[i]],
            'producer_accuracy': producer_accuracy[i],
            'user_accuracy': user_accuracy[i],
            'fom': fom_dict[i]
        }
    
    # 添加混淆矩阵
    results['confusion_matrix'] = cm
    results['confusion_matrix_normalized'] = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)
    
    return results

def print_evaluation_summary(results: Dict, save_path: Optional[str] = None):
    """
    打印评估结果摘要
    
    Args:
        results: comprehensive_classification_evaluation的返回结果
        save_path: 保存路径（可选）
    """
    output_lines = []
    
    # 总体指标
    output_lines.append("=" * 60)
    output_lines.append("综合分类评估结果")
    output_lines.append("=" * 60)
    output_lines.append(f"总体精度 (Overall Accuracy): {results['overall_accuracy']:.4f}")
    output_lines.append(f"总精度 (Total Accuracy): {results['total_accuracy']:.4f}")
    output_lines.append(f"Kappa系数: {results['kappa_coefficient']:.4f}")
    output_lines.append(f"平均IoU: {results['mean_iou']:.4f}")
    output_lines.append(f"平均FoM: {results['fom_average']:.4f}")
    output_lines.append("")
    
    # 宏平均和加权平均
    output_lines.append("宏平均指标:")
    output_lines.append(f"  精确率: {results['precision_macro']:.4f}")
    output_lines.append(f"  召回率: {results['recall_macro']:.4f}")
    output_lines.append(f"  F1分数: {results['f1_macro']:.4f}")
    output_lines.append("")
    
    output_lines.append("加权平均指标:")
    output_lines.append(f"  精确率: {results['precision_weighted']:.4f}")
    output_lines.append(f"  召回率: {results['recall_weighted']:.4f}")
    output_lines.append(f"  F1分数: {results['f1_weighted']:.4f}")
    output_lines.append("")
    
    # 每个类别的详细指标
    output_lines.append("各类别详细指标:")
    output_lines.append("-" * 60)
    for class_name, metrics in results['class_metrics'].items():
        output_lines.append(f"{class_name}:")
        output_lines.append(f"  精确率: {metrics['precision']:.4f}")
        output_lines.append(f"  召回率: {metrics['recall']:.4f}")
        output_lines.append(f"  F1分数: {metrics['f1_score']:.4f}")
        output_lines.append(f"  IoU: {metrics['iou']:.4f}")
        output_lines.append(f"  生产者精度: {metrics['producer_accuracy']:.4f}")
        output_lines.append(f"  用户精度: {metrics['user_accuracy']:.4f}")
        output_lines.append(f"  FoM: {metrics['fom']:.4f}")
        output_lines.append(f"  支持度: {metrics['support']}")
        output_lines.append("")
    
    # 打印结果
    for line in output_lines:
        print(line)
    
    # 保存结果
    if save_path:
        with open(save_path, 'w', encoding='utf-8') as f:
            for line in output_lines:
                f.write(line + '\n')
        print(f"评估结果已保存到: {save_path}")

def example_usage():
    """
    使用示例：展示如何使用comprehensive_classification_evaluation函数
    """
    import torch
    import numpy as np
    
    # 模拟数据：创建一些示例张量列表
    batch_size = 2
    num_classes = 6
    height, width = 256, 256
    
    # 创建预测结果列表 (模拟模型输出)
    y_pred_list = []
    for _ in range(3):  # 3个批次
        # 创建one-hot编码的预测结果
        pred = torch.randn(batch_size, num_classes, height, width)
        y_pred_list.append(pred)
    
    # 创建真实标签列表
    y_real_list = []
    for _ in range(3):
        # 创建one-hot编码的真实标签
        real = torch.zeros(batch_size, num_classes, height, width)
        # 随机分配类别
        labels = torch.randint(0, num_classes, (batch_size, height, width))
        for b in range(batch_size):
            for h in range(height):
                for w in range(width):
                    real[b, labels[b, h, w], h, w] = 1
        y_real_list.append(real)
    
    # 创建基准标签列表（用于FoM计算）
    y_base_list = []
    for _ in range(3):
        # 创建one-hot编码的基准标签
        base = torch.zeros(batch_size, num_classes, height, width)
        # 随机分配类别
        labels = torch.randint(0, num_classes, (batch_size, height, width))
        for b in range(batch_size):
            for h in range(height):
                for w in range(width):
                    base[b, labels[b, h, w], h, w] = 1
        y_base_list.append(base)
    
    # 定义类别名称
    class_names = ["Cropland", "Forest", "Grassland", "Water", "Built-up", "Barren"]
    
    # 执行综合评估
    results = comprehensive_classification_evaluation(
        y_pred_list=y_pred_list,
        y_real_list=y_real_list,
        y_base_list=y_base_list,
        num_classes=6,
        invalid_class=6,
        class_names=class_names
    )
    
    # 打印评估结果
    print_evaluation_summary(results, save_path="evaluation_results.txt")
    
    # 访问特定指标
    print(f"\n总体精度: {results['overall_accuracy']:.4f}")
    print(f"Kappa系数: {results['kappa_coefficient']:.4f}")
    print(f"平均IoU: {results['mean_iou']:.4f}")
    print(f"平均FoM: {results['fom_average']:.4f}")
    
    # 访问特定类别的指标
    print(f"\n农田类别的F1分数: {results['class_metrics']['Cropland']['f1_score']:.4f}")
    print(f"森林类别的IoU: {results['class_metrics']['Forest']['iou']:.4f}")

if __name__ == "__main__":
    import os
    import re
    import rasterio
    import numpy as np
    import pandas as pd

    # 获取outputs文件夹下所有infer和real的tif文件及其年份
    infer_files = []
    real_files = []
    infer_years = []
    real_years = []

    for fname in os.listdir("outputs"):
        infer_match = re.match(r"infer_(\d{4})\.tif", fname)
        real_match = re.match(r"real_(\d{4})\.tif", fname)
        if infer_match:
            year = int(infer_match.group(1))
            infer_files.append((year, os.path.join("outputs", fname)))
            infer_years.append(year)
        if real_match:
            year = int(real_match.group(1))
            real_files.append((year, os.path.join("outputs", fname)))
            real_years.append(year)

    infer_files.sort()
    real_files.sort()
    infer_years = sorted(list(set(infer_years)))
    real_years = sorted(list(set(real_years)))

    # 读取所有真实标签
    real_labels = {}
    for year, path in real_files:
        with rasterio.open(path) as src:
            arr = src.read()
            # 如果是one-hot，取argmax
            if arr.shape[0] > 1:
                arr = np.argmax(arr, axis=0)
            else:
                arr = arr[0]
            real_labels[year] = arr

    # 读取所有预测标签
    pred_labels = {}
    for year, path in infer_files:
        with rasterio.open(path) as src:
            arr = src.read(1)
            pred_labels[year] = arr

    # 统计每个时间点的各项系数
    results = {}
    kappa_dict = {}
    oa_dict = {}
    fom_dict_all = {}  # {year: {old_year: {class: fom, ...}, ...}}
    fom_avg_dict = {}  # {year: {old_year: avg_fom, ...}}

    for year in infer_years:
        y_pred = pred_labels[year]
        y_true = real_labels[year]
        # 只保留有效像元
        invalid_class = 6
        valid_mask = (y_true != invalid_class) & (y_pred != invalid_class)
        y_pred_valid = y_pred[valid_mask]
        y_true_valid = y_true[valid_mask]

        kappa = compute_kappa(y_pred, y_true, invalid_class=invalid_class)
        oa = compute_oa(y_pred, y_true, invalid_class=invalid_class)
        confusion, confusion_percent = compute_confusion_matrix(y_pred, y_true, invalid_class=invalid_class)
        results[year] = {
            "kappa": kappa,
            "oa": oa,
            "confusion": confusion,
            "confusion_percent": confusion_percent
        }
        kappa_dict[year] = kappa
        oa_dict[year] = oa

        print(f"==== {year}年 ====")
        print("Kappa：", kappa)
        print("OA：", oa)
        fom_dict = {}
        fom_avg_list = []
        fom_avg_dict[year] = {}
        # FoM的计算：对于每个y_pred（如2015），与所有比它早的y_true（如2010、2005、2000、1995）分别计算FoM
        old_years = [y for y in real_years if y < year]
        for old in old_years:
            y_old = real_labels[old]
            # FoM计算时也要去除无效像元
            valid_mask_fom = (y_true != invalid_class) & (y_pred != invalid_class) & (y_old != invalid_class)
            fom = compute_fom(
                y_pred[valid_mask_fom],
                y_true[valid_mask_fom],
                y_old[valid_mask_fom],
                num_classes=6,
                invalid_class=invalid_class
            )
            fom_dict[old] = fom
            fom_avg = np.mean(list(fom.values()))
            fom_avg_list.append(fom_avg)
            fom_avg_dict[year][old] = fom_avg
            print(f"与{old}年作为old的FoM：", fom)
            print(f"与{old}年作为old的平均FoM：", fom_avg)
        fom_dict_all[year] = fom_dict
        if fom_avg_list:
            print(f"{year}年所有FoM平均值：", np.mean(fom_avg_list))
        else:
            print(f"{year}年没有可用的old年份进行FoM计算。")
        # 绘制混淆矩阵
        plot_confusion_matrix(confusion, class_names=["Cropland", "Forest", "Grassland",  "Water", "Built-up", "Barren"], normalize=False, save_path=f"confusion_matrix_{year}.png")
        plot_confusion_matrix(confusion_percent, class_names=["Cropland", "Forest", "Grassland", "Water", "Built-up", "Barren"], normalize=True, save_path=f"confusion_matrix_percent_{year}.png")

    # 整理Kappa和OA到DataFrame
    kappa_oa_df = pd.DataFrame({
        "Year": infer_years,
        "Kappa": [kappa_dict[y] for y in infer_years],
        "OA": [oa_dict[y] for y in infer_years]
    })

    # 整理FoM到DataFrame
    # 以(year, old_year)为行，class为列
    fom_rows = []
    for year in fom_dict_all:
        for old in fom_dict_all[year]:
            row = {"Year": year, "Old_Year": old}
            for cls, fom_val in fom_dict_all[year][old].items():
                row[f"Class_{cls}"] = fom_val
            fom_rows.append(row)
    fom_df = pd.DataFrame(fom_rows)
    # 平均FoM
    fom_avg_rows = []
    for year in fom_avg_dict:
        for old in fom_avg_dict[year]:
            fom_avg_rows.append({
                "Year": year,
                "Old_Year": old,
                "FoM_Avg": fom_avg_dict[year][old]
            })
    fom_avg_df = pd.DataFrame(fom_avg_rows)

    # 写入Excel
    with pd.ExcelWriter("outputs/metrics_summary.xlsx") as writer:
        kappa_oa_df.to_excel(writer, sheet_name="Kappa_OA", index=False)
        fom_df.to_excel(writer, sheet_name="FoM", index=False)
        fom_avg_df.to_excel(writer, sheet_name="FoM_Avg", index=False)
    print("所有指标已保存到 outputs/metrics_summary.xlsx")