import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import matplotlib
# 设置matplotlib后端
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import numpy as np

def plot_training_losses(checkpoint_path):
    """
    加载模型检查点并绘制所有损失值对训练迭代数的曲线
    
    Args:
        checkpoint_path: 模型检查点文件路径
    """
    
    # 检查文件是否存在
    if not os.path.exists(checkpoint_path):
        print(f"错误：文件 {checkpoint_path} 不存在")
        return
    
    try:
        # 加载检查点
        print(f"正在加载检查点: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # 提取损失数据
        losses = {
            'true_losses': checkpoint.get('true_losses', []),
            'fake_losses': checkpoint.get('fake_losses', []),
            'disc_losses': checkpoint.get('disc_losses', []),
            'gen_losses': checkpoint.get('gen_losses', []),
            'focal_losses': checkpoint.get('focal_losses', []),
            'bce_losses': checkpoint.get('bce_losses', [])
        }
        
        iter_num = checkpoint.get('iter_num', 0)
        epoch = checkpoint.get('epoch', 0)
        
        print(f"训练信息:")
        print(f"  当前迭代数: {iter_num}")
        print(f"  当前轮次: {epoch}")
        
        # 检查哪些损失数据存在
        available_losses = {k: v for k, v in losses.items() if v and len(v) > 0}
        
        if not available_losses:
            print("错误：没有找到任何损失数据")
            return
        
        print(f"找到的损失类型: {list(available_losses.keys())}")
        
        # 创建迭代数数组
        iterations = list(range(1, len(next(iter(available_losses.values()))) + 1))
        
        # 创建子图
        num_losses = len(available_losses)
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        # 颜色映射
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
        # 绘制每种损失
        for i, (loss_name, loss_values) in enumerate(available_losses.items()):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            # 绘制损失曲线
            ax.plot(iterations, loss_values, color=colors[i % len(colors)], linewidth=2, label=loss_name)
            
            # 设置标题和标签
            ax.set_title(f'{loss_name} 损失曲线', fontsize=14, fontweight='bold')
            ax.set_xlabel('训练迭代数', fontsize=12)
            ax.set_ylabel('损失值', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # 添加统计信息
            if len(loss_values) > 0:
                min_loss = min(loss_values)
                max_loss = max(loss_values)
                avg_loss = np.mean(loss_values)
                ax.text(0.02, 0.98, f'最小值: {min_loss:.4f}\n最大值: {max_loss:.4f}\n平均值: {avg_loss:.4f}', 
                       transform=ax.transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # 隐藏多余的子图
        for i in range(num_losses, len(axes)):
            axes[i].set_visible(False)
        
        # 调整布局
        plt.tight_layout()
        
        # 添加总标题
        fig.suptitle(f'训练损失曲线 (迭代数: {iter_num}, 轮次: {epoch})', 
                    fontsize=16, fontweight='bold', y=1.02)
        
        # 保存图片
        output_path = checkpoint_path.replace('.pth', '_losses.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"损失曲线图已保存到: {output_path}")
        
        # 关闭图形以释放内存
        plt.close(fig)
        
        # 打印损失统计信息
        print("\n损失统计信息:")
        for loss_name, loss_values in available_losses.items():
            if len(loss_values) > 0:
                print(f"{loss_name}:")
                print(f"  最小值: {min(loss_values):.6f}")
                print(f"  最大值: {max(loss_values):.6f}")
                print(f"  平均值: {np.mean(loss_values):.6f}")
                print(f"  标准差: {np.std(loss_values):.6f}")
                print(f"  数据点数: {len(loss_values)}")
                print()
        
    except Exception as e:
        print(f"加载检查点时出错: {str(e)}")
        import traceback
        traceback.print_exc()

def plot_loss_comparison(checkpoint_path):
    """
    绘制损失对比图，将所有损失放在同一个图中进行比较
    """
    
    try:
        # 加载检查点
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # 提取损失数据
        losses = {
            'true_losses': checkpoint.get('true_losses', []),
            'fake_losses': checkpoint.get('fake_losses', []),
            'disc_losses': checkpoint.get('disc_losses', []),
            'gen_losses': checkpoint.get('gen_losses', []),
            'focal_losses': checkpoint.get('focal_losses', []),
            'bce_losses': checkpoint.get('bce_losses', [])
        }
        
        # 过滤有效的损失数据
        available_losses = {k: v for k, v in losses.items() if v and len(v) > 0}
        
        if not available_losses:
            print("没有找到有效的损失数据")
            return
        
        # 创建迭代数数组
        iterations = list(range(1, len(next(iter(available_losses.values()))) + 1))
        
        
        # 创建对比图
        fig = plt.figure(figsize=(15, 8))
        
        # 颜色映射
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
        # 绘制所有损失
        for i, (loss_name, loss_values) in enumerate(available_losses.items()):
            plt.plot(iterations, loss_values, color=colors[i % len(colors)], 
                    linewidth=2, label=loss_name, alpha=0.8)
        
        plt.title('训练损失对比图', fontsize=16, fontweight='bold')
        plt.xlabel('训练迭代数', fontsize=12)
        plt.ylabel('损失值', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)
        
        # 保存对比图
        output_path = checkpoint_path.replace('.pth', '_comparison.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"损失对比图已保存到: {output_path}")
        
        # 关闭图形以释放内存
        plt.close(fig)
        
    except Exception as e:
        print(f"绘制对比图时出错: {str(e)}")

def plot_loss_trends(checkpoint_path):
    """
    绘制损失趋势分析图，包括移动平均线
    """
    
    try:
        # 加载检查点
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # 提取损失数据
        losses = {
            'true_losses': checkpoint.get('true_losses', []),
            'fake_losses': checkpoint.get('fake_losses', []),
            'disc_losses': checkpoint.get('disc_losses', []),
            'gen_losses': checkpoint.get('gen_losses', []),
            'focal_losses': checkpoint.get('focal_losses', []),
            'bce_losses': checkpoint.get('bce_losses', [])
        }
        
        # 过滤有效的损失数据
        available_losses = {k: v for k, v in losses.items() if v and len(v) > 0}
        
        if not available_losses:
            print("没有找到有效的损失数据")
            return
        
        # 创建迭代数数组
        iterations = list(range(1, len(next(iter(available_losses.values()))) + 1))
        
        
        # 创建趋势分析图
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        # 颜色映射
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
        # 绘制每种损失的趋势
        for i, (loss_name, loss_values) in enumerate(available_losses.items()):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            # 计算移动平均线（窗口大小为5）
            window_size = min(5, len(loss_values) // 4)
            if window_size > 1:
                moving_avg = np.convolve(loss_values, np.ones(window_size)/window_size, mode='valid')
                moving_avg_iterations = iterations[window_size-1:]
                
                # 绘制原始数据和移动平均线
                ax.plot(iterations, loss_values, color=colors[i % len(colors)], 
                       linewidth=1, alpha=0.6, label=f'{loss_name} (原始)')
                ax.plot(moving_avg_iterations, moving_avg, color=colors[i % len(colors)], 
                       linewidth=2, label=f'{loss_name} (移动平均)')
            else:
                ax.plot(iterations, loss_values, color=colors[i % len(colors)], 
                       linewidth=2, label=loss_name)
            
            # 设置标题和标签
            ax.set_title(f'{loss_name} 损失趋势', fontsize=14, fontweight='bold')
            ax.set_xlabel('训练迭代数', fontsize=12)
            ax.set_ylabel('损失值', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        # 隐藏多余的子图
        for i in range(len(available_losses), len(axes)):
            axes[i].set_visible(False)
        
        # 调整布局
        plt.tight_layout()
        
        # 添加总标题
        fig.suptitle('训练损失趋势分析', fontsize=16, fontweight='bold', y=1.02)
        
        # 保存趋势图
        output_path = checkpoint_path.replace('.pth', '_trends.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"损失趋势图已保存到: {output_path}")
        
        # 关闭图形以释放内存
        plt.close(fig)
        
    except Exception as e:
        print(f"绘制趋势图时出错: {str(e)}")

def visualize_tensor(tensor, save_path, title="Visualize"):
    """
    输入一张 (C, H, W) 的多通道张量，通过argmax获取地类并可视化。

    参数:
        tensor (torch.Tensor): 形状为 (C, H, W)
        save_path (str): 保存路径
        title (str): 图像标题
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np

    # 定义类别颜色映射
    # 0: 耕地（淡黄色），1: 林地（深绿色），2: 草地（浅绿色），3: 水域（淡蓝色），4: 建设用地（浅红色），5: 未利用地（灰色），6: 无效值（透明）
    class_colors = {
        0: (1.0, 1.0, 0.6, 1.0),    # 淡黄色
        1: (0.0, 0.4, 0.0, 1.0),    # 深绿色
        2: (0.6, 1.0, 0.6, 1.0),    # 浅绿色
        3: (0.6, 0.8, 1.0, 1.0),    # 淡蓝色
        4: (1.0, 0.6, 0.6, 1.0),    # 浅红色
        5: (0.7, 0.7, 0.7, 1.0),    # 灰色
        6: (0.0, 0.0, 0.0, 0.0),    # 透明
    }
    class_names = {
        0: "Cropland",
        1: "Forest",
        2: "Grassland",
        3: "Water",
        4: "Built-up",
        5: "Unused",
    }

    C, H, W = tensor.shape

    # 转为numpy
    tensor_np = tensor.clone().detach().cpu().numpy()  # (C, H, W)

    # 通过argmax获取地类标签
    label_map = np.argmax(tensor_np, axis=0)  # (H, W)

    rgb = np.zeros((H, W, 4), dtype=np.float32)
    for k, color in class_colors.items():
        mask = (label_map == k)
        rgb[mask] = color

    # 创建画布
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(rgb)
    ax.axis('off')

    # 构建图例
    legend_patches = []
    for k in range(6):  # 只显示0-5，6为无效值不显示
        color = class_colors[k]
        legend_patches.append(
            mpatches.Patch(color=color, label=class_names[k])
        )

    # 添加图例
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    fig.legend(
        handles=legend_patches,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.02),
        ncol=len(legend_patches),
        fontsize=12,
        title="Legend"
    )

    if title:
        fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Temporal tensor visualization saved to: {save_path}")



def visualize_sequence_tensor(tensor, save_path, titles=None):
    """
    输入一张 (T, C, H, W) 的多时序多通道张量，对每个时间点通过argmax获取地类并可视化为子图。

    参数:
        tensor (torch.Tensor): 形状为 (T, C, H, W)
        save_path (str): 保存路径
        title (str): 图像标题
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
    import math

    # 定义类别颜色映射
    # 0: Cropland (light yellow), 1: Forest (dark green), 2: Grassland (light green), 3: Water (light blue), 4: Built-up (light red), 5: Unused (gray), 6: Invalid (transparent)
    class_colors = {
        0: (1.0, 1.0, 0.6, 1.0),    # light yellow
        1: (0.0, 0.4, 0.0, 1.0),    # dark green
        2: (0.6, 1.0, 0.6, 1.0),    # light green
        3: (0.6, 0.8, 1.0, 1.0),    # light blue
        4: (1.0, 0.6, 0.6, 1.0),    # light red
        5: (0.7, 0.7, 0.7, 1.0),    # gray
        6: (0.0, 0.0, 0.0, 0.0),    # transparent
    }
    class_names = {
        0: "Cropland",
        1: "Forest",
        2: "Grassland",
        3: "Water",
        4: "Built-up",
        5: "Unused",
    }

    # 转为numpy
    tensor_np = tensor.clone().detach().cpu().numpy()  # (T, C, H, W)
    T, C, H, W = tensor_np.shape

    # 自动计算子图布局
    ncols = min(T, 4)
    nrows = math.ceil(T / ncols)

    # 增大figsize并设置wspace/hspace拉开子图距离
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, 
        figsize=(5.5*ncols, 5.5*nrows),  # 原4->5.5，整体更宽松
        squeeze=False
    )

    # 拉开子图间距
    plt.subplots_adjust(wspace=0.35, hspace=0.35)  # 增大间距

    for t in range(T):
        row = t // ncols
        col = t % ncols
        ax = axes[row, col]

        # 通过argmax获取地类标签
        label_map = np.argmax(tensor_np[t], axis=0)  # (H, W)

        # 生成RGB图像
        rgb = np.zeros((H, W, 4), dtype=np.float32)
        for k, color in class_colors.items():
            mask = (label_map == k)
            rgb[mask] = color

        ax.imshow(rgb)
        ax.axis('off')
        if titles is not None:
            ax.set_title(titles[t], fontsize=16, pad=16)  # pad加大标题与图像距离

    # 去除多余的子图
    for t in range(T, nrows * ncols):
        row = t // ncols
        col = t % ncols
        axes[row, col].axis('off')

    # 构建图例
    legend_patches = []
    for k in range(6):  # 只显示0-5，6为无效值不显示
        color = class_colors[k]
        legend_patches.append(
            mpatches.Patch(color=color, label=class_names[k])
        )

    # 添加图例
    plt.tight_layout(rect=[0, 0.08, 1, 1])  # rect下移，给图例更多空间
    fig.legend(
        handles=legend_patches,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.06),  # 图例整体下移
        ncol=len(legend_patches),
        fontsize=14,
        title="Legend"
    )

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Temporal tensor visualization saved to: {save_path}")


if __name__ == "__main__":
    tensor = torch.randn(3, 6, 512, 512)
    visualize_sequence_tensor(tensor, "test.png", titles=["2025", "2030", "Predicted 3"])