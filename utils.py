import sys
import os
import random
import torch

# Add project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch.nn.functional as F
from data.loader import find_common_valid_positions_class, sample_positions
from data.utils import load_hierarchical_tif,get_tensor_dimensions

import logging
import sys

# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def get_data(factors_data_path: str, labels_data_path: str, num_patches: int = 1000):
    logger.info("开始加载因子数据...")
    factors_data, _= load_hierarchical_tif(factors_data_path)
    if factors_data is not None:
        logger.info(f"因子数据加载完成，形状: {factors_data.shape}")
    
    logger.info("开始加载标签数据...")
    labels_data, _ = load_hierarchical_tif(labels_data_path)
    if labels_data is not None:
        logger.info(f"标签数据加载完成，形状: {labels_data.shape}")

    assert factors_data is not None
    assert labels_data is not None

    valid_positions, invalid_positions, valid_mask = find_common_valid_positions_class(labels_data)
    coordinates = sample_positions(valid_positions, num_patches)
    invalid_coordinates = sample_positions(invalid_positions, int(num_patches * 0.2))

    coordinates = torch.tensor(coordinates).to(labels_data.device)
    invalid_coordinates = torch.tensor(invalid_coordinates).to(labels_data.device)

    factors_data = torch.cat([factors_data[:, :6, ...], factors_data[:, 7:, ...]], dim=1)
    labels_data = labels_data[:,:6]
    valid_mask = valid_mask[:,:6]
    
    num_classes = 6  # 固定为6类，因为我们已经丢弃了第七类

    return factors_data, labels_data, valid_mask, coordinates, invalid_coordinates, num_classes

def overlay_mask(valid_mask: torch.Tensor, addition_mask: torch.Tensor) -> torch.Tensor:
    """
    叠加掩膜：输入valid_mask和addition_mask，取它们的交集（逐元素与），返回更新后的有效性掩膜。

    参数:
        valid_mask (torch.Tensor): 原始有效性掩膜，形状为 (T, C, H, W)
        addition_mask (torch.Tensor): 额外掩膜，形状需与valid_mask相同，或可广播到相同形状

    返回:
        torch.Tensor: 更新后的有效性掩膜
    """
    # 检查形状是否一致或可广播
    if valid_mask.shape != addition_mask.shape:
        try:
            addition_mask = addition_mask.expand_as(valid_mask)
        except Exception as e:
            raise ValueError(f"addition_mask的形状{addition_mask.shape}无法广播到valid_mask的形状{valid_mask.shape}")

    # 取交集（逐元素与）
    updated_mask = valid_mask & addition_mask

    return updated_mask


def visualize_temporal_comparison(tensor1, tensor2, save_path, title1="Tensor 1", title2="Tensor 2"):
    """
    Visualize comparison of two temporal tensors
    
    Args:
        tensor1 (torch.Tensor): First tensor, shape (T, H, W)
        tensor2 (torch.Tensor): Second tensor, shape (T, H, W)
        save_path (str): Save path
        title1 (str): Title for first tensor
        title2 (str): Title for second tensor
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

    # 保证tensor在CPU
    if tensor1.device.type == 'cuda':
        tensor1 = tensor1.cpu()
    if tensor2.device.type == 'cuda':
        tensor2 = tensor2.cpu()

    # 转为numpy
    tensor1_np = tensor1.numpy()
    tensor2_np = tensor2.numpy()

    # 时间步数
    T = tensor1_np.shape[0]
    rows = 2
    cols = T

    # 生成RGB图像
    def label2rgb(label_img):
        # label_img: (H, W)，值为0-6
        h, w = label_img.shape
        rgb = np.zeros((h, w, 4), dtype=np.float32)
        for k, color in class_colors.items():
            mask = (label_img == k)
            rgb[mask] = color
        return rgb

    # 创建画布
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    if rows == 1:
        axes = axes.reshape(1, -1)
    if cols == 1:
        axes = axes.reshape(-1, 1)

    for t in range(T):
        img1 = tensor1_np[t]
        img2 = tensor2_np[t]
        rgb1 = label2rgb(img1)
        rgb2 = label2rgb(img2)

        im1 = axes[0, t].imshow(rgb1)
        axes[0, t].set_title(f'{title1} t={t}')
        axes[0, t].axis('off')

        im2 = axes[1, t].imshow(rgb2)
        axes[1, t].set_title(f'{title2} t={t}')
        axes[1, t].axis('off')

    # 构建图例
    legend_patches = []
    for k in range(6):  # 只显示0-5，6为无效值不显示
        color = class_colors[k]
        legend_patches.append(
            mpatches.Patch(color=color, label=class_names[k])
        )

    # 调整布局，图例放在下方一横排
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    fig.legend(
        handles=legend_patches,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.02),
        ncol=len(legend_patches),
        fontsize=12,
        title="Legend"
    )

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Comparison plot saved to: {save_path}")


def plot_tif_distribution(tif_path: str, save_path = None, title: str = "TIF File Value Distribution", 
                         bins: int = 50, kernel: str = 'gaussian', bandwidth = None):
    """
    Read the first band of a tif file and plot histogram and kernel density curve
    Note: Zero values are excluded from the analysis
    
    Args:
        tif_path: Path to tif file
        save_path: Save path, if None then display the image
        title: Image title
        bins: Number of histogram bins
        kernel: Kernel function type ('gaussian', 'tophat', 'epanechnikov', 'exponential', 'linear', 'cosine')
        bandwidth: Bandwidth parameter, if None then auto-calculate
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.stats import gaussian_kde
    import rasterio
    
    # Read tif file
    try:
        with rasterio.open(tif_path) as src:
            # Read first band
            data = src.read(1)  # Shape: (H, W)

            print(data.shape)
            
            # Flatten data to 1D array and filter invalid values and zeros
            valid_data = data[~np.isnan(data) & (data != src.nodata) & (data != 0) if src.nodata is not None else ~np.isnan(data) & (data != 0)]
            
            if len(valid_data) == 0:
                raise ValueError("No valid data found")
                
    except Exception as e:
        logger.error(f"Cannot read tif file {tif_path}: {str(e)}")
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot histogram
    n, bins_edges, patches = ax.hist(valid_data, bins=bins, density=True, alpha=0.7, 
                                     color='skyblue', edgecolor='black', label='Histogram')
    
    # Calculate kernel density estimation
    if bandwidth is None:
        # Use Scott's rule to auto-calculate bandwidth
        bandwidth = 1.06 * np.std(valid_data) * (len(valid_data) ** (-1/5))
    
    # Create kernel density estimation object
    kde = gaussian_kde(valid_data, bw_method=bandwidth/np.std(valid_data))
    
    # Generate x values for plotting kernel density curve
    x_range = np.linspace(valid_data.min(), valid_data.max(), 200)
    kde_values = kde(x_range)
    
    # Plot kernel density curve
    ax.plot(x_range, kde_values, 'r-', linewidth=2, label=f'Kernel Density ({kernel})')
    
    # Set figure properties
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add statistical information
    stats_text = f'样本数量: {len(valid_data):,}\n'
    stats_text += f'均值: {np.mean(valid_data):.4f}\n'
    stats_text += f'标准差: {np.std(valid_data):.4f}\n'
    stats_text += f'最小值: {np.min(valid_data):.4f}\n'
    stats_text += f'最大值: {np.max(valid_data):.4f}\n'
    stats_text += f'(已排除0值)'
    
    # Add statistical information text box in upper right corner
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, 
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            fontsize=9)
    
    plt.tight_layout()
    
    # Save or display image
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Distribution plot saved to: {save_path}")
    else:
        plt.show()


def plot_tif_distribution_optimized(tif_path: str, save_path = None, title: str = "TIF File Value Distribution", 
                                   bins: int = 50, kernel: str = 'gaussian', bandwidth = None,
                                   max_samples: int = 100000, use_sampling: bool = True, 
                                   chunk_size: int = 1000000, progress_bar: bool = True):
    """
    Optimized TIF file distribution plotting function for large datasets
    Note: Zero values are excluded from analysis
    
    Args:
        tif_path: TIF file path
        save_path: Save path, None to display image
        title: Image title
        bins: Number of histogram bins
        kernel: Kernel function type
        bandwidth: Bandwidth parameter, None for auto-calculation
        max_samples: Maximum number of sample points (for KDE calculation)
        use_sampling: Whether to use sampling to accelerate KDE calculation
        chunk_size: Chunk size for chunked processing
        progress_bar: Whether to show progress bar
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.stats import gaussian_kde
    import rasterio
    from tqdm import tqdm
    
    # Read TIF file
    try:
        with rasterio.open(tif_path) as src:
            data = src.read(1)
            print(f"Original data shape: {data.shape}")
            
            # Chunked processing for large datasets
            if data.size > chunk_size:
                print(f"Large dataset ({data.size:,} points), using chunked processing...")
                valid_data = []
                
                # Calculate number of chunks
                total_chunks = (data.size + chunk_size - 1) // chunk_size
                
                for i in tqdm(range(total_chunks), desc="Processing data chunks", disable=not progress_bar):
                    start_idx = i * chunk_size
                    end_idx = min((i + 1) * chunk_size, data.size)
                    
                    # Get current chunk data
                    chunk = data.flatten()[start_idx:end_idx]
                    
                    # Filter invalid values and zeros
                    if src.nodata is not None:
                        valid_chunk = chunk[~np.isnan(chunk) & (chunk != src.nodata) & (chunk != 0)]
                    else:
                        valid_chunk = chunk[~np.isnan(chunk) & (chunk != 0)]
                    
                    valid_data.extend(valid_chunk)
                
                valid_data = np.array(valid_data)
            else:
                # Direct processing for small datasets
                if src.nodata is not None:
                    valid_data = data[~np.isnan(data) & (data != src.nodata) & (data != 0)]
                else:
                    valid_data = data[~np.isnan(data) & (data != 0)]
                valid_data = valid_data.flatten()
            
            if len(valid_data) == 0:
                raise ValueError("No valid data found")
                
    except Exception as e:
        logger.error(f"Cannot read TIF file {tif_path}: {str(e)}")
        return
    
    print(f"Valid data points: {len(valid_data):,}")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot histogram (using all data)
    print("Plotting histogram...")
    n, bins_edges, patches = ax.hist(valid_data, bins=bins, density=True, alpha=0.7, 
                                     color='skyblue', edgecolor='black', label='Histogram')
    
    # Optimize KDE calculation
    if use_sampling and len(valid_data) > max_samples:
        print(f"Dataset too large, using {max_samples:,} random sample points for KDE calculation...")
        # Random sampling
        np.random.seed(42)  # Ensure reproducible results
        kde_data = np.random.choice(valid_data, size=max_samples, replace=False)
    else:
        kde_data = valid_data
    
    print("Calculating kernel density estimation...")
    # Calculate kernel density estimation
    if bandwidth is None:
        # Use Scott's rule to auto-calculate bandwidth
        bandwidth = 1.06 * np.std(kde_data) * (len(kde_data) ** (-1/5))
    
    # Create kernel density estimation object
    kde = gaussian_kde(kde_data, bw_method=bandwidth/np.std(kde_data))
    
    # Generate x values for plotting kernel density curve
    x_range = np.linspace(valid_data.min(), valid_data.max(), 200)
    kde_values = kde(x_range)
    
    # Plot kernel density curve
    ax.plot(x_range, kde_values, 'r-', linewidth=2, label=f'Kernel Density ({kernel})')
    
    # Set figure properties
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add statistical information
    stats_text = f'Sample count: {len(valid_data):,}\n'
    stats_text += f'Mean: {np.mean(valid_data):.4f}\n'
    stats_text += f'Std: {np.std(valid_data):.4f}\n'
    stats_text += f'Min: {np.min(valid_data):.4f}\n'
    stats_text += f'Max: {np.max(valid_data):.4f}\n'
    stats_text += f'Median: {np.median(valid_data):.4f}'
    
    if use_sampling and len(valid_data) > max_samples:
        stats_text += f'\nKDE sample points: {len(kde_data):,}'
    
    stats_text += f'\n(Zero values excluded)'
    
    # Add statistical information text box in upper right corner
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, 
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            fontsize=9)
    
    plt.tight_layout()
    
    # Save or display image
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Distribution plot saved to: {save_path}")
    else:
        plt.show()


def plot_tif_distribution_fast(tif_path: str, save_path = None, title: str = "TIF File Value Distribution", 
                               bins: int = 50, sample_ratio: float = 0.1, max_samples: int = 50000):
    """
    Fast version of TIF file distribution plotting function for ultra-large datasets
    Note: Zero values are excluded from analysis
    
    Args:
        tif_path: TIF file path
        save_path: Save path
        title: Image title
        bins: Number of histogram bins
        sample_ratio: Sampling ratio (between 0-1)
        max_samples: Maximum number of sample points
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import rasterio
    
    try:
        with rasterio.open(tif_path) as src:
            data = src.read(1)
            print(f"Original data shape: {data.shape}")
            
            # Calculate sample size
            total_pixels = data.size
            sample_size = min(int(total_pixels * sample_ratio), max_samples)
            
            print(f"Total pixels: {total_pixels:,}")
            print(f"Sample points: {sample_size:,}")
            
            # Random sampling
            np.random.seed(42)
            flat_data = data.flatten()
            sample_indices = np.random.choice(flat_data.size, size=sample_size, replace=False)
            sampled_data = flat_data[sample_indices]
            
            # Filter invalid values and zeros
            if src.nodata is not None:
                valid_data = sampled_data[~np.isnan(sampled_data) & (sampled_data != src.nodata) & (sampled_data != 0)]
            else:
                valid_data = sampled_data[~np.isnan(sampled_data) & (sampled_data != 0)]
            
            if len(valid_data) == 0:
                raise ValueError("No valid data found")
                
    except Exception as e:
        logger.error(f"Cannot read TIF file {tif_path}: {str(e)}")
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot histogram
    n, bins_edges, patches = ax.hist(valid_data, bins=bins, density=True, alpha=0.7, 
                                     color='skyblue', edgecolor='black', label='Sampled Histogram')
    
    # Calculate and plot kernel density estimation (using fewer points)
    from scipy.stats import gaussian_kde
    
    # Further sampling for KDE
    kde_sample_size = min(10000, len(valid_data))
    kde_data = np.random.choice(valid_data, size=kde_sample_size, replace=False)
    
    kde = gaussian_kde(kde_data)
    x_range = np.linspace(valid_data.min(), valid_data.max(), 100)
    kde_values = kde(x_range)
    
    ax.plot(x_range, kde_values, 'r-', linewidth=2, label='Kernel Density Estimation')
    
    # Set figure properties
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.set_title(f"{title} (Sample Ratio: {sample_ratio:.1%})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add statistical information
    stats_text = f'Total Pixels: {total_pixels:,}\n'
    stats_text += f'Sample Points: {len(valid_data):,}\n'
    stats_text += f'Mean: {np.mean(valid_data):.4f}\n'
    stats_text += f'Std: {np.std(valid_data):.4f}\n'
    stats_text += f'Min: {np.min(valid_data):.4f}\n'
    stats_text += f'Max: {np.max(valid_data):.4f}\n'
    stats_text += f'(Excluded 0 values)'
    
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, 
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Quick distribution plot saved to: {save_path}")
    else:
        plt.show()


def plot_tif_distribution_summary(tif_path: str, save_path = None, title: str = "TIF File Summary"):
    """
    Generate statistical summary plot for TIF files, suitable for ultra-large datasets
    Note: Zero values are excluded from analysis
    
    Args:
        tif_path: TIF file path
        save_path: Save path
        title: Image title
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import rasterio
    
    try:
        with rasterio.open(tif_path) as src:
            data = src.read(1)
            
            # Calculate basic statistics (using numpy optimized methods)
            valid_mask = ~np.isnan(data) & (data != 0)
            if src.nodata is not None:
                valid_mask &= (data != src.nodata)
            
            valid_data = data[valid_mask]
            
            if len(valid_data) == 0:
                raise ValueError("No valid data found")
                
    except Exception as e:
        logger.error(f"Cannot read TIF file {tif_path}: {str(e)}")
        return
    
    # Calculate statistics
    stats = {
        'count': len(valid_data),
        'mean': np.mean(valid_data),
        'std': np.std(valid_data),
        'min': np.min(valid_data),
        'max': np.max(valid_data),
        'median': np.median(valid_data),
        'q25': np.percentile(valid_data, 25),
        'q75': np.percentile(valid_data, 75)
    }
    
    # Create figure
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Box plot
    ax1.boxplot(valid_data, vert=False)
    ax1.set_title('Box Plot')
    ax1.set_xlabel('Value')
    
    # 2. Histogram (using fewer bins)
    ax2.hist(valid_data, bins=30, density=True, alpha=0.7, color='skyblue', edgecolor='black')
    ax2.set_title('Density Histogram')
    ax2.set_xlabel('Value')
    ax2.set_ylabel('Density')
    
    # 3. Statistical information table
    ax3.axis('off')
    table_data = [
        ['Statistic', 'Value'],
        ['Sample Count', f"{stats['count']:,}"],
        ['Mean', f"{stats['mean']:.4f}"],
        ['Std', f"{stats['std']:.4f}"],
        ['Min', f"{stats['min']:.4f}"],
        ['Max', f"{stats['max']:.4f}"],
        ['Median', f"{stats['median']:.4f}"],
        ['Q25', f"{stats['q25']:.4f}"],
        ['Q75', f"{stats['q75']:.4f}"],
        ['', ''],
        ['', '(Excluded 0 values)']
    ]
    
    table = ax3.table(cellText=table_data[1:], colLabels=table_data[0], 
                      cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    ax3.set_title('Statistical Summary')
    
    # 4. Cumulative distribution function
    sorted_data = np.sort(valid_data)
    cumulative = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    ax4.plot(sorted_data, cumulative, 'b-', linewidth=2)
    ax4.set_title('Cumulative Distribution Function')
    ax4.set_xlabel('Value')
    ax4.set_ylabel('Cumulative Probability')
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Statistical summary plot saved to: {save_path}")
    else:
        plt.show()

def plot_all_factors_distribution():
    """
    使用optimized方法画出所有factors的分布情况
    使用年份+因子序号来命名绘出的图像
    """
    import os
    import glob
    
    # 基础路径
    base_path = "data/factors"
    output_path = "visual/factors_distribution"
    
    # 确保输出目录存在
    os.makedirs(output_path, exist_ok=True)
    
    # 获取所有年份目录
    year_dirs = glob.glob(os.path.join(base_path, "*"))
    year_dirs = [d for d in year_dirs if os.path.isdir(d)]
    
    print(f"找到年份目录: {[os.path.basename(d) for d in year_dirs]}")
    
    total_files = 0
    processed_files = 0
    
    for year_dir in year_dirs:
        year = os.path.basename(year_dir)
        print(f"\n处理年份: {year}")
        
        # 获取该年份下的所有tif文件
        tif_files = glob.glob(os.path.join(year_dir, "*.tif"))
        tif_files.sort()  # 按文件名排序
        
        print(f"找到 {len(tif_files)} 个因子文件")
        
        for tif_file in tif_files:
            # 提取因子序号（文件名去掉.tif后缀）
            factor_name = os.path.splitext(os.path.basename(tif_file))[0]
            
            # 生成输出文件名：年份_因子序号.png
            output_filename = f"{year}_{factor_name}.png"
            output_filepath = os.path.join(output_path, output_filename)
            
            print(f"处理文件: {os.path.basename(tif_file)} -> {output_filename}")
            
            try:
                # 使用optimized方法绘制分布图
                plot_tif_distribution_optimized(
                    tif_path=tif_file,
                    save_path=output_filepath,
                    title=f"Factor Distribution - {year} {factor_name}",
                    max_samples=50000,
                    use_sampling=True,
                    chunk_size=500000,
                    progress_bar=True
                )
                processed_files += 1
                print(f"✓ 成功处理: {output_filename}")
                
            except Exception as e:
                print(f"✗ 处理失败 {os.path.basename(tif_file)}: {str(e)}")
            
            total_files += 1
    
    print(f"\n{'='*60}")
    print(f"处理完成!")
    print(f"总文件数: {total_files}")
    print(f"成功处理: {processed_files}")
    print(f"失败文件: {total_files - processed_files}")
    print(f"输出目录: {output_path}")
    print(f"{'='*60}")

def normalize_tensor_and_save_params(features):
    """
    使用data/torch_normalization中的DataNormalizer对给定张量进行归一化，并保存归一化参数。

    参数:
        features (torch.Tensor): 需要归一化的张量

    返回:
        normalized_tensor (torch.Tensor): 归一化后的张量
        normalize_params (dict): 归一化参数字典
    """
    normalize_params = {}

    for c in range(features.shape[1]):
        max_value = torch.max(features[:, c])
        min_value = torch.min(features[:, c])

        scale = max_value - min_value
        if scale == 0:
            logger.info(f"scale == 0, skip {c}")
            continue
        else:
            features[:, c] = (features[:, c] - min_value) / scale
            normalize_params[c] = (max_value, min_value, scale)

    return features, normalize_params

def normalize_tensor_with_params(features, normalize_params):
    """
    使用给定的归一化参数对features进行归一化。

    参数:
        features (torch.Tensor): 需要归一化的张量，形状为 [T, C, H, W] 或 [B, T, C, H, W]
        normalize_params (dict): 归一化参数字典，格式为{c: (max, min, scale)}

    返回:
        features_norm (torch.Tensor): 归一化后的张量
    """
    features_norm = features.clone()
    # 判断输入维度
    if features_norm.dim() == 4:
        # [T, C, H, W]
        for c in range(features_norm.shape[1]):
            if c not in normalize_params:
                print(f"通道{c}未归一化，跳过")
                continue
            max_value, min_value, scale = normalize_params[c]
            if scale == 0:
                print(f"scale == 0, skip {c}")
                continue
            features_norm[:, c] = (features_norm[:, c] - min_value) / scale
    elif features_norm.dim() == 5:
        # [B, T, C, H, W]
        for c in range(features_norm.shape[2]):
            if c not in normalize_params:
                print(f"通道{c}未归一化，跳过")
                continue
            max_value, min_value, scale = normalize_params[c]
            if scale == 0:
                print(f"scale == 0, skip {c}")
                continue
            features_norm[:, :, c] = (features_norm[:, :, c] - min_value) / scale
    else:
        raise ValueError("features的维度必须为4或5")
    return features_norm


def denormalize_tensor(features_norm, normalize_params):
    """
    对归一化后的张量进行逆归一化操作。

    参数:
        features_norm (torch.Tensor): 归一化后的张量
        normalize_params (dict): 归一化参数字典，格式为{c: (max, min, scale)}

    返回:
        features (torch.Tensor): 逆归一化后的张量
    """
    features = features_norm.clone()
    for c in range(features.shape[1]):
        if c not in normalize_params:
            print(f"通道{c}未归一化，跳过逆归一化")
            continue
        max_value, min_value, scale = normalize_params[c]
        if scale == 0:
            print(f"scale == 0, skip {c}")
            continue
        features[:, c] = features[:, c] * scale + min_value
    return features


import tifffile
import numpy as np

def process_factors_dir(factors_dir, threshold=1e-5):
    """
    遍历factors_dir下所有子目录，找到每个子目录下的1.1.tif文件，
    将所有绝对值小于等于threshold的像素置为0，并覆盖保存。
    """
    import rasterio

    for root, dirs, files in os.walk(factors_dir):
        for file in files:
            if file == "1.1.tif":
                file_path = os.path.join(root, file)
                print(f"处理: {file_path}")
                with rasterio.open(file_path) as src:
                    data = src.read()
                    mask = np.abs(data) <= threshold
                    modified = False
                    if np.any(mask):
                        data[mask] = 0
                        modified = True
                        with rasterio.open(file_path, 'w', **src.profile) as dst:
                            dst.write(data)
                        print(f"已修改并保存: {file_path}")
                    else:
                        print(f"无需修改: {file_path}")

                # 结果检查
                # 1. 任一像元所有波段相加和1没有显著差异
                # 2. 没有负值
                # data shape: (bands, H, W)
                sum_bands = np.sum(data, axis=0)  # shape: (H, W)
                # 允许的误差
                atol = 1e-4
                # 检查和1的差异
                diff_mask = np.abs(sum_bands - 1) > atol
                if np.any(diff_mask):
                    idx = np.argwhere(diff_mask)
                    print(f"警告: 文件{file_path}存在{idx.shape[0]}个像元所有波段和与1显著不符，最大偏差: {np.max(np.abs(sum_bands-1))}")
                else:
                    print(f"检查通过: 所有像元波段和≈1")

                # 检查负值
                neg_mask = data < 0
                if np.any(neg_mask):
                    idx = np.argwhere(neg_mask)
                    print(f"警告: 文件{file_path}存在{idx.shape[0]}个负值像元，最小值: {data.min()}")
                else:
                    print(f"检查通过: 无负值")

# INSERT_YOUR_CODE
import os
import shutil

def reset_data():
    """
    实现如下功能：
    1. 将/root/autodl-fs/raw/中每个子文件夹下的tif文件（只有一个）复制到/root/autodl-fs/factors的对应子文件夹下，命名为1.1.tif。如果已存在1.1.tif，先删除。
    2. 同样的文件复制到/root/autodl-fs/label_floats的对应子文件夹下，但年份减5（如2005->2000，2010->2005）。
    """
    raw_root = "/root/autodl-fs/onehot"
    factors_root = "/root/autodl-fs/factors"
    label_floats_root = "/root/autodl-fs/label_floats"

    for subdir in os.listdir(raw_root):
        raw_subdir = os.path.join(raw_root, subdir)
        if not os.path.isdir(raw_subdir):
            continue
        # 查找tif文件
        tif_files = [f for f in os.listdir(raw_subdir) if f.lower().endswith('.tif')]
        if len(tif_files) == 0:
            print(f"子文件夹{subdir}下没有tif文件，跳过")
            continue
        if len(tif_files) > 1:
            print(f"警告：子文件夹{subdir}下有多个tif文件，仅处理第一个")
        tif_file = tif_files[0]
        src_tif_path = os.path.join(raw_subdir, tif_file)

        # 1. 复制到factors
        factors_subdir = os.path.join(factors_root, subdir)
        os.makedirs(factors_subdir, exist_ok=True)
        dst_factors_tif = os.path.join(factors_subdir, "1.1.tif")
        if os.path.exists(dst_factors_tif):
            os.remove(dst_factors_tif)
        shutil.copy2(src_tif_path, dst_factors_tif)
        print(f"复制到: {dst_factors_tif}")

        # 2. 复制到label_floats，年份减5
        try:
            year = int(subdir)
        except Exception as e:
            print(f"子文件夹名{subdir}不是年份，跳过")
            continue
        target_year = year - 5
        label_subdir = os.path.join(label_floats_root, str(target_year))
        os.makedirs(label_subdir, exist_ok=True)
        dst_label_tif = os.path.join(label_subdir, "1.1.tif")
        if os.path.exists(dst_label_tif):
            os.remove(dst_label_tif)
        shutil.copy2(src_tif_path, dst_label_tif)
        print(f"复制到: {dst_label_tif}")


# 测试visualize_temporal_comparison函数
def test_visualize_temporal_comparison():
    # 随机生成两个张量，形状为 (T, H, W)，类别为0-5，部分位置为6表示无效值
    T, H, W = 4, 64, 64
    num_classes = 6
    # 生成真实标签
    real_tensor = torch.randint(0, num_classes, (T, H, W))
    # 随机将部分像素设为无效值（6）
    mask = torch.rand(T, H, W) < 0.1
    real_tensor[mask] = 6

    # 生成伪造标签
    fake_tensor = torch.randint(0, num_classes, (T, H, W))
    mask2 = torch.rand(T, H, W) < 0.1
    fake_tensor[mask2] = 6

    # 保存路径
    save_path = "test_temporal_comparison.png"
    visualize_temporal_comparison(real_tensor, fake_tensor, save_path, title1="Real", title2="Generated")
    print(f"测试图片已保存到: {save_path}")

def load_mask_from_tif(mask_tif_path):
    import os
    import rasterio
    import torch

    if not os.path.exists(mask_tif_path):
        raise FileNotFoundError(f"掩膜tif文件不存在: {mask_tif_path}")
    with rasterio.open(mask_tif_path) as src:
        mask_np = src.read()  # (C, H, W) 或 (1, H, W)
    mask = torch.from_numpy(mask_np)
    # 如果是(1, H, W)，去掉第0维
    if mask.shape[0] == 1:
        mask = mask[0]
    return mask

def load_mask(history_mask_tif, longterm_mask_tif, num_history, num_longterm):
    # 加载掩膜
    history_mask = load_mask_from_tif(history_mask_tif)  # (C, H, W) 或 (H, W)
    longterm_mask = load_mask_from_tif(longterm_mask_tif)  # (C, H, W) 或 (H, W)

    # 保证掩膜形状为 (C, H, W)
    if history_mask.dim() == 2:
        # 单通道，扩展为 (1, H, W)
        history_mask = history_mask.unsqueeze(0)
    if longterm_mask.dim() == 2:
        longterm_mask = longterm_mask.unsqueeze(0)

    # 构造历史掩膜和长线掩膜的时序堆叠
    mask_seqs = []
    if num_history > 0:
        history_mask_seq = history_mask.unsqueeze(0).repeat(num_history, 1, 1, 1)  # (num_history, C, H, W)
        mask_seqs.append(history_mask_seq)
    if num_longterm > 0:
        longterm_mask_seq = longterm_mask.unsqueeze(0).repeat(num_longterm, 1, 1, 1)  # (num_longterm, C, H, W)
        mask_seqs.append(longterm_mask_seq)
    if len(mask_seqs) == 0:
        raise ValueError("num_history和num_longterm不能同时为0")
    elif len(mask_seqs) == 1:
        combined_mask = mask_seqs[0]
    else:
        combined_mask = torch.cat(mask_seqs, dim=0)  # (T, C, H, W)
    return combined_mask

def split_image_into_patches(image, patch_size=256, overlap=64):
    """
    将图像分割成重叠的patch，严格不允许patch超出边界，始终记录左上角坐标。
    当最后一行/列不足一个patch时，从边界反向查找，保证patch全部在图像内。
    Args:
        image: 输入图像 [T, C, H, W]
        patch_size: patch大小
        overlap: 重叠像素数
    Returns:
        patches: patch列表
        positions: 每个patch左上角在原图中的位置 [(start_h, start_w), ...]
    """
    T, C, H, W = image.shape
    stride = patch_size - overlap

    patches = []
    positions = []

    # 计算所有合法的左上角坐标（不允许patch超出边界）
    h_starts = list(range(0, H - patch_size + 1, stride))
    w_starts = list(range(0, W - patch_size + 1, stride))

    # 边界反向查找，确保最后一行/列也能覆盖到
    if len(h_starts) == 0 or h_starts[-1] != H - patch_size:
        if H - patch_size >= 0:
            h_starts.append(H - patch_size)
    if len(w_starts) == 0 or w_starts[-1] != W - patch_size:
        if W - patch_size >= 0:
            w_starts.append(W - patch_size)

    # 去重
    h_starts = sorted(set(h_starts))
    w_starts = sorted(set(w_starts))

    for h in h_starts:
        for w in w_starts:
            patch = image[:, :, h:h+patch_size, w:w+patch_size]
            # patch大小始终为patch_size
            if patch.shape[-2] != patch_size or patch.shape[-1] != patch_size:
                # 理论上不会出现，但保险起见
                pad_h = patch_size - patch.shape[-2]
                pad_w = patch_size - patch.shape[-1]
                patch = F.pad(patch, (0, pad_w, 0, pad_h), mode='reflect')
                # 如果有第7个通道（下标为6），将填充区域设为1
                if patch.shape[1] > 6 and (pad_h > 0 or pad_w > 0):
                    if pad_h > 0:
                        patch[:, 6, -pad_h:, :] = 1
                    if pad_w > 0:
                        patch[:, 6, :, -pad_w:] = 1
            patches.append(patch)
            positions.append((h, w))

    if len(patches) > 0:
        patches = torch.stack(patches, dim=0)
    else:
        patches = torch.empty(0)

    if len(positions) > 0:
        positions = torch.tensor(positions, dtype=torch.long)
    else:
        positions = torch.empty((0, 2), dtype=torch.long)

    return patches, positions

def reconstruct_image_from_patches(patches, positions, original_shape, patch_size=256, overlap=64):
    """
    从patches重建原始图像，对重叠区域使用平均值。
    叠加时严格按照patch的左上角坐标进行，避免重影和位移。
    Args:
        patches: patch张量 [N, T, C, patch_size, patch_size]
        positions: 位置张量 [N, 2]，每行为(h, w)
        original_shape: 原始图像形状 [T, C, H, W]
        patch_size: patch大小
        overlap: 重叠像素数
    Returns:
        重建的图像
    """
    T, C, H, W = original_shape
    device = patches.device if isinstance(patches, torch.Tensor) else patches[0].device

    output_image = torch.zeros(original_shape, device=device)
    weight_image = torch.zeros(original_shape, device=device)

    for idx in range(patches.shape[0]):
        patch = patches[idx]
        start_h, start_w = positions[idx].tolist()
        end_h = start_h + patch_size
        end_w = start_w + patch_size

        # 计算实际可用区域（防止patch超出边界，理论上不会发生）
        actual_h = min(end_h, H) - start_h
        actual_w = min(end_w, W) - start_w

        # 创建权重mask（中心高，边缘低）
        weight_mask = torch.ones_like(patch)
        if overlap > 0:
            for i in range(overlap):
                weight_mask[:, :, i, :] *= (i + 1) / overlap  # 上
                weight_mask[:, :, -(i+1), :] *= (i + 1) / overlap  # 下
                weight_mask[:, :, :, i] *= (i + 1) / overlap  # 左
                weight_mask[:, :, :, -(i+1)] *= (i + 1) / overlap  # 右

        # 只取实际区域
        actual_patch = patch[:, :, :actual_h, :actual_w]
        actual_weight = weight_mask[:, :, :actual_h, :actual_w]

        output_image[:, :, start_h:start_h+actual_h, start_w:start_w+actual_w] += actual_patch * actual_weight
        weight_image[:, :, start_h:start_h+actual_h, start_w:start_w+actual_w] += actual_weight

    # 避免除零
    weight_image = torch.where(weight_image > 0, weight_image, torch.ones_like(weight_image))
    reconstructed_image = output_image / weight_image

    return reconstructed_image

def create_positions_sequence(big_img, patch_size, overlap, batch_size):

    _, positions = split_image_into_patches(big_img, patch_size, overlap)
    position_sequences = []

    for i in range(batch_size):
        positions_clone = positions.clone()
        # 使用torch.randperm来正确打乱张量
        indices = torch.randperm(len(positions_clone))
        positions_clone = positions_clone[indices]

        position_sequences.append(positions_clone)

    position_sequences = torch.cat(position_sequences, dim=0)

    return position_sequences, len(positions) # 返回positions序列和positions数量

if __name__ == "__main__":
    # 历史掩膜tif路径（2020）
    history_mask_tif = "/root/autodl-fs/mask/2020/2020.tif"
    # 长线掩膜tif路径（2035）
    longterm_mask_tif = "/root/autodl-fs/mask/2035/2035.tif"
    combined_mask = load_mask(history_mask_tif, longterm_mask_tif, 4, 3)
    print(combined_mask.shape)


