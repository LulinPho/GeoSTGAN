import os
import shutil
import logging
import torch
import rasterio
import numpy as np

# 配置日志记录器
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def flatten_directory(directory: str) -> None:
    """将目录下所有子文件夹中的文件移动到顶层目录，并删除子文件夹
    
    Args:
        directory: 要处理的目录路径
    """
    if not os.path.exists(directory):
        logger.error(f"目录不存在: {directory}")
        return
        
    try:
        # 遍历目录树
        for root, dirs, files in os.walk(directory, topdown=False):
            # 跳过顶层目录
            if root == directory:
                continue
                
            # 移动文件到顶层目录
            for file in files:
                src_path = os.path.join(root, file)
                dst_path = os.path.join(directory, file)
                
                # 如果目标路径已存在文件，则重命名
                if os.path.exists(dst_path):
                    base, ext = os.path.splitext(file)
                    counter = 1
                    while os.path.exists(dst_path):
                        new_name = f"{base}_{counter}{ext}"
                        dst_path = os.path.join(directory, new_name)
                        counter += 1
                
                shutil.move(src_path, dst_path)
                logger.info(f"移动文件: {src_path} -> {dst_path}")
            
            # 删除空目录
            try:
                os.rmdir(root)
                logger.info(f"删除目录: {root}")
            except OSError as e:
                logger.error(f"删除目录失败 {root}: {str(e)}")
                
        logger.info("目录扁平化完成")
        
    except Exception as e:
        logger.error(f"处理过程中出错: {str(e)}")

def get_tensor_dimensions(directory: str, reference_shape: tuple[int, ...] | None = None) -> tuple[int, ...]:
    """
    从目录结构确定张量的维度，支持tif文件多波段，叶子层级的深度为所有tif波段数之和
    对于每一个子文件夹，检查其中的波段总数是否一致，返回多级深度+空间形状
    Args:
        directory: 数据目录路径
        reference_shape: 参考形状元组，用于确定最后的空间维度，默认为None
    Returns:
        Tuple[int, ...]: 完整的张量维度元组，包含深度维度和形状维度
        如果目录结构不合法，返回空元组
    Example:
        >>> get_tensor_dimensions("data/factors", (256, 256))
        (2, 3, 256, 256)  # 2个年份目录，每个年份都有3个因子文件
    """
    try:
        import rasterio
        def recursive_check(dirs: list[str]) -> list[int]:
            """递归检查每一层的子文件夹数量或波段数，并保证一致性"""
            # 检查当前层是否为叶子层（即包含tif文件）
            bands_list = []
            subdirs_list = []
            for dir_path in dirs:
                items = os.listdir(dir_path)
                subdirs = [os.path.join(dir_path, i) for i in items if os.path.isdir(os.path.join(dir_path, i))]
                tif_files = [f for f in items if f.endswith('.tif')]
                if subdirs:
                    subdirs_list.append(subdirs)
                elif tif_files:
                    # 统计该叶子目录下所有tif文件的波段数之和
                    bands_sum = 0
                    for tif_file in tif_files:
                        tif_path = os.path.join(dir_path, tif_file)
                        try:
                            with rasterio.open(tif_path) as src:
                                bands_sum += src.count
                        except Exception as e:
                            logger.error(f"读取tif文件波段数失败: {tif_path}: {str(e)}")
                            return []
                    bands_list.append(bands_sum)
                else:
                    # 既没有子目录也没有tif文件，非法
                    logger.error(f"目录 {dir_path} 既没有子目录也没有tif文件")
                    return []
            if subdirs_list:
                # 检查所有dir_path的子目录数量是否一致
                subdir_counts = [len(subdirs) for subdirs in subdirs_list]
                if len(set(subdir_counts)) != 1:
                    logger.error(f"同级目录的子目录数量不一致: {subdir_counts}")
                    return []
                # 递归进入下一层
                next_dirs = []
                for subdirs in subdirs_list:
                    next_dirs.extend(subdirs)
                return [subdir_counts[0]] + recursive_check(next_dirs)
            elif bands_list:
                # 检查所有叶子目录的波段数是否一致
                if len(set(bands_list)) != 1:
                    logger.error(f"同级叶子目录的波段总数不一致: {bands_list}")
                    return []
                return [bands_list[0]]
            else:
                logger.error("未能识别的目录结构")
                return []

        # 递归获取深度维度
        depth_dims = recursive_check([directory])
        if not depth_dims:
            return ()

        # 获取空间形状
        if reference_shape is not None:
            shape_dims = reference_shape
        else:
            # 找到第一个tif文件，读取其空间形状
            found = False
            for root, _, files in os.walk(directory):
                tif_files = [f for f in files if f.endswith('.tif')]
                if tif_files:
                    first_file = os.path.join(root, tif_files[0])
                    with rasterio.open(first_file) as src:
                        data = src.read(1)  # 读取第一个波段获取空间形状
                        shape_dims = data.shape
                    found = True
                    break
            if not found:
                logger.error("未找到tif文件")
                return ()
        tensor_dims = tuple(depth_dims) + shape_dims
        logger.info(f"确定张量维度: {tensor_dims}")
        return tensor_dims

    except Exception as e:
        logger.error(f"获取张量维度失败: {str(e)}")
        return ()


def load_hierarchical_tif(directory: str):
    """从目录中读取TIF文件并组织成多维张量，支持多波段数据，并记录波段索引与文件名的对应关系

    Args:
        directory: 数据目录路径

    Returns:
        (torch.Tensor, dict): 
            - 根据目录结构自适应创建的多维张量
            - band_index_map: {tuple(索引): 文件名字符串}
        如果加载失败返回(None, None)
    """
    try:
        # 首先获取第一个tif文件的形状作为参考
        first_tif = None
        for root, _, files in os.walk(directory):
            tif_files = [f for f in files if f.endswith('.tif')]
            if tif_files:
                first_tif = os.path.join(root, tif_files[0])
                break

        if first_tif is None:
            logger.error("未找到任何tif文件")
            return None, None

        with rasterio.open(first_tif) as src:
            reference_shape = src.read(1).shape

        # 获取张量维度,传入参考形状
        tensor_dims = get_tensor_dimensions(directory, reference_shape=reference_shape)

        logger.info(f"张量维度: {tensor_dims}")
        if not tensor_dims:
            logger.error("无法确定张量维度")
            return None, None

        # 创建结果张量 - 支持多波段
        result = torch.zeros(tensor_dims, dtype=torch.float32)
        band_index_map = {}

        # 递归遍历目录并填充数据
        def fill_tensor(current_dir: str, current_tensor: torch.Tensor):
            items = os.listdir(current_dir)
            tif_files = [f for f in items if f.endswith('.tif')]
            subdirs = [d for d in items if os.path.isdir(os.path.join(current_dir, d))]
            subdirs.sort()  # 保证顺序一致

            if tif_files:
                tif_files.sort()
                logger.info(f"正在读取目录: {current_dir}，共 {len(tif_files)} 个文件")
                data_list = []
                band_index = 0
                for idx, tif_file in enumerate(tif_files):
                    tif_path = os.path.join(current_dir, tif_file)

                    try:
                        with rasterio.open(tif_path) as src:
                            data = src.read()  # [bands, H, W]
                            if data.dtype != np.float32:
                                data = data.astype(np.float32)
                            data_list.append(torch.from_numpy(data))
                            # 记录每个波段的索引和文件名
                            for band in range(data.shape[0]):
                                band_index_map[band_index] = tif_path
                                band_index += 1
                            if idx % 10 == 0:
                                logger.info(f"已读取 {idx+1}/{len(tif_files)} 个文件，数据形状: {data.shape}")
                        if np.isnan(data).any():
                            logger.warning(f"文件 {tif_path} 中存在NaN值")
                    except Exception as e:
                        logger.error(f"无法读取文件 {tif_path}: {str(e)}")
                if data_list:
                    cat_tensor = torch.cat(data_list, dim=0 if len(data_list[0].shape) == 3 else -1)
                    assert torch.allclose(cat_tensor[0], data_list[0][0]), "cat_tensor的第0个切片与data_list第0个张量的第0个切片不一致"
                    # 检查cat_tensor和current_tensor形状是否一致
                    if current_tensor.shape == cat_tensor.shape:
                        current_tensor[:] = cat_tensor
                    else:
                        logger.warning(f"cat后张量形状与目标张量不一致: {cat_tensor.shape} vs {current_tensor.shape}，尝试自动适配")
                        # 只要空间和波段一致即可
                        if current_tensor.shape[-2:] == cat_tensor.shape[-2:] and current_tensor.shape[0] == cat_tensor.shape[0]:
                            current_tensor[:] = cat_tensor
                        else:
                            logger.error(f"cat后张量形状无法适配: {current_tensor.shape} vs {cat_tensor.shape}")
                return

            # 递归处理子目录
            if len(current_tensor.shape) < 1 or len(subdirs) != current_tensor.shape[0]:
                logger.error(f"子目录数量与张量维度不匹配: {len(subdirs)} vs {current_tensor.shape}")
                return
            for idx, subdir in enumerate(subdirs):
                next_dir = os.path.join(current_dir, subdir)
                fill_tensor(next_dir, current_tensor[idx])

        # 开始填充数据
        fill_tensor(directory, result)

        logger.info(f"成功创建{len(tensor_dims)}维张量，形状: {result.shape}")
        return result, band_index_map

    except Exception as e:
        logger.error(f"处理过程中出错: {str(e)}")
        return None, None

if __name__ == "__main__":
    factors_data = load_hierarchical_tif("./data/factors")
    labels_data = load_hierarchical_tif("./data/labels")
    
    if factors_data is None or labels_data is None:
        print("数据加载失败")
        exit(1)
        
    print(factors_data.shape)
    print(labels_data.shape)

    labels_data = labels_data.unsqueeze(1)
    print(torch.cat([factors_data, labels_data], dim=1).shape)