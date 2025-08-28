import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss
import logging

from .gru import GRUWithGAUNetPP

from .utils import adaptive_cube_split, StableDiscriminatorBlock

logger = logging.getLogger(__name__)

class Generator(nn.Module):
    def __init__(self, hidden_channels=32, num_classes=10, feature_decoder=True, features_channels=128, dropout=0.1):
        super().__init__()
        self.gru = GRUWithGAUNetPP(hidden_channels=hidden_channels, num_classes=num_classes, feature_decoder=feature_decoder, features_channels=features_channels, dropout=dropout)

    def forward(self, x):
        return self.gru(x) # [B, T, N, H, W]
    
class Discriminator(nn.Module):
    def __init__(self, hidden_channels=64, block_num=256):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.block_num = block_num
        self.conv_block = None

        # PatchGAN结构保持不变
        self.descriminator_blocks = nn.Sequential(
            StableDiscriminatorBlock(hidden_channels, hidden_channels, stride=2),
            StableDiscriminatorBlock(hidden_channels, hidden_channels * 2),
            StableDiscriminatorBlock(hidden_channels * 2, hidden_channels * 2, stride=2),
            StableDiscriminatorBlock(hidden_channels * 2, hidden_channels * 4),
            StableDiscriminatorBlock(hidden_channels * 4, hidden_channels * 4, stride=2),
        )
        # 预定义分类器模板
        self.classifier_template = None
        self.classifier = None


    def _get_conv_block(self, input_channels, output_channels, kernel_size, stride=1, padding=0):
        return nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.InstanceNorm2d(output_channels),
            nn.LeakyReLU(0.2, inplace=False),
        )

    def _get_classifier(self, input_shape, device):
        """获取或创建分类器"""
        if self.classifier is None or self.classifier_template != input_shape:
            b, c, h, w = input_shape
            self.classifier_template = input_shape
            self.classifier = nn.Sequential(
                nn.Conv2d(c, self.hidden_channels * 8, kernel_size=(h, w)),
                nn.LeakyReLU(0.2, inplace=False),
                nn.Conv2d(self.hidden_channels * 8, 1, kernel_size=(1, 1))
            ).to(device)
        return self.classifier

    def forward(self, x, y):
        """
        x: [B, T, C, H, W]  条件输入
        y: [B, N, H, W]     目标输入（真实或生成）
        融合x的时间维度为通道
        """
        b, t, c, h, w = x.shape
        by, n, hy, wy = y.shape
        assert b == by and h == hy and w == wy, "x和y的batch、空间尺寸需一致"

        # 将x的时间维度融合到通道： [B, T, C, H, W] -> [B, T*C, H, W]
        x_merged = x.permute(0,2,1,3,4).contiguous().view(b, t*c, h, w)  # [B, T*C, H, W]
        # 拼接y到通道
        x_cat = torch.cat([x_merged, y], dim=1)  # [B, T*C+N, H, W]

        if self.conv_block is None:
            self.conv_block = self._get_conv_block(t*c+n, self.hidden_channels, kernel_size=1).to(x.device).to(x.dtype)

        # 分块
        x_split, n_splits, _ = adaptive_cube_split(x_cat, split_dims=[2, 3], target_num=self.block_num, overlap=0.2)
        x_split_0 = x_split[0, :, :, :, :]
        x_split_0 = self.conv_block(x_split_0)
        x_split_0 = self.descriminator_blocks(x_split_0)


        classifier = self._get_classifier(x_split_0.shape, x_cat.device)
        assert classifier is not None
        output = torch.zeros((b, x_split.shape[0]), device=x_cat.device)
        for j in range(x_split.shape[0]):
            x_split_j = x_split[j, :, :, :, :]
            x_split_j = self.conv_block(x_split_j)
            x_split_j = self.descriminator_blocks(x_split_j)
            x_split_j = classifier(x_split_j)
            output[:, j] = x_split_j.view(b)
        output = output.view(b, n_splits[0], n_splits[1])
        # 输出只有一个时间维
        outputs = output.unsqueeze(1)  # [B, 1, n1, n2]
        assert outputs.shape == (b, 1, n_splits[0], n_splits[1])
        return outputs, n_splits


def mask_penalty(fake_samples, mask, alpha=0.5, eps=1e-6):
    """
    掩膜约束损失（避免对每个有效类都推向1的结构性偏差）。

    思路：
    - 只要求“有效类的总概率质量”大（接近1），而非每个有效类各自接近1；
    - 同时抑制“无效类的总概率质量”。

    Args:
        fake_samples: 未激活的logits，形状 (b, c, h, w)
        mask: 有效性掩膜，形状 (b, c, h, w)。1表示该像素可被分为该类，0表示不可
        alpha: 无效项权重（0~1）。loss = (1-alpha)*valid_term + alpha*invalid_term
        eps: 数值稳定项

    Returns:
        标量损失
    """
    probs = torch.softmax(fake_samples, dim=1)  # [b, c, h, w]

    # 掩膜集合
    valid_mask = (mask > 0.5).to(probs.dtype)
    invalid_mask = 1.0 - valid_mask

    # 每像元的有效/无效总概率
    sum_valid = (probs * valid_mask).sum(dim=1)      # [b, h, w]
    sum_invalid = (probs * invalid_mask).sum(dim=1)  # [b, h, w]

    # 仅在存在有效类的位置计算有效项
    has_valid = (valid_mask.sum(dim=1) > 0).to(probs.dtype)  # [b, h, w]
    valid_term = -(torch.log(sum_valid + eps) * has_valid).sum() / (has_valid.sum() + eps)

    # 无效项：无效总概率均值
    invalid_term = sum_invalid.mean()

    loss = (1.0 - alpha) * valid_term + alpha * invalid_term
    return loss

def inverse_quantity_constraint(fake_samples, current_path, classes_counts, constraint_matrix, xi=(1.0, 1.0, 1.0)):
    """
    逆向数量约束损失函数。
    支持目标类约束、上限约束、下限约束三种类型，约束信息以(3, N)的张量E输入。
    公式参考：
        L_t = i^T * abs(l_c - e_1)
        L_u = i^T * max(l_c - e_2, 0)
        L_l = i^T * max(e_3 - l_c, 0)
        L_num = (ξ_1 L_t + ξ_2 L_u + ξ_3 L_l) / (H*W)
    其中i^T为单位向量转置，e_1为目标类约束，e_2为上限，e_3为下限。

    Args:
        fake_samples: 生成器输出，形状为 (b, c, h, w)，未激活的logits
        constraint_matrix: 约束提示矩阵E，形状为 (3, N)，
            E[0]为目标类约束(target)，E[1]为上限(upper)，E[2]为下限(lower)
        xi: 损失权重参数 (ξ_1, ξ_2, ξ_3)，默认全为1.0

    Returns:
        loss: 数量约束损失（标量）
    """
    c = constraint_matrix.shape[1]
    device = fake_samples.device

    # 使用softmax概率而不是argmax，保持梯度连续性
    pred_probs = torch.softmax(fake_samples, dim=1)  # [b, c, h, w]
    true_probs = current_path  # 假设current_path已经是one-hot编码
    
    # 计算每个类别的期望数量（保持梯度）
    pred_counts = pred_probs.sum(dim=(0, 2, 3))  # [c] - 每个类别的总概率
    true_counts = true_probs.sum(dim=(0, 2, 3))  # [c] - 每个类别的总概率

    # 计算累计类别数量（预测-真实的变化量，正值为增加，负值为减少）
    accumulated_pred_counts = classes_counts + (pred_counts - true_counts)

    # 约束向量
    e_1 = constraint_matrix[0].to(device)  # 目标类约束
    e_2 = constraint_matrix[1].to(device)  # 上限
    e_3 = constraint_matrix[2].to(device)  # 下限

    # 仅对非零约束项计算损失，零表示无约束
    mask_t = (e_1 != 0).float()
    mask_u = (e_2 != 0).float()
    mask_l = (e_3 != 0).float()

    L_t = torch.sum(mask_t * torch.abs(accumulated_pred_counts - e_1))
    # 上限约束损失 L_u
    L_u = torch.sum(mask_u * torch.clamp(accumulated_pred_counts - e_2, min=0))
    # 下限约束损失 L_l
    L_l = torch.sum(mask_l * torch.clamp(e_3 - accumulated_pred_counts, min=0))

    # 权重
    xi_1, xi_2, xi_3 = xi

    # 总像素数（不含类别维）
    H = fake_samples.shape[2]
    W = fake_samples.shape[3]
    total_pixels = H * W

    # 总损失
    L_num = (xi_1 * L_t + xi_2 * L_u + xi_3 * L_l) / (total_pixels)

    # 汇报累计像元数量和约束目标
    logger.info("类别\t下限\t目标值\t上限")
    for i in range(c):
        lower = e_3[i].item()
        target = e_1[i].item()
        upper = e_2[i].item()
        acc = accumulated_pred_counts[i].item()
        logger.info(f"{i}:\t{lower:.1f} ≤ ({acc:.1f}) ≈ ({target:.1f}) ≤ {upper:.1f}")

    return L_num, accumulated_pred_counts

def efficient_constraint(fake_samples, current_patch, efficient_matrix):
    """
    效应约束损失函数：根据current_patch（当前patch的真实标签）和fake_samples（生成器输出的logits），
    统计类型转换（如A->B），并结合efficient_matrix（效应矩阵）计算总效应损失。

    Args:
        fake_samples: 生成器输出，形状为 (b, c, h, w)，未激活的logits
        current_patch: 当前patch的真实标签，形状为 (b, c, h, w)，独热编码
        efficient_matrix: 效应矩阵，形状为 (num_classes, num_classes)，
            efficient_matrix[i, j] 表示类型i转为j的效应损失

    Returns:
        loss: 效应约束损失（标量）
        transition_counts: 转换计数矩阵，形状为 (num_classes, num_classes)
    """
    # 获取类别数
    num_classes = fake_samples.shape[1]
    device = fake_samples.device

    # 预测类别 [b, h, w]
    pred_class = fake_samples.argmax(dim=1)  # [b, h, w]
    # 真实类别 [b, h, w]
    true_class = current_patch.argmax(dim=1)  # [b, h, w]

    # 统计类型转换矩阵
    transition_counts = torch.zeros((num_classes, num_classes), device=device)

    # 遍历每个batch
    for b in range(pred_class.shape[0]):
        t_cls = true_class[b].flatten()  # [h*w]
        p_cls = pred_class[b].flatten()  # [h*w]
        for i in range(num_classes):
            for j in range(num_classes):
                # 统计从i->j的像素数
                count = ((t_cls == i) & (p_cls == j)).sum()
                transition_counts[i, j] += count

    # 计算效应损失
    # 每个转换(i->j)的损失 = 转换数量 * efficient_matrix[i, j]
    effect_loss = (transition_counts * efficient_matrix.to(device)).sum()
    # 归一化，防止patch大小影响
    total_pixels = pred_class.numel()
    loss = effect_loss / (total_pixels + 1e-8)

    return loss, transition_counts

    


def generator_loss_v0(fake_samples, y, fake_validity, beta=0.5):
    """
    计算生成器的损失函数（用于亚元素级别占比预测），使用MSE + 比例约束损失。
    
    Args:
        fake_samples: 生成样本，形状为 (b, t, c, h, w) - 未经过激活的原始输出
        y: 真实标签，形状为 (b, t, c, h, w) - 土地利用类型独热编码
        fake_validity: 判别器输出，形状为 (b, t, n_splits[0], n_splits[1])
    """
    # 1. 计算focal loss (gamma=2)
    fake_samples_reshape = fake_samples.permute(0, 1, 3, 4, 2).reshape(-1, fake_samples.shape[2])  # [b*t*h*w, c]
    y_reshape = y.permute(0, 1, 3, 4, 2).reshape(-1, y.shape[2])  # [b*t*h*w, c]
    y_label = y_reshape.argmax(dim=1)  # [b*t*h*w]

    # 计算softmax概率
    log_probs = F.log_softmax(fake_samples_reshape, dim=1)  # [N, C]
    probs = torch.exp(log_probs)  # [N, C]
    # 取出每个样本的真实类别概率
    pt = probs[torch.arange(probs.size(0)), y_label]  # [N]
    log_pt = log_probs[torch.arange(log_probs.size(0)), y_label]  # [N]
    gamma = 2.0
    focal_loss = -((1 - pt) ** gamma) * log_pt
    loss_focal = focal_loss.mean()
    
    fake_validity = fake_validity.to(torch.float32)
    loss_bce = BCEWithLogitsLoss()(fake_validity,torch.ones_like(fake_validity))

    # 自适应调整 beta，根据当前的 loss_focal 动态调整 beta，损失越大，beta 越大，给予更高权重
    if beta is None or (isinstance(beta, str) and beta == "auto"):
        beta_min = 0.2
        beta_max = 0.8
        loss_focal_detach = loss_focal.detach()
        norm_focal = torch.sigmoid((loss_focal_detach - 1.0) * 2.0)  # 0~1
        beta = beta_min + (beta_max - beta_min) * norm_focal

    # 3. 总损失
    gen_loss = beta * loss_focal + (1 - beta) * loss_bce
    
    return gen_loss, loss_focal, loss_bce
    

def generator_loss_v1(fake_samples, y, fake_validity, beta=0.5):
    """
    计算生成器的损失函数（用于亚元素级别占比预测），使用MSE + 比例约束损失。
    
    Args:
        fake_samples: 生成样本，形状为 (b, t, c, h, w) - 未经过激活的原始输出
        y: 真实标签，形状为 (b, t, c, h, w) - 亚元素级别占比值
        fake_validity: 判别器输出，形状为 (b, t, n_splits[0], n_splits[1])
    """
    # 重新设计的生成器损失函数（不包含置信度损失，允许混合像元）
    # 1. 生成器输出softmax，得到每个像素的亚元素占比分布
    fake_ratios = torch.softmax(fake_samples, dim=2)  # [b, t, c, h, w]

    y_dist = y/ y.sum(dim=2, keepdim=True)

    # 3. KL散度损失（衡量两个分布的差异，鼓励生成分布接近真实分布）
    kl_loss = (y_dist * ((y_dist + 1e-8).log() - (fake_ratios + 1e-8).log())).sum(dim=2)  # [b, t, h, w]
    kl_loss = kl_loss.mean()  # 对所有样本、时间、空间求均值

    # 4. 比例约束损失（确保每个像素所有通道的和接近1）
    total_ratios = fake_ratios.sum(dim=2)  # [b, t, h, w]
    ratio_constraint_loss = F.mse_loss(total_ratios, torch.ones_like(total_ratios))

    
    fake_validity = fake_validity.to(torch.float32)
    loss_bce = BCEWithLogitsLoss()(fake_validity, torch.ones_like(fake_validity))

    # 如果beta没有给定或者beta为"auto"，则自动调整beta，否则使用传入的beta值
    if beta is None or (isinstance(beta, str) and beta == "auto"):
        # 自适应调整 beta，根据当前的 kl_loss 动态调整 beta
        beta_min = 0.2
        beta_max = 0.8
        kl_loss_detach = kl_loss.detach()
        norm_focal = torch.sigmoid((kl_loss_detach - 1.0) * 2.0)  # 0~1
        beta = beta_min + (beta_max - beta_min) * norm_focal

    # 6. 总损失 = KL散度损失 + 比例约束损失 + 判别器损失
    gen_loss = (
        beta * kl_loss
        + 0.1 * ratio_constraint_loss
        + (1 - beta) * loss_bce
    )

    return gen_loss, kl_loss, loss_bce


def generator_loss_v2(fake_samples, y_float, fake_validity, beta=0.5, gamma=2.0):
    """
    计算生成器的损失函数（用于多通道浮点型特征值预测），使用MSE计算像素损失。
    
    Args:
        fake_samples: 生成样本，形状为 (b, t, c, h, w)
        y_float: 真实标签，形状为 (b, t, c, h, w)
        fake_validity: 生成样本的判别结果，形状为 (b, t, n_splits[0], n_splits[1])
    """
    # 1. 计算MSE损失
    loss_mse = F.mse_loss(fake_samples, y_float, reduction='none')
    loss_mse = loss_mse.mean(dim=[2, 3, 4])  # 对每个样本每个时间步的所有通道空间求均值
    loss_mse = loss_mse.mean()  # 再对batch和时间步求均值

    # 2. 计算Focal MSE损失
    pt = torch.exp(-loss_mse)
    focal_loss = ((1 - pt) ** gamma * loss_mse).mean()

    # 3. 判别器的BCE损失（鼓励生成器骗过判别器）
    fake_validity = fake_validity.to(torch.float32)
    loss_bce = BCEWithLogitsLoss()(fake_validity, torch.ones_like(fake_validity))

    # 如果beta没有给定或者beta为"auto"，则自动调整beta，否则使用传入的beta值
    if beta is None or (isinstance(beta, str) and beta == "auto"):
        # 自适应调整 beta，根据当前的 focal_loss 动态调整 beta
        beta_min = 0.2
        beta_max = 0.8
        focal_loss_detach = focal_loss.detach()
        norm_focal = torch.sigmoid((focal_loss_detach - 1.0) * 2.0)  # 0~1
        beta = beta_min + (beta_max - beta_min) * norm_focal

    # 4. 总损失
    gen_loss = beta * focal_loss + (1 - beta) * loss_bce

    return gen_loss, focal_loss, loss_bce


def generator_loss_v3(fake_samples, y, fake_validity, beta=0.5):
    """
    计算生成器的损失函数（用于亚元素级别占比预测），使用MSE + 比例约束损失。
    
    Args:
        fake_samples: 生成样本，形状为 (b, c, h, w) - 未经过激活的原始输出
        y: 真实标签，形状为 (b, c, h, w) - 土地利用类型独热编码
        fake_validity: 判别器输出，形状为 (b, n_splits[0], n_splits[1])
    """
    # 1. 计算focal loss (gamma=2)
    fake_samples_reshape = fake_samples.permute(0, 2, 3, 1).reshape(-1, fake_samples.shape[1])  # [b*h*w, c]
    y_reshape = y.permute(0, 2, 3, 1).reshape(-1, y.shape[1])  # [b*h*w, c]
    y_label = y_reshape.argmax(dim=1)  # [b*h*w]

    # 计算softmax概率
    log_probs = F.log_softmax(fake_samples_reshape, dim=1)  # [N, C]
    probs = torch.exp(log_probs)  # [N, C]
    # 取出每个样本的真实类别概率
    pt = probs[torch.arange(probs.size(0)), y_label]  # [N]
    log_pt = log_probs[torch.arange(log_probs.size(0)), y_label]  # [N]
    gamma = 2.0
    focal_loss = -((1 - pt) ** gamma) * log_pt
    loss_focal = focal_loss.mean()
    
    fake_validity = fake_validity.to(torch.float32)
    loss_bce = BCEWithLogitsLoss()(fake_validity, torch.full_like(fake_validity, 0.95))

    # 自适应调整 beta，根据当前的 loss_focal 动态调整 beta，损失越大，beta 越大，给予更高权重
    if beta is None or (isinstance(beta, str) and beta == "auto"):
        beta_min = 0.2
        beta_max = 0.8
        loss_focal_detach = loss_focal.detach()
        norm_focal = torch.sigmoid((loss_focal_detach - 1.0) * 2.0)  # 0~1
        beta = beta_min + (beta_max - beta_min) * norm_focal

    # 3. 总损失
    gen_loss = beta * loss_focal + (1 - beta) * loss_bce
    
    return gen_loss, loss_focal, loss_bce

def discriminator_loss(true_validity, fake_validity):
    """
    计算判别器的损失函数
    
    Args:
        real_validity: 真实样本的判别结果，形状为 (b, t, n_splits[0], n_splits[1])
        fake_validity: 生成样本的判别结果，形状为 (b, t, n_splits[0], n_splits[1])
    """
    true_loss = BCEWithLogitsLoss()(true_validity, torch.full_like(true_validity, 0.95))
    fake_loss = BCEWithLogitsLoss()(fake_validity, torch.full_like(fake_validity, 0))

    disc_loss = (true_loss + fake_loss) * 0.5
    
    return disc_loss, true_loss, fake_loss