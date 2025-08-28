import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import PatchAttention
from .utils import conv_block
import logging

logger = logging.getLogger(__name__)
# 配置日志记录器
logger.setLevel(logging.INFO)

# 移除全局 device 变量

class DownBlock(nn.Module):
    def __init__(self, out_channels):
        super().__init__()
        self.out_channels = out_channels
        self.conv = None

    def _init_conv(self, x):
        if self.conv is None:
            self.conv = conv_block(x, self.out_channels, stride=2, padding='same').to(x.device)

    def forward(self, x):
        self._init_conv(x)

        assert self.conv is not None

        x = self.conv(x)

        return x

class UpBlock(nn.Module):
    """UNet++的上采样块"""
    def __init__(self, in_channels, addition_channels, out_channels, up_scale=True):
        super().__init__()
        if up_scale:
            self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        else:
            self.up = None
        self.conv = conv_block(in_channels + addition_channels, out_channels, padding='same')
        self.conv_1 = conv_block(out_channels, out_channels, padding='same')
        
    def forward(self, x1, x2):
        if self.up is not None and x1.shape[2] != x2.shape[2]:
            x1 = self.up(x1)
        # 处理尺寸不匹配
        if x1.size()[2:] != x2.size()[2:]:
            x1 = F.interpolate(x1, size=x2.size()[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        identity = x
        x = self.conv_1(x)
        x = x + identity
        return x

class UnetEncoder(nn.Module):
    """
    类似Unet的编码器，逐级生成1/2, 1/4, 1/8尺度的特征图
    输入: [B, C, H, W]
    输出: [(1/2特征), (1/4特征), (1/8特征)]
    """
    def __init__(self, base_channels=64):
        super().__init__()
        self.down1 = DownBlock(base_channels * 2)     # 1/2 
        self.down2 = DownBlock(base_channels * 4)   # 1/4
        self.down3 = DownBlock(base_channels * 8)   # 1/8
        self.down4 = DownBlock(base_channels * 16)   # 1/16

    def forward(self, x):
        feats = []
        feats.append(x)
        x1 = self.down1(x)   # x1: 1/2, [B, hidden_channels *2, H/2, W/2]
        feats.append(x1)
        x2 = self.down2(x1)   # x2: 1/4, [B, hidden_channels *4, H/4, W/4]
        feats.append(x2)
        x3 = self.down3(x2)   # x3: 1/8, [B, hidden_channels *8, H/8, W/8]
        feats.append(x3)
        x4 = self.down4(x3)   # x4: 1/16, [B, hidden_channels *16, H/16, W/16]
        feats.append(x4)
        return feats  # [1/2, 1/4, 1/8, 1/16]

class UnetPlusPlusDecoder(nn.Module):
    """UNet++解码器，具有密集跳跃连接"""
    def __init__(self, base_channels=64):
        super().__init__()
        
        # 第一层解码器 (从x4到x3)
        self.up1_1 = UpBlock(base_channels * 16, base_channels * 8, base_channels * 8)
        
        # 第二层解码器 (从x3到x2)
        self.up2_1 = UpBlock(base_channels * 8, base_channels * 4, base_channels * 4)
        self.up2_2 = UpBlock(base_channels * 8, base_channels * 8, base_channels * 4)  # 密集连接

        # 第三层解码器 (从x2到x1)
        self.up3_1 = UpBlock(base_channels * 4, base_channels * 2, base_channels * 2)
        self.up3_2 = UpBlock(base_channels * 4, base_channels * 4, base_channels * 2)  # 密集连接
        self.up3_3 = UpBlock(base_channels * 4, base_channels * 6, base_channels * 2)  # 密集连接
        
        # 第四层解码器 (从x1到原始尺寸)
        self.up4_1 = UpBlock(base_channels * 2, base_channels, base_channels)
        self.up4_2 = UpBlock(base_channels * 2, base_channels * 2, base_channels)  # 密集连接
        self.up4_3 = UpBlock(base_channels * 2, base_channels * 3, base_channels)  # 密集连接
        self.up4_4 = UpBlock(base_channels * 2, base_channels * 4, base_channels)  # 密集连接
        
    def forward(self, feats):
        x0, x1, x2, x3, x4 = feats
        
        # 第一层解码
        d3_1 = self.up1_1(x4, x3)
        
        # 第二层解码
        d2_1 = self.up2_1(x3, x2)  # 直接连接
        d2_2 = self.up2_2(d3_1, torch.cat([x2,d2_1], dim=1))
        
        # 第三层解码
        d1_1 = self.up3_1(x2, x1)
        d1_2 = self.up3_2(d2_1, torch.cat([x1,d1_1], dim=1))  # 直接连接
        d1_3 = self.up3_3(d2_2, torch.cat([x1,d1_2,d1_1], dim=1))  # 直接连接
        
        # 第四层解码 (到原始输入尺寸)
        d0_1 = self.up4_1(x1, x0)
        d0_2 = self.up4_2(d1_1, torch.cat([x0,d0_1], dim=1))
        d0_3 = self.up4_3(d1_2, torch.cat([x0,d0_1,d0_2], dim=1))
        d0_4 = self.up4_4(d1_3, torch.cat([x0,d0_1,d0_2,d0_3], dim=1))
        
        # 返回所有深度的输出，用于深度加权
        return [d0_1, d0_2, d0_3, d0_4]

class GAUNetPP(nn.Module):
    def __init__(self, hidden_channels, out_channels, classify=False, multi_scale_features=True):
        super().__init__()
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.multi_scale_features = multi_scale_features
        
        self.patch_attention = PatchAttention(hidden_channels=hidden_channels, out_channels=hidden_channels * 16)
        self.encoder = UnetEncoder(base_channels=hidden_channels)
        self.decoder = UnetPlusPlusDecoder(base_channels=hidden_channels)
        
        # 多尺度特征融合模块
        if multi_scale_features:
            # 特征融合网络，将不同深度的特征融合为统一表示
            self.feature_fusion = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(hidden_channels, out_channels, kernel_size=1, padding=0),
                    nn.InstanceNorm2d(out_channels),
                    nn.LeakyReLU(inplace=False, negative_slope=0.2)
                ) for _ in range(4)
            ])
            
            # 注意力权重学习
            self.attention_weights = nn.Parameter(torch.ones(4) / 4)
            self.softmax = nn.Softmax(dim=0)
        
        self.conv1 = None
        self.conv1_initialized = False

        self.classifier = None
        self.classifier_initialized = False
        self.classify = classify
        
    def _init_conv1(self, x):
        if not self.conv1_initialized:
            self.conv1 = conv_block(x, self.hidden_channels, padding='same').to(x.device)
            self.conv1_initialized = True
        # 每次都同步 device
        if self.conv1 is not None:
            self.conv1 = self.conv1.to(x.device)
            
    def _init_classifier(self, x):
        if not self.classifier_initialized:
            tensor_channels = x.shape[1]
            self.classifier = nn.Sequential(
                nn.Conv2d(tensor_channels, self.out_channels, 1),
                nn.InstanceNorm2d(self.out_channels, affine=True, track_running_stats=True),
                nn.LeakyReLU(inplace=False, negative_slope=0.2)
            ).to(x.device)
            self.classifier_initialized = True
        # 每次都同步 device
        if self.classifier is not None:
            self.classifier = self.classifier.to(x.device)
        
    def forward(self, x):
        x = x.to(x.device)
        b, c, h, w = x.shape
        
        # 初始卷积
        self._init_conv1(x)
        assert self.conv1 is not None
        x = self.conv1(x)
        
        # 编码器
        feats = self.encoder(x)
        
        # 注意力机制应用到最深层特征
        x4 = feats[-1]
        attended, weights = self.patch_attention(x4)
        feats[-1] = attended  # 替换最深层特征
        
        # 解码器
        decoder_outputs = self.decoder(feats)
        
        # 确保所有输出尺寸正确
        outputs = []
        for i, output in enumerate(decoder_outputs):
            if output.size()[2:] != (h, w):
                output = F.interpolate(output, size=(h, w), mode='bilinear', align_corners=False)
            outputs.append(output)
        
        if self.multi_scale_features:
            # 多尺度特征融合
            fused_features = self._fuse_multi_scale_features(outputs)
            
            if self.classify:
                # 如果需要解码，应用分类器
                self._init_classifier(fused_features)
                assert self.classifier is not None
                out = self.classifier(fused_features)
                return out
            else:
                # 作为特征提取器，返回融合后的特征
                return fused_features
        else:
            output = torch.cat(outputs, dim=1)
            if self.classify:
                self._init_classifier(output)
                assert self.classifier is not None
                out = self.classifier(output)
                return out
            else:
                return output
    
    def _fuse_multi_scale_features(self, features):
        """
        融合多尺度特征
        Args:
            features: 列表，包含4个不同深度的特征 [B, C, H, W]
        Returns:
            融合后的特征 [B, C, H, W]
        """
        # 应用特征融合网络
        processed_features = []
        for i, (feature, fusion_net) in enumerate(zip(features, self.feature_fusion)):
            processed_feature = fusion_net(feature)
            processed_features.append(processed_feature)
        
        # 学习注意力权重
        attention_weights = self.softmax(self.attention_weights)
        
        # 加权融合
        fused_feature = torch.zeros_like(processed_features[0])
        for i, (feature, weight) in enumerate(zip(processed_features, attention_weights)):
            fused_feature += weight * feature
        
        return fused_feature
    
    def get_multi_scale_features(self, x):
        """
        获取多尺度特征，用于其他模块使用
        Args:
            x: 输入张量
        Returns:
            dict: 包含不同深度特征的字典
        """
        x = x.to(x.device)
        b, c, h, w = x.shape
        
        # 初始卷积
        self._init_conv1(x)
        x = self.conv1(x)
        
        # 编码器
        feats = self.encoder(x)
        
        # 注意力机制
        x4 = feats[-1]
        attended, weights = self.patch_attention(x4)
        feats[-1] = attended
        
        # 解码器
        decoder_outputs = self.decoder(feats)
        
        # 确保尺寸正确
        outputs = []
        for output in decoder_outputs:
            if output.size()[2:] != (h, w):
                output = F.interpolate(output, size=(h, w), mode='bilinear', align_corners=False)
            outputs.append(output)
        
        return {
            'depth_0': outputs[0],  # 最深的特征
            'depth_1': outputs[1],  # 中等深度特征
            'depth_2': outputs[2],  # 较浅特征
            'depth_3': outputs[3],  # 最浅特征
            'attention_weights': weights  # 注意力权重
        }