import torch
import torch.nn as nn

class SoftWeightedLoss(nn.Module):
    """
    实现动态加权平均（Dynamic Weighted Average, DWA）损失加权。
    适用于多任务损失的加权。
    """
    def __init__(self, num_tasks, T=2.0):
        super(SoftWeightedLoss, self).__init__()
        self.num_tasks = num_tasks
        self.T = T
        # 初始化前两步的损失为1，避免除零
        self.register_buffer('loss_history', torch.ones(2, num_tasks))

    def forward(self, losses):
        """
        losses: shape (num_tasks,) 或 (batch_size, num_tasks)
        """
        if losses.dim() == 1:
            losses = losses.unsqueeze(0)  # (1, num_tasks)
        current_loss = losses.mean(dim=0)  # (num_tasks,)

        # 更新损失历史
        self.loss_history = torch.cat([self.loss_history[1:].detach(), current_loss.unsqueeze(0)], dim=0)

        # 计算DWA权重
        w = torch.ones(self.num_tasks, device=losses.device)
        if self.loss_history.shape[0] >= 2:
            prev = self.loss_history[0]
            last = self.loss_history[1]
            # 避免除零
            ratio = last / (prev + 1e-8)
            exp_ratio = torch.exp(ratio / self.T)
            w = self.num_tasks * exp_ratio / exp_ratio.sum()


        # 归一化权重
        w = w / w.sum()

        # 加权损失
        weighted_loss = (w * current_loss).sum()
        return weighted_loss
