import torch.nn as nn
import torch


# 可见评分损失函数
class ObservedMSELoss(nn.Module):
    def __init__(self, reduction='sum'):
        super(ObservedMSELoss, self).__init__()
        self.reduction = reduction
        self.mse = nn.MSELoss(reduction='sum')

    def forward(self, y_true, y_predict):
        # 获取可见评分mask
        mask = (y_true != 0)
        if self.reduction == 'sum':
            loss = self.mse(y_true, y_predict * mask)
        else:
            loss = self.mse(y_true, y_predict * mask) / torch.sum(mask)
        return loss, mask


# U-AutoRec损失函数
class UDARecLoss(nn.Module):
    def __init__(self, args, reduction='sum'):
        super(UDARecLoss, self).__init__()
        self.args = args
        self.reduction = reduction
        # 参数设置
        self.beta = args['DARec']['parameter']['beta']
        self.mu = args['DARec']['parameter']['mu']

        # 设置损失函数
        self.observedLoss = ObservedMSELoss(reduction=self.reduction)
        self.crossEntropyLoss = nn.CrossEntropyLoss(reduction=self.reduction)

    def forward(self, y_true_T, y_true_S, y_predict_T, y_predict_S, c, label):
        # 评分预测损失
        loss_RP_T, mask_T = self.observedLoss(y_true_T, y_predict_T)
        loss_RP_S, mask_S = self.observedLoss(y_true_S, y_predict_S)
        # 域分类损失
        loss_DC = self.crossEntropyLoss(c, label)

        loss = loss_RP_T + self.beta * loss_RP_S + self.mu * loss_DC

        return loss, loss_RP_T, mask_T, loss_RP_S, loss_DC, mask_S


