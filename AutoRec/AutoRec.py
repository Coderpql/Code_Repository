import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error


class AutoRec(nn.Module):
    def __init__(self, args, num_users, num_items):
        super(AutoRec, self).__init__()
        # 模型基本参数
        self.args = args
        self.num_users = num_users
        self.num_items = num_items
        self.hidden_dim = self.args.hidden_dim

        # 编码器
        self.encoder = nn.Sequential()
        self.L_encode = nn.Linear(self.num_items, self.hidden_dim)
        self.encoder.add_module('L_encoder', self.L_encode)
        self.encoder.add_module('Relu', nn.ReLU())

        # 解码器
        self.decoder = nn.Sequential()
        self.L_decode = nn.Linear(self.hidden_dim, self.num_items)
        self.decoder.add_module('L_decoder', self.L_decode)
        self.decoder.add_module('Relu', nn.ReLU())

    def forward(self, x):
        y_encoder = self.encoder(x)
        y_decoder = self.decoder(y_encoder)

        # 编码器、解码器权重和偏置
        w_encoder = self.encoder.state_dict()['L_encoder.weight']
        w_decoder = self.decoder.state_dict()['L_decoder.weight']
        b_encoder = self.encoder.state_dict()['L_encoder.bias']
        b_decoder = self.decoder.state_dict()['L_decoder.bias']

        return y_encoder, y_decoder, w_encoder, w_decoder, b_encoder, b_decoder


# 损失函数
class AutoRecLoss(nn.Module):
    def __init__(self, args):
        super(AutoRecLoss, self).__init__()
        self.args = args
        self.lambd = self.args.lambd

    def forward(self, model, y_true, y_predict, mask):

        # 计算正则项
        l2_reg = 0
        for param in model.parameters():
            l2_reg = l2_reg + param.norm(2)
        loss = torch.norm((y_true - y_predict) * mask).pow(2) / mask.sum()
        loss = loss + (self.lambd / 2) * l2_reg

        # mae = mean_absolute_error(y_true.cpu().data * mask, y_predict.cpu().data * mask)
        if mask.numpy().sum() != 0:
            mae = (np.abs(y_true.cpu().data - y_predict.cpu().data) * mask.numpy()).sum() / mask.numpy().sum()
            mae = mae.cpu().data.numpy()
        else:
            mae = 0.0

        return loss, mae
