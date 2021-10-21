import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd.gradcheck import gradcheck
from model.AutoRec import AutoRec


# GRL
class GRL(Function):

    @staticmethod
    def forward(ctx, x, mu):
        ctx.mu = mu
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.neg() * ctx.mu

        return grad_input, None


# 梯度检验
# grl = GRL.apply
# input = torch.randn((20, 20), dtype=torch.double, requires_grad=True)
# test = gradcheck(grl, (input, 0.6), eps=1e-6, atol=1e-4)


class U_DARec(nn.Module):
    def __init__(self, args, num_item_T, num_item_S):
        super(U_DARec, self).__init__()
        self.args = args
        # 评分模式提取器相关参数
        self.RPE_input_dim = self.args['DARec']['RPE']['input_dim']
        self.RPE_output_dim = self.args['DARec']['RPE']['output_dim']

        # 目标域评分预测模块参数
        self.RP_T_input_dim = self.args['DARec']['RP_T']['input_dim']
        self.RP_T_hidden_dim_1 = self.args['DARec']['RP_T']['hidden_dim'][0]
        self.RP_T_hidden_dim_2 = self.args['DARec']['RP_T']['hidden_dim'][1]
        self.RP_T_output_dim = num_item_T

        # 辅助域评分预测模块参数
        self.RP_S_input_dim = self.args['DARec']['RP_S']['input_dim']
        self.RP_S_hidden_dim_1 = self.args['DARec']['RP_S']['hidden_dim'][0]
        self.RP_S_hidden_dim_2 = self.args['DARec']['RP_S']['hidden_dim'][1]
        self.RP_S_output_dim = num_item_S

        # 域分类器参数
        self.DC_input_dim = self.args['DARec']['DC']['input_dim']
        self.DC_hidden_dim = self.args['DARec']['DC']['hidden_dim']
        self.DC_output_dim = self.args['DARec']['DC']['output_dim']

        # AutoRec参数
        self.args_AutoRec = self.args['AutoRec']

        # GRL参数
        self.mu = self.args['DARec']['parameter']['mu']

        # 评分模式提取器
        self.RPE = nn.Sequential(
            nn.Linear(in_features=self.RPE_input_dim, out_features=self.RPE_output_dim),
            nn.ReLU()
        )

        # 目标域评分预测模块
        self.RP_T = nn.Sequential(
            nn.Linear(in_features=self.RP_T_input_dim, out_features=self.RP_T_hidden_dim_1),
            nn.ReLU(),
            nn.Linear(in_features=self.RP_T_hidden_dim_1, out_features=self.RP_T_hidden_dim_2),
            nn.ReLU(),
            nn.Linear(in_features=self.RP_T_hidden_dim_2, out_features=self.RP_T_output_dim)
        )

        # 辅助域评分预测模块
        self.RP_S = nn.Sequential(
            nn.Linear(in_features=self.RP_S_input_dim, out_features=self.RP_S_hidden_dim_1),
            nn.ReLU(),
            nn.Linear(in_features=self.RP_S_hidden_dim_1, out_features=self.RP_S_hidden_dim_2),
            nn.ReLU(),
            nn.Linear(in_features=self.RP_S_hidden_dim_2, out_features=self.RP_S_output_dim)
        )

        # 域分类器
        self.DC = nn.Sequential(
            nn.Linear(in_features=self.DC_input_dim, out_features=self.DC_hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=self.DC_hidden_dim, out_features=self.DC_output_dim),
            nn.Sigmoid()
        )

        # 目标域AutoRec
        self.T_AutoRec = AutoRec(self.args_AutoRec, num_item_T)
        # 冻结相关参数
        for para in self.T_AutoRec.parameters():
            para.requires_grad = False
        # 辅助域AutoRec
        self.S_AutoRec = AutoRec(self.args_AutoRec, num_item_S)
        # 冻结相关参数
        for para in self.S_AutoRec.parameters():
            para.requires_grad = False

    def forward(self, x, is_target=True):
        if is_target:
            # 提取用户特征
            f = self.T_AutoRec(x)[0]
        else:
            f = self.S_AutoRec(x)[0]

        # 评分模式提取
        f_RPE = self.RPE(f)

        # 分数预测
        y_T = self.RP_T(f_RPE)
        y_S = self.RP_S(f_RPE)

        # GRL层处理
        f_GRL = GRL.apply(f_RPE, self.mu)

        # 域分类
        c = self.DC(f_GRL)

        return y_T, y_S, c
