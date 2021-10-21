import torch.nn as nn


class AutoRec(nn.Module):
    def __init__(self, args, num_items):
        super(AutoRec, self).__init__()
        # 模型基本参数
        self.args = args
        self.num_items = num_items
        self.hidden_dim = self.args['hidden_dim']

        # 编码器
        self.encoder = nn.Sequential()
        self.L_encode = nn.Linear(self.num_items, self.hidden_dim)
        self.dropout = nn.Dropout(self.args['dropout'])
        self.encoder.add_module('L_encoder', self.L_encode)
        self.encoder.add_module('Relu', nn.ReLU())
        self.encoder.add_module('Dropout', self.dropout)

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

