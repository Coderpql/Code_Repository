import torch
import torch.nn as nn


class CIN(nn.Module):
    def __init__(self, args):
        super(CIN, self).__init__()

        self.input_channels = args.CIN_input_channels
        self.num_layers = args.CIN_num_layers
        self.num_units = args.CIN_num_units
        self.output_dim = args.CIN_output_dim

        # CIN网络
        self.conv_layers = nn.ModuleList()
        for i in range(self.num_layers):
            if i == 0:
                self.conv1d = nn.Conv1d(in_channels=self.input_channels * self.input_channels,
                                        out_channels=self.num_units[i], kernel_size=1)
            else:
                self.conv1d = nn.Conv1d(in_channels=self.num_units[i - 1] * self.input_channels,
                                        out_channels=self.num_units[i], kernel_size=1)
            self.conv_layers.append(self.conv1d)
        # 激活
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        # 输出
        self.output_layer = nn.Linear(sum(self.num_units), self.output_dim)

    def forward(self, x):
        out = None
        x_0, h_old = x.unsqueeze(2), x
        for i in range(self.num_layers):
            h_old = h_old.unsqueeze(1)
            x = x_0 * h_old
            batch_size, x_0_dim, h_old_dim, embed_dim = x.shape
            x = x.view(batch_size, x_0_dim * h_old_dim, embed_dim)  # (20, 25, 5)
            y = self.conv_layers[i](x)  # (20, 100, 5)
            h_old = y
            # sumPooling
            y = y.sum(dim=2)  # (20, 100)
            if i == 0:
                out = y
            else:
                out = torch.cat((out, y), dim=1)
        out = self.relu(out)
        out = self.output_layer(out)
        out = self.sigmoid(out)
        return out


class DNN(nn.Module):
    def __init__(self, args):
        super(DNN, self).__init__()

        self.input_dim = args.DNN_input_dim
        self.num_layers = args.DNN_num_layers
        self.num_units = args.DNN_num_units
        self.output_dim = args.DNN_output_dim

        # 构建PNN网络
        self.dnn_layers = nn.ModuleList()
        for i in range(self.num_layers):
            if i == 0:
                self.dnn = nn.Linear(self.input_dim, self.num_units[i])
            else:
                self.dnn = nn.Linear(self.num_units[i - 1], self.num_units[i])
            self.dnn_layers.append(self.dnn)

        # 激活
        self.sigmoid = nn.Sigmoid()

        # 输出
        self.output_layer = nn.Linear(self.num_units[-1], self.output_dim)

    def forward(self, x):
        for i in range(self.num_layers):
            y = self.dnn_layers[i](x)
            y = self.sigmoid(y)
            x = y
        out = self.output_layer(x)
        out = self.sigmoid(out)
        return out


class LINEAR(nn.Module):

    def __init__(self, args):
        super(LINEAR, self).__init__()

        self.input_dim = args.LINEAR_input_dim
        self.output_dim = args.LINEAR_output_dim

        # 线性计算网络
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, self.output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.net(x)
        return y


class EMBEDDING(nn.Module):
    def __init__(self, args, input_dim):
        super(EMBEDDING, self).__init__()

        self.input_dim = input_dim
        self.embedding_dim = args.EMBEDDING_embedding_dim

        self.embedding = nn.Linear(self.input_dim, self.embedding_dim)

    def forward(self, x):
        y = self.embedding(x)
        return y


class XDeepFM(nn.Module):
    def __init__(self, args):
        super(XDeepFM, self).__init__()

        self.cinNet = CIN(args)
        self.dnnNet = DNN(args)
        self.linearNet = LINEAR(args)
        self.embeddingNet_f1 = EMBEDDING(args, args.EMBEDDING_input_dim_f1)
        self.embeddingNet_f2 = EMBEDDING(args, args.EMBEDDING_input_dim_f2)
        self.embeddingNet_f3 = EMBEDDING(args, args.EMBEDDING_input_dim_f3)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x_f1, x_f2, x_f3):
        # 线性计算部分
        x_linear = torch.cat((x_f1, x_f2, x_f3), dim=1)
        y_linear = self.linearNet(x_linear)

        # embedding
        embedding_f1 = self.embeddingNet_f1(x_f1)
        embedding_f2 = self.embeddingNet_f2(x_f2)
        embedding_f3 = self.embeddingNet_f3(x_f3)

        # DNN
        x_dnn = torch.cat((embedding_f1, embedding_f2, embedding_f3), dim=1)
        y_dnn = self.dnnNet(x_dnn)

        # CIN
        x_cin = torch.cat((embedding_f1.unsqueeze(1), embedding_f2.unsqueeze(1), embedding_f3.unsqueeze(1)), dim=1)
        y_cin = self.cinNet(x_cin)

        y = self.sigmoid(y_linear + y_dnn + y_cin)

        return y
