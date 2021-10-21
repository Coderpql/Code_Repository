import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import argparse
import torch.utils.data as Data
import torch.optim as optim
from XDeepFMModel import CIN, DNN, LINEAR, EMBEDDING, XDeepFM
import itertools
from sklearn.metrics import mean_squared_error, mean_absolute_error


# 训练
def train(epoch):
    Loss = 0
    MAE = 0
    RMSE = 0
    total_steps = len(train_dataloader)
    for step, (batch_f1, batch_f2, batch_f3, batch_y) in enumerate(train_dataloader):
        batch_f1 = batch_f1.type(torch.FloatTensor).to(device)
        batch_f2 = batch_f2.type(torch.FloatTensor).to(device)
        batch_f3 = batch_f3.type(torch.FloatTensor).to(device)
        batch_y = batch_y.long().to(device)

        y = model_XDeepFM(batch_f1, batch_f2, batch_f3).to(device)

        loss = loss_func(y, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        Loss += loss.cpu().data.numpy()
        y_pred = torch.max(y, 1)[1].cpu().data.numpy()
        mae = mean_absolute_error(batch_y.cpu().data.numpy(), y_pred)
        rmse = np.sqrt(mean_squared_error(batch_y.cpu().data.numpy(), y_pred))
        MAE += mae
        RMSE += rmse
        if step % 10 == 0:
            print('Epoch [{}/{}], Step [{}/{}], loss: {:.4f},mae: {:.4f}, rmse: {:.4f}'
                  .format(epoch + 1, args.train_epoch, step, total_steps, loss, mae, rmse))
    Loss /= total_steps
    MAE /= total_steps
    RMSE /= total_steps
    print('Train-Epoch [{}/{}],loss: {:.4f},mae: {:.4f}, rmse: {:.4f}'
          .format(epoch + 1, args.train_epoch, Loss, MAE, RMSE))
    return Loss, MAE, RMSE


def test(epoch):
    Loss = 0
    MAE = 0
    RMSE = 0
    total_steps = len(test_dataloader)
    with torch.no_grad():
        for step, (batch_f1, batch_f2, batch_f3, batch_y) in enumerate(test_dataloader):
            batch_f1 = batch_f1.type(torch.FloatTensor).to(device)
            batch_f2 = batch_f2.type(torch.FloatTensor).to(device)
            batch_f3 = batch_f3.type(torch.FloatTensor).to(device)
            batch_y = batch_y.long().to(device)

            y = model_XDeepFM(batch_f1, batch_f2, batch_f3).to(device)

            loss = loss_func(y, batch_y)
            Loss += loss.cpu().data.numpy()
            y_pred = torch.max(y, 1)[1].cpu().data.numpy()
            mae = mean_absolute_error(batch_y.cpu().data.numpy(), y_pred)
            rmse = np.sqrt(mean_squared_error(batch_y.cpu().data.numpy(), y_pred))
            MAE += mae
            RMSE += rmse
    Loss /= total_steps
    MAE /= total_steps
    RMSE /= total_steps
    print('Test-Epoch [{}/{}],loss: {:.4f},mae: {:.4f},rmse: {:.4f}'
          .format(epoch + 1, args.train_epoch, Loss, MAE, RMSE))
    return Loss, MAE, RMSE


# 获取数据
def get_data():
    '''
    1.将数据分割为训练集和测试集
    2.将数据按领域（属性）进行划分，拆分为train_y, train_f1, train_f2, train_f3, test_y, test_f1, test_f2, test_f3

    train_y:训练集标签数据
    train_f1:训练集属性1的数据
    train_f2:训练集属性2的数据
    train_f3:训练集属性3的数据
    test_y:测试集标签数据
    test_f1:测试集属性1的数据
    test_f2:测试集属性2的数据
    test_f3:测试集属性3的数据

    Returns:

        train_y:训练集标签数据
        train_f1:训练集属性1的数据
        train_f2:训练集属性2的数据
        train_f3:训练集属性3的数据
        test_y:测试集标签数据
        test_f1:测试集属性1的数据
        test_f2:测试集属性2的数据
        test_f3:测试集属性3的数据

    '''
    # 训练数据
    train_data_f = np.array([[0, 1, 0, 0, 1, 0, 0, 0, 1],
                             [0, 0, 1, 0, 0, 1, 0, 0, 1],
                             [1, 0, 0, 0, 1, 0, 0, 0, 1],
                             [0, 0, 0, 1, 1, 0, 0, 1, 0],
                             [0, 1, 0, 0, 0, 0, 1, 0, 1]])
    train_data_y = np.array([5, 1, 4, 2, 3])
    # 测试数据
    test_data_f = np.array([[0, 1, 0, 0, 1, 0, 0, 0, 1],
                            [0, 0, 1, 0, 0, 1, 0, 0, 1],
                            [1, 0, 0, 0, 1, 0, 0, 0, 1]])
    test_data_y = np.array([5, 1, 4])

    train_y = train_data_y
    train_f1 = train_data_f[:, 0:4]
    train_f2 = train_data_f[:, 4:7]
    train_f3 = train_data_f[:, 7:10]
    test_y = test_data_y
    test_f1 = test_data_f[:, 0:4]
    test_f2 = test_data_f[:, 4:7]
    test_f3 = test_data_f[:, 7:10]
    return train_y, train_f1, train_f2, train_f3, test_y, test_f1, test_f2, test_f3


if __name__ == '__main__':
    path = 'data/'
    embedding_dim = 5
    lambd = 100

    # 加载数据
    train_y, train_f1, train_f2, train_f3, test_y, test_f1, test_f2, test_f3 = get_data()

    # 设置模型参数
    parser = argparse.ArgumentParser(description='XDeepFM')
    # CIN参数
    parser.add_argument('--CIN_input_channels', type=int, default=3)
    parser.add_argument('--CIN_num_layers', type=int, default=3)
    parser.add_argument('--CIN_num_units', type=int, default=[10, 10, 10])
    parser.add_argument('--CIN_output_dim', type=int, default=5)
    # DNN参数
    parser.add_argument('--DNN_input_dim', type=int, default=3 * embedding_dim)
    parser.add_argument('--DNN_num_layers', type=int, default=4)
    parser.add_argument('--DNN_num_units', type=int, default=[400, 400, 400, 400])
    parser.add_argument('--DNN_output_dim', type=int, default=5)
    # LINEAR参数
    parser.add_argument('--LINEAR_input_dim', type=int, default=9)
    parser.add_argument('--LINEAR_output_dim', type=int, default=5)
    # EMBEDDING参数
    parser.add_argument('--EMBEDDING_embedding_dim', type=int, default=embedding_dim)
    parser.add_argument('--EMBEDDING_input_dim_f1', type=int, default=4)
    parser.add_argument('--EMBEDDING_input_dim_f2', type=int, default=3)
    parser.add_argument('--EMBEDDING_input_dim_f3', type=int, default=2)

    parser.add_argument('--train_epoch', type=int, default=20)
    parser.add_argument('--base_lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--lambd', type=float, default=lambd)
    parser.add_argument('--device', type=str, default='cpu')  # [cuda:5,cpu]

    args = parser.parse_args()

    # 模型构建
    device = torch.device(args.device)
    # 创建网络
    model_XDeepFM = XDeepFM(args).to(device)
    # 损失函数
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model_XDeepFM.parameters(),
                           lr=args.base_lr,
                           weight_decay=args.lambd)

    # 设置训练数据
    train_dataset = Data.TensorDataset(torch.from_numpy(train_f1), torch.from_numpy(train_f2),
                                       torch.from_numpy(train_f3), torch.from_numpy(train_y.reshape(-1) - 1))
    train_dataloader = Data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )

    # 设置测试数据
    test_dataset = Data.TensorDataset(torch.from_numpy(test_f1), torch.from_numpy(test_f2),
                                      torch.from_numpy(test_f3), torch.from_numpy(test_y.reshape(-1) - 1))
    test_dataloader = Data.DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )

    l_train_loss = []
    l_test_loss = []
    l_train_mae = []
    l_test_mae = []
    l_train_rmse = []
    l_test_rmse = []
    for epoch in range(args.train_epoch):
        train_loss, train_mae, train_rmse = train(epoch)
        test_loss, test_mae, test_rmse = test(epoch)
        l_train_loss.append(train_loss)
        l_test_loss.append(test_loss)
        l_train_mae.append(train_mae)
        l_test_mae.append(test_mae)
        l_train_rmse.append(train_rmse)
        l_test_rmse.append(test_rmse)

    # 模型保存
    torch.save({'XDeepFM': model_XDeepFM.state_dict()}, 'model/model_XDeepFM.pt')

    np.save('log/train_loss.npy', l_train_loss)
    np.save('log/test_loss.npy', l_test_loss)
    np.save('log/train_mae.npy', l_train_mae)
    np.save('log/test_mae.npy', l_test_mae)
    np.save('log/train_rmse.npy', l_train_rmse)
    np.save('log/test_rmse.npy', l_test_rmse)
