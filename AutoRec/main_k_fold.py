import pandas as pd
import numpy as np
import torch
import data
import math
import time
import argparse
import torch.utils.data as Data
import torch.optim as optim
from AutoRec import AutoRec, AutoRecLoss


# 训练
def train(epoch, h, lambd):
    Loss = 0
    MAE = 0
    total_steps = len(train_dataloader)
    for step, (batch_x, batch_mask_x, batch_y) in enumerate(train_dataloader):
        batch_x = batch_x.type(torch.FloatTensor).to(device)
        batch_y = batch_y.type(torch.FloatTensor).to(device)
        batch_mask_x = batch_mask_x.type(torch.FloatTensor).to(device)

        y_pred = autoRec(batch_x)[1]
        loss, mae = loss_func(autoRec, batch_y, y_pred, batch_mask_x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        Loss += loss.cpu().data.numpy()
        MAE += mae
        if step % 2 == 0:
            print('H:{}, Lambda:{:.4f}, Epoch [{}/{}], Step [{}/{}], loss: {:.4f},mae: {:.4f}'
                  .format(h, lambd, epoch + 1, args.train_epoch, step, total_steps, loss, mae))
    Loss /= total_steps
    MAE /= total_steps
    print('Train-Epoch [{}/{}],loss: {:.4f},mae: {:.4f}'
          .format(epoch + 1, args.train_epoch, Loss, MAE))
    return Loss, MAE


def test(epoch):
    Loss = 0
    MAE = 0
    total_steps = len(test_dataloader)
    with torch.no_grad():
        for step, (batch_x, batch_mask_x, batch_y) in enumerate(test_dataloader):
            batch_x = batch_x.type(torch.FloatTensor).to(device)
            batch_y = batch_y.type(torch.FloatTensor).to(device)
            batch_mask_x = batch_mask_x.type(torch.FloatTensor).to(device)

            y_pred = autoRec(batch_x)[1]
            loss, mae = loss_func(autoRec, batch_y, y_pred, batch_mask_x)
            Loss += loss.cpu().data.numpy()
            MAE += mae
    Loss /= total_steps
    MAE /= total_steps
    print('Test-Epoch [{}/{}],loss: {:.4f},mae: {:.4f}'
          .format(epoch + 1, args.train_epoch, Loss, MAE))
    return Loss, MAE


if __name__ == '__main__':
    k = 5
    H = [10, 20, 40, 80, 100, 200, 300, 400, 500]
    Lambd = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    Domain_name = ['Cell_Phones_and_Accessories', 'Industrial_and_Scientific', 'Software']
    for domain_name in Domain_name:
        # 分割交叉验证数据
        data_path = 'data/amazon/correlated/' + domain_name + '/' + domain_name + '.csv'
        k_path = 'data/amazon/correlated/' + domain_name + '/k_fold/'
        data.k_fold_triple(data_path, k, k_path)
        # 设置交叉验证结果表格
        col_names = ['h', 'lambda', 'mae']
        result = pd.DataFrame(columns=col_names)
        result.to_csv('data/amazon/correlated/' + domain_name + '/log/k_fold_mae.csv', index=False)
        # 交叉验证
        for h in H:
            for lambd in Lambd:
                # 设置模型参数
                parser = argparse.ArgumentParser(description='U-AutoRec ')
                parser.add_argument('--hidden_dim', type=int, default=h)
                parser.add_argument('--lambd', type=float, default=lambd)
                parser.add_argument('--train_epoch', type=int, default=200)
                parser.add_argument('--batch_size', type=int, default=20)
                parser.add_argument('--base_lr', type=float, default=1e-4)
                parser.add_argument('--device', type=str, default='cpu')  # [cuda:5,cpu]

                args = parser.parse_args()
                # 加载数据
                MAE = 0
                for test_id in range(k):
                    train_data, train_mask, test_data, test_mask = data.get_data_k_fold(k_path, test_id, k)

                    # 用户数目和项目数目
                    num_users, num_items = train_data.shape

                    # 模型构建
                    device = torch.device(args.device)
                    autoRec = AutoRec(args, num_users, num_items).to(device)
                    loss_func = AutoRecLoss(args).to(device)
                    optimizer = optim.Adam(autoRec.parameters(), lr=args.base_lr)

                    # 设置训练数据
                    train_dataset = Data.TensorDataset(torch.from_numpy(train_data), torch.from_numpy(train_mask),
                                                       torch.from_numpy(train_data))
                    train_dataloader = Data.DataLoader(
                        dataset=train_dataset,
                        batch_size=args.batch_size,
                        shuffle=True
                    )

                    # 设置测试数据
                    test_dataset = Data.TensorDataset(torch.from_numpy(test_data), torch.from_numpy(test_mask),
                                                      torch.from_numpy(test_data))
                    test_dataloader = Data.DataLoader(
                        dataset=test_dataset,
                        batch_size=args.batch_size,
                        shuffle=True
                    )

                    # 训练
                    test_mae = 0
                    for epoch in range(args.train_epoch):
                        train_loss, train_mae = train(epoch, h, lambd)
                        test_loss, test_mae = test(epoch)
                    MAE += test_mae

                MAE /= k
                # 存储交叉验证结果
                result_slice = pd.DataFrame(
                    {'h': [h], 'lambda': [lambd], 'mae': [MAE]})
                result_slice.to_csv('data/amazon/correlated/' + domain_name + '/log/k_fold_mae.csv'
                                    , index=False, mode='a', header=False)
