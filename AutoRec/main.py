import pandas as pd
import numpy as np
import torch
import data
import argparse
import torch.utils.data as Data
import torch.optim as optim
from AutoRec import AutoRec, AutoRecLoss


# 训练
def train(epoch):
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
        Loss += loss.cpu().data
        MAE += mae
        if step % 2 == 0:
            print('Epoch [{}/{}], Step [{}/{}], loss: {:.4f},mae: {:.4f}'
                  .format(epoch + 1, args.train_epoch, step, total_steps, loss, mae))
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
            Loss += loss.cpu().data
            MAE += mae
    Loss /= total_steps
    MAE /= total_steps
    print('Test-Epoch [{}/{}],loss: {:.4f},mae: {:.4f}'
          .format(epoch + 1, args.train_epoch, Loss, MAE))
    return Loss, MAE


if __name__ == '__main__':

    optimal_para = pd.read_csv('data/amazon/correlated/optimal_parameter.csv', index_col='domain_name')
    Domain_name = ['Cell_Phones_and_Accessories', 'Industrial_and_Scientific', 'Software']
    for domain_name in Domain_name:
        h = optimal_para.loc[domain_name, 'h']
        lambd = optimal_para.loc[domain_name, 'lambda']
        # 设置模型参数
        parser = argparse.ArgumentParser(description='U-AutoRec')
        parser.add_argument('--hidden_dim', type=int, default=int(h))
        parser.add_argument('--lambd', type=float, default=lambd)
        parser.add_argument('--train_epoch', type=int, default=200)
        parser.add_argument('--batch_size', type=int, default=20)
        parser.add_argument('--base_lr', type=float, default=1e-4)
        parser.add_argument('--device', type=str, default='cpu')  # [cuda:5,cpu]

        args = parser.parse_args()
        # 加载数据
        path = 'data/amazon/correlated/' + domain_name + '/'
        ratio = 1
        train_data, train_mask, test_data, test_mask = data.get_data(path, ratio, domain_name)

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

        l_train_loss = []
        l_test_loss = []
        l_train_mae = []
        l_test_mae = []
        for epoch in range(args.train_epoch):
            train_loss, train_mae = train(epoch)
            test_loss, test_mae = test(epoch)
            l_train_loss.append(train_loss)
            l_test_loss.append(test_loss)
            l_train_mae.append(train_mae)
            l_test_mae.append(test_mae)

        # 模型保存
        torch.save({'autoRec': autoRec.state_dict()}, 'data/amazon/correlated/' + domain_name + '/model'
                                                                                                   '/autoRec_model.pt')
        np.save('data/amazon/correlated/' + domain_name + '/log/train_loss.npy', l_train_loss)
        np.save('data/amazon/correlated/' + domain_name + '/log/test_loss.npy', l_test_loss)
        np.save('data/amazon/correlated/' + domain_name + '/log/train_mae.npy', l_train_mae)
        np.save('data/amazon/correlated/' + domain_name + '/log/test_mae.npy', l_test_mae)
