import torch
import yaml
import pandas as pd
import torch.optim as optim
import numpy as np
from utils.loss import ObservedMSELoss
from torch.utils.data import DataLoader
from utils.dataset import AutoRecDataset
from model.AutoRec import AutoRec


# 训练
def train():
    Loss = 0
    MAE = 0
    total_steps = len(train_loader)
    for step, (batch_x, batch_y) in enumerate(train_loader):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        y_pred = autoRec(batch_x)[1]
        loss, mask = loss_func(batch_y, y_pred)
        mae = torch.sum(torch.abs(batch_y - y_pred * mask)) / torch.sum(mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        Loss += loss.cpu().data
        MAE += mae.cpu().data
        if step % 20 == 0:
            print('Epoch [{}/{}], Step [{}/{}], loss: {:.4f}, mae: {:.4f}'
                  .format(epoch + 1, config['EPOCH'], step, total_steps, loss, mae))
    Loss /= total_steps
    MAE /= total_steps
    print('Train-Epoch [{}/{}],loss: {:.4f}, mae: {:.4f}'
          .format(epoch + 1, config['EPOCH'], Loss, MAE))
    return Loss, MAE


def test():
    Loss = 0
    MAE = 0
    total_steps = len(test_loader)
    with torch.no_grad():
        for step, (batch_x, batch_y) in enumerate(test_loader):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            y_pred = autoRec(batch_x)[1]
            loss, mask = loss_func(batch_y, y_pred)
            mae = torch.sum(torch.abs(batch_y - y_pred * mask)) / torch.sum(mask)
            MAE += mae.cpu().data
            Loss += loss.cpu().data
    Loss /= total_steps
    MAE /= total_steps
    print('Test-Epoch [{}/{}],loss: {:.4f}, mae: {:.4f}'
          .format(epoch + 1, config['EPOCH'], Loss, MAE))
    return Loss, MAE


# 获取数据信息
def get_data_info():
    path = config['path'] + 'train/' + domain + '_train_triple.csv'
    data = pd.read_csv(path)
    user_set = set(list(data['user_id']))
    item_set = set(list(data['item_id']))

    num_u = len(user_set)
    num_i = len(item_set)

    return num_u, num_i


if __name__ == '__main__':
    # 读取配置文件
    path_config = '../config/config.yaml'
    f = open(path_config, 'r', encoding='utf-8')
    config = f.read()
    config = yaml.load(config, Loader=yaml.FullLoader)['AutoRec']

    is_filter = config['is_filter']
    # 加载数据
    for domain in config['domains']:
        # 训练数据
        train_dataset = AutoRecDataset(config, domain, mode='train', is_process=config['is_process'],
                                       is_filter=is_filter)
        is_filter = False
        train_loader = DataLoader(train_dataset, batch_size=config['BATCH_SIZE'], shuffle=True)
        # 测试数据
        test_dataset = AutoRecDataset(config, domain, mode='test')
        test_loader = DataLoader(test_dataset, batch_size=config['BATCH_SIZE'], shuffle=True)
        # 获取用户和物品数目
        num_user, num_item = get_data_info()
        
        # 创建模型
        autoRec = AutoRec(config, num_item)
        # 设置损失函数
        loss_func = ObservedMSELoss()
        # 设置优化器
        optimizer = optim.Adam(autoRec.parameters(), weight_decay=config['lambd'], lr=config['LR'])
        # 设置训练设备
        device = torch.device(config['device'][0])

        # 模型训练和测试
        l_train_loss = []
        l_test_loss = []
        l_train_mae = []
        l_test_mae = []
        for epoch in range(config['EPOCH']):
            train_loss, train_mae = train()
            test_loss, test_mae = test()
            l_train_loss.append(train_loss)
            l_test_loss.append(test_loss)
            l_train_mae.append(train_mae)
            l_test_mae.append(test_mae)

        # 模型保存
        torch.save({'autoRec': autoRec.state_dict()}, config['path_model'] + domain + '_autoRec_model.pt')
        np.save(config['path_log'] + domain + '_train_loss.npy', l_train_loss)
        np.save(config['path_log'] + domain + '_test_loss.npy', l_test_loss)
        np.save(config['path_log'] + domain + '_train_mae.npy', l_train_mae)
        np.save(config['path_log'] + domain + '_test_mae.npy', l_test_mae)
