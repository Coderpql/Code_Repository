import torch
import yaml
import pandas as pd
import torch.optim as optim
import numpy as np
from utils.loss import UDARecLoss
from torch.utils.data import DataLoader
from utils.dataset import UDARecDataset
from model.U_DARec import U_DARec


# 训练
def train():
    Loss = 0
    MAE_T = 0
    MAE_S = 0
    RMSE_T = 0
    RMSE_S = 0
    total_steps = len(train_loader)
    for step, (batch_x_T, batch_y_T, label_T, batch_x_S, batch_y_S, label_S) in enumerate(train_loader):
        batch_x_T = batch_x_T.to(device)
        batch_y_T = batch_y_T.to(device)
        label_T = label_T.type(torch.LongTensor).to(device)
        batch_x_S = batch_x_S.to(device)
        batch_y_S = batch_y_S.to(device)
        label_S = label_S.type(torch.LongTensor).to(device)

        # 目标域
        y_T, y_S, c = model_DARec(batch_x_T)
        loss_T, loss_RP_T, mask_T, loss_RP_S, loss_DC, mask_S = loss_func(batch_y_T, batch_y_S, y_T, y_S, c, label_T)
        mae_T = torch.sum(torch.abs(batch_y_T - y_T * mask_T)) / torch.sum(mask_T)
        rmse_T = torch.sum(torch.square(batch_y_T - y_T * mask_T)) / torch.sum(mask_T)
        # 辅助域
        y_T, y_S, c = model_DARec(batch_x_S, is_target=False)
        loss_S, loss_RP_T, mask_T, loss_RP_S, loss_DC, mask_S = loss_func(batch_y_T, batch_y_S, y_T, y_S, c, label_S)
        mae_S = torch.sum(torch.abs(batch_y_S - y_S * mask_S)) / torch.sum(mask_S)
        rmse_S = torch.sum(torch.square(batch_y_S - y_S * mask_S)) / torch.sum(mask_S)

        loss = loss_T + loss_S
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        Loss += loss.cpu().data
        MAE_T += mae_T.cpu().data
        MAE_S += mae_S.cpu().data
        RMSE_T += rmse_T.cpu().data
        RMSE_S += rmse_S.cpu().data
        if step % 20 == 0:
            print('Epoch [{}/{}], Step [{}/{}], loss: {:.4f}, mae_T: {:.4f}, mae_S: {:.4f}, rmse_T: {:.4f}, '
                  'rmse_S: {:.4f}'.format(epoch + 1, config['DARec']['parameter']['EPOCH'],
                                          step, total_steps, loss, mae_T, mae_S, rmse_T, rmse_S))
    Loss /= total_steps
    MAE_T /= total_steps
    MAE_S /= total_steps
    RMSE_T /= total_steps
    RMSE_S /= total_steps
    print('Train-Epoch [{}/{}],loss: {:.4f}, mae_T: {:.4f}, mae_S: {:.4f}, rmse_T: {:.4f}, rmse_S: {:.4f}'
          .format(epoch + 1, config['DARec']['parameter']['EPOCH'], Loss, MAE_T, MAE_S, RMSE_T, RMSE_S))
    return Loss, MAE_T, MAE_S, RMSE_T, RMSE_S


def test():
    Loss = 0
    MAE_T = 0
    MAE_S = 0
    RMSE_T = 0
    RMSE_S = 0
    total_steps = len(test_loader)
    with torch.no_grad():
        for step, (batch_x_T, batch_y_T, label_T, batch_x_S, batch_y_S, label_S) in enumerate(test_loader):
            batch_x_T = batch_x_T.to(device)
            batch_y_T = batch_y_T.to(device)
            label_T = label_T.type(torch.LongTensor).to(device)
            batch_y_S = batch_y_S.to(device)

            # 目标域
            y_T, y_S, c = model_DARec(batch_x_T)
            loss_T, loss_RP_T, mask_T, loss_RP_S, loss_DC, mask_S = loss_func(batch_y_T, batch_y_S, y_T, y_S, c, label_T)
            mae_T = torch.sum(torch.abs(batch_y_T - y_T * mask_T)) / torch.sum(mask_T)
            rmse_T = torch.sum(torch.square(batch_y_T - y_T * mask_T)) / torch.sum(mask_T)

            loss = loss_T
            mae_S = torch.sum(torch.abs(batch_y_S - y_S * mask_S)) / torch.sum(mask_S)
            rmse_S = torch.sum(torch.square(batch_y_S - y_S * mask_S)) / torch.sum(mask_S)
            MAE_T += mae_T.cpu().data
            MAE_S += mae_S.cpu().data
            RMSE_T += rmse_T.cpu().data
            RMSE_S += rmse_S.cpu().data
            Loss += loss.cpu().data
    Loss /= total_steps
    MAE_T /= total_steps
    MAE_S /= total_steps
    RMSE_T /= total_steps
    RMSE_S /= total_steps
    print('Test-Epoch [{}/{}],loss: {:.4f}, mae_T: {:.4f}, mae_S: {:.4f}, rmse_T: {:.4f}, rmse_S: {:.4f}'
          .format(epoch + 1, config['DARec']['parameter']['EPOCH'], Loss, MAE_T, MAE_S, RMSE_T, RMSE_S))
    return Loss, MAE_T, MAE_S, RMSE_T, RMSE_S


# 获取数据信息
def get_data_info(path):
    data = pd.read_csv(path)
    item_set = set(list(data['item_id']))

    num_i = len(item_set)

    return num_i


if __name__ == '__main__':
    # 读取配置文件
    path_config = '../config/config.yaml'
    f = open(path_config, 'r', encoding='utf-8')
    config = f.read()
    config = yaml.load(config, Loader=yaml.FullLoader)

    target = config['DARec']['parameter']['domains'][0]
    source = config['DARec']['parameter']['domains'][1]
    # 训练数据
    train_dataset = UDARecDataset(config, target, source, mode='train', is_process=True)
    train_loader = DataLoader(train_dataset, batch_size=config['DARec']['parameter']['BATCH_SIZE'], shuffle=True)
    # 测试数据
    test_dataset = UDARecDataset(config, target, source, mode='test')
    test_loader = DataLoader(test_dataset, batch_size=config['DARec']['parameter']['BATCH_SIZE'], shuffle=True)
    # 获取用户和物品数目
    path_T = config['DARec']['parameter']['path'] + 'target/train/' + target + '_train_triple.csv'
    num_item_T = get_data_info(path_T)
    path_S = config['DARec']['parameter']['path'] + 'source/' + source + '_triple.csv'
    num_item_S = get_data_info(path_S)

    # 创建模型
    model_DARec = U_DARec(config, num_item_T, num_item_S)
    state_dict_T = torch.load(config['AutoRec']['path_model'] + target + '_autoRec_model.pt',
                              map_location=torch.device('cpu'))
    state_dict_S = torch.load(config['AutoRec']['path_model'] + source + '_autoRec_model.pt',
                              map_location=torch.device('cpu'))
    model_DARec.T_AutoRec.load_state_dict(state_dict_T['autoRec'])
    model_DARec.S_AutoRec.load_state_dict(state_dict_S['autoRec'])
    # 设置损失函数
    loss_func = UDARecLoss(config)
    # 设置优化器
    optimizer = optim.Adam(model_DARec.parameters(), weight_decay=config['DARec']['parameter']['lambd'],
                           lr=config['DARec']['parameter']['LR'])
    # 设置训练设备
    device = torch.device(config['DARec']['parameter']['device'][0])

    # 模型训练和测试
    l_train_loss = []
    l_test_loss = []
    l_train_mae_T = []
    l_train_mae_S = []
    l_test_mae_T = []
    l_test_mae_S = []
    l_train_rmse_T = []
    l_train_rmse_S = []
    l_test_rmse_T = []
    l_test_rmse_S = []
    for epoch in range(config['DARec']['parameter']['EPOCH']):
        train_loss, train_mae_T, train_mae_S, train_rmse_T, train_rmse_S = train()
        test_loss, test_mae_T, test_mae_S, test_rmse_T, test_rmse_S = test()
        l_train_loss.append(train_loss)
        l_test_loss.append(test_loss)
        l_train_mae_T.append(train_mae_T)
        l_train_mae_S.append(train_mae_S)
        l_test_mae_T.append(test_mae_T)
        l_test_mae_S.append(test_mae_S)
        l_train_rmse_T.append(train_rmse_T)
        l_train_rmse_S.append(train_rmse_S)
        l_test_rmse_T.append(test_rmse_T)
        l_test_rmse_S.append(test_rmse_S)

    # 模型保存
    torch.save({'DARec': model_DARec.state_dict()}, config['DARec']['parameter']['path_model'] + 'C1_DARec_model.pt')
    np.save(config['DARec']['parameter']['path_log'] + 'C1_train_loss.npy', l_train_loss)
    np.save(config['DARec']['parameter']['path_log'] + 'C1_test_loss.npy', l_test_loss)
    np.save(config['DARec']['parameter']['path_log'] + 'C1_train_mae_T.npy', l_train_mae_T)
    np.save(config['DARec']['parameter']['path_log'] + 'C1_train_mae_S.npy', l_train_mae_S)
    np.save(config['DARec']['parameter']['path_log'] + 'C1_test_mae_T.npy', l_test_mae_T)
    np.save(config['DARec']['parameter']['path_log'] + 'C1_test_mae_S.npy', l_test_mae_S)
    np.save(config['DARec']['parameter']['path_log'] + 'C1_train_rmse_T.npy', l_train_rmse_T)
    np.save(config['DARec']['parameter']['path_log'] + 'C1_train_rmse_S.npy', l_train_rmse_S)
    np.save(config['DARec']['parameter']['path_log'] + 'C1_test_rmse_T.npy', l_test_rmse_T)
    np.save(config['DARec']['parameter']['path_log'] + 'C1_test_rmse_S.npy', l_test_rmse_S)
