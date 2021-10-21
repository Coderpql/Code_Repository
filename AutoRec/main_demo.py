import pandas as pd
import numpy as np
import torch
import data
import argparse
import torch.utils.data as Data
import torch.optim as optim
from AutoRec import AutoRec, AutoRecLoss

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

        data_triple = pd.read_csv('data/amazon/correlated/' + domain_name + '/' + domain_name + '.csv')
        m = data.create_rating_matrix(data_triple)
        data_m = m.values

        # 用户数目和项目数目
        num_users, num_items = data_m.shape

        # 创建模型
        device = torch.device(args.device)
        autoRec = AutoRec(args, num_users, num_items)
        autoRec = autoRec.to(device)
        state_dict = torch.load('data/amazon/correlated/' + domain_name + '/model/autoRec_model.pt',
                                map_location=torch.device('cpu'))
        autoRec.load_state_dict(state_dict['autoRec'])

        # 设置训练数据
        dataset = Data.TensorDataset(torch.from_numpy(data_m))
        dataloader = Data.DataLoader(
            dataset=dataset,
            batch_size=args.batch_size,
            shuffle=False
        )

        # 使用相应模型对数据进行降维
        code_low_dim = None
        with torch.no_grad():
            for s, batch_x in enumerate(dataloader):
                batch_x = batch_x[0]
                batch_x = batch_x.type(torch.FloatTensor).to(device)

                # 对数据进行降维
                code = autoRec(batch_x)[0]

                # 数据存储
                code = code.cpu().data.numpy()

                if s == 0:
                    code_low_dim = code
                else:
                    code_low_dim = np.vstack((code_low_dim, code))

        df_code = pd.DataFrame(code_low_dim)
        df_code.index = m.index
        df_code.to_csv('data/amazon/correlated/' + domain_name + '/feature/' + domain_name + '_feature.csv')
