from random import sample
import numpy as np
import pandas as pd
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error


# 客户端
class Client(object):
    def __init__(self, conf, l_u_items, user_id, Data, Data_eval):
        self.conf = conf
        self.items = l_u_items
        self.user_id = user_id
        # 获取模型参数
        self.lambd = self.conf['lambda']
        self.K = self.conf['K']
        self.alpha = self.conf['alpha']
        # 提取当前用户数据
        self.Data = Data
        self.Data_eval = Data_eval
        # 获取已购商品列表
        self.local_items = list(self.Data['item_id'])
        # 初始化本地用户隐特征
        self.X = np.random.rand(1, self.K).astype(np.float32)  # (1, K)
        # 构建置信度矩阵
        self.C = pd.DataFrame(index=[self.items], columns=self.items)
        self.C = self.C.fillna(0)
        # 构建偏好向量
        self.P = pd.DataFrame(index=[self.user_id], columns=self.items)
        self.P = self.P.fillna(0)
        # 对C和P进行填充
        for item in self.items:
            if item in self.local_items:
                block = self.Data.loc[self.Data['item_id'] == item]
                self.C.loc[item, item] = 1 + self.alpha * block.loc[block.index[0], 'score']
                self.P.loc[self.user_id, item] = 1
            else:
                self.C.loc[item, item] = 1
        # # 构建偏好向量
        # self.P = pd.DataFrame(index=[self.user_id], columns=self.items)
        # self.P = self.P.fillna(0)
        # for item in self.items:
        #     if item in self.local_items:
        #         self.P.loc[self.user_id, item] = 1
        # 创建更新物品隐特征的变量
        self.F = pd.DataFrame(index=[self.user_id], columns=self.items)
        self.F = self.F.fillna(0)
        # 创建本地商品隐特征
        self.Y_local = None

    # 本地训练
    def local_train(self, Y):
        # 将物品隐特征转换为矩阵
        i = 0
        for item in self.items:
            if i == 0:
                self.Y_local = Y[item]
            else:
                self.Y_local = np.vstack((self.Y_local, Y[item]))
            i += 1
        self.Y_local = self.Y_local.T  # (K,M)
        # 更新本地用户隐特征
        self.X = (np.matmul(np.matmul(np.matmul(np.linalg.inv(
            np.matmul(np.matmul(self.Y_local, self.C), self.Y_local.T) +
            self.lambd * np.ones([self.K, self.K])), self.Y_local), self.C), self.P.values.T)).T

    # 计算损失
    def get_loss(self, Y):
        # 计算本地损失
        Loss_local = 0
        for item in self.items:
            Loss_local += self.C.loc[item, item] * np.square(
                self.P.loc[self.user_id, item] - np.matmul(self.X, Y[item].T)) \
                          + self.lambd * np.square(np.linalg.norm(self.X))
        return Loss_local

    # 上传f_ui
    def upload(self, Y):
        for item in self.items:
            print(self.C.loc[item, item])
            f = (self.C.loc[item, item] * (self.P.loc[self.user_id, item] - np.matmul(self.X, Y[item].T))) * self.X
            self.F.loc[self.user_id, item] = f
        return self.F.values

    # 本地测试
    def eval_local(self):
        score_pred = np.matmul(self.X, self.Y_local)
        score_real = np.array(self.Data_eval['score'])
        eval_res = np.hstack((score_pred, score_real))
        return eval_res


# 服务端
class Server(object):
    def __init__(self, conf, l_items, l_users):
        self.conf = conf
        self.items = l_items
        self.users = l_users
        # 获取模型参数
        self.gamma = self.conf['gamma']
        self.iterations = self.conf['iterations']
        self.K = conf['K']
        # 初始化物品因特征
        self.Y = dict(zip(
            self.items,
            np.random.rand(len(self.items), self.K).astype(np.float32)
        ))

    # 模型融合
    def model_aggregate(self, F):
        item_id = sample(self.items, 1)
        # 更新物品隐特征
        self.Y[item_id] -= self.gamma * F[item_id]


# 读取数据
def get_data(train_path, test_path):
    data_train = pd.read_csv(train_path)
    data_test = pd.read_csv(test_path)
    return data_train, data_test


if __name__ == '__main__':
    # 导入数据
    path_train = 'data/ml-1m/train_test/train_80_20.csv'
    path_test = 'data/ml-1m/train_test/test_80_20.csv'
    train_data, test_data = get_data(path_train, path_test)

    # 获取物品列表
    items = list(set(list(train_data['item_id'])))
    # 获取用户列表
    users = list(set(list(train_data['user_id'])))

    # 对数据进行裁切
    users = users[0:10]
    train_data = train_data.loc[train_data['user_id'].isin(users)]
    items = list(set(list(train_data['item_id'])))
    test_data = test_data.loc[test_data['user_id'].isin(users)]
    test_data = test_data.loc[test_data['item_id'].isin(items)]

    # 加载服务器配置文件
    path_conf_server = 'conf/Server_conf.json'
    with open(path_conf_server, 'r') as file_server:
        conf_server = json.load(file_server)
    # 加载客户端配置文件
    path_conf_client = 'conf/Client_conf.json'
    with open(path_conf_client, 'r') as file_client:
        conf_client = json.load(file_client)

    # 构建服务端对象
    server = Server(conf_server, items, users)
    # 构建客户端对象
    clients = []
    for user in users:
        print('Client:', user)
        local_Data = train_data.loc[train_data['user_id'].isin([user])]
        local_Data_eval = test_data.loc[test_data['user_id'].isin([user])]
        clients.append(Client(conf_client, items, user, local_Data, local_Data_eval))

    # 模型训练
    L_Loss = []
    L_MAE_train = []
    L_MAE_test = []
    for epoch in range(conf_server['Epoch']):
        # 客户端训练
        F_total = None
        c_id = 0
        for client in clients:
            client.local_train(server.Y)
            F_u = client.upload(server.Y)
            if c_id == 0:
                F_total = F_u
            else:
                F_total = np.vstack((F_total, F_u))
            c_id += 1
        F_total = dict(zip(items, F_total.T))

        # 服务器端训练
        for iteration in range(server.iterations):
            server.model_aggregate(F_total)
            Loss = 0
            res_train, res_test = None, None
            for client in clients:
                # 获取测试结果
                if Loss == 0:
                    res_train, res_test = client.eval_local()
                else:
                    res_train = np.vstack((res_train, client.eval_local()[0]))
                    res_test = np.vstack((res_test, client.eval_local()[1]))
                # 计算损失值
                Loss += client.get_loss(server.Y)
            print('Epoch:', epoch, 'Iteration', iteration, 'Loss:', Loss)
            Loss += conf_client['lambda'] * np.square(np.linalg.norm(server.Y))
            L_Loss.append(Loss)
            # 计算测试集和训练集mae
            MAE_train = mean_absolute_error(res_train[-1, 0], res_train[-1, 1])
            MAE_test = mean_absolute_error(res_test[-1, 0], res_test[-1, 1])
            L_MAE_train.append(MAE_train)
            L_MAE_test.append(MAE_test)
        # 保存测试结果
        np.save('eval/mae_train.npy', L_MAE_train)
        np.save('eval/mae_train.npy', L_MAE_test)

    # 模型保存
    np.save('model/Server/Y.npy', server.Y)
    c_id = 0
    for client in clients:
        np.save('model/Client/' + str(users[c_id]) + '.npy', client.X)
