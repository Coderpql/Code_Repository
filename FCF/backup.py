import math
from random import sample
import numpy as np

import pandas as pd


# 训练集测试集分割(构建矩阵)
def split_matrix():
    data = pd.read_csv('data/ml-1m/rating_matrix.csv', index_col='user_id')
    data_train = pd.DataFrame(index=data.index, columns=data.columns, dtype=np.float32)
    data_train = data_train.fillna(0)
    data_test = pd.read_csv('data/ml-1m/rating_matrix.csv', index_col='user_id')

    # 数据分割
    # 设置分割比例
    percent = 0.8
    i_k = 0
    for index in data.index:
        print('Split , index_id:', i_k)
        list_items = []
        for column in data.columns:
            if data.at[index, column] > 0:
                list_items.append(column)
        # 计算每行取的评分数
        if len(list_items) == 1:
            num_rating = 1
            choose_items = sample(list_items, num_rating)
            print('Split , index_id:', i_k, 'num_rating:', num_rating)
            for col in choose_items:
                data_train.at[index, col] = data.at[index, col]
                data_test.at[index, col] = 0

        else:
            num_rating = math.floor(len(list_items) * percent)
            choose_items = sample(list_items, num_rating)
            print('Split , index_id:', i_k, 'num_rating:', num_rating)
            for col in choose_items:
                data_train.at[index, col] = data.at[index, col]
                data_test.at[index, col] = 0
        i_k += 1
    # num_rating1 = np.count_nonzero(data_train.values)

    # 数据填充
    df_num = pd.DataFrame((data_train != 0).astype('int').sum(axis=0))
    for idx in df_num.index:
        print('Fill Data, item:', idx)
        if df_num.at[idx, 0] == 0:
            # 在测试数据中随机选取数据进行填充
            df = data_test.loc[(data_test[idx] != 0)]
            n = df.values.shape[0]
            pos = np.random.randint(0, n)
            pos = df.index[pos - 1]
            data_train.at[pos, idx] = data.at[pos, idx]
            data_test.at[pos, idx] = 0
    # 数据检测
    df_num = pd.DataFrame((data_train != 0).astype('int').sum(axis=0))
    for idx in df_num.index:
        if df_num.at[idx, 0] == 0:
            print('data_train:', idx, 'error')

    # 数据存储
    df_triple_train = pd.DataFrame(columns=['user_id', 'item_id', 'score'])
    df_triple_train.to_csv('data/ml-1m/train_test/train_80_triple.csv', index=False)
    data_train = data_train.astype('int')
    i = 1
    for index in data_train.index:
        for col in data_train.columns:
            if data_train.loc[index, col] != 0:
                i += 1
                train_slice = pd.DataFrame({'user_id': [index], 'item_id': [col],
                                            'score': [data_train.loc[index, col]]})
                train_slice.to_csv('data/ml-1m/train_test/train_80_triple.csv', index=False, header=False, mode='a')

    df_triple_test = pd.DataFrame(columns=['user_id', 'item_id', 'score'])
    df_triple_test.to_csv('data/ml-1m/train_test/test_80_triple.csv', index=False)
    data_test = data_test.astype('int')
    for index in data_test.index:
        for col in data_test.columns:
            if data_test.loc[index, col] != 0:
                test_slice = pd.DataFrame({'user_id': [index], 'item_id': [col],
                                           'score': [data_test.loc[index, col]]})
                test_slice.to_csv('data/ml-1m/train_test/test_80_triple.csv', index=False, header=False, mode='a')


# 创建K折交叉验证数据
def k_fold_matrix(k, data):
    # 创建分割数据的dataframe
    L_data = locals()
    for i in range(k):
        L_data['data' + str(i)] = pd.DataFrame(index=data.index, columns=data.columns, dtype=np.float32)
        L_data['data' + str(i)] = L_data['data' + str(i)].fillna(0)

    # 设置分割比例
    percent = 1 / float(k)

    # 数据分割
    i_k = 0
    for index in data.index:
        list_items = []
        for column in data.columns:
            if data.at[index, column] > 0:
                list_items.append(column)
        # 计算每行取的评分数
        num_rating = math.floor(len(list_items) * percent)
        if num_rating > len(list_items):
            num_rating = len(list_items)
        for i in range(k):
            if num_rating != 0:
                if len(list_items) <= num_rating and len(list_items) != 0:
                    choose_items = list_items
                elif len(list_items) == 0:
                    break
                else:
                    choose_items = sample(list_items, num_rating)
                list_items = [x for x in list_items if x not in choose_items]
                for col in choose_items:
                    L_data['data' + str(i)].at[index, col] = data.at[index, col]
                    print('Division-k ,index_id:', i_k, 'num_rating:', num_rating, 'col:', col)
        i_k += 1
    # 数据存储
    for i in range(k):
        df_triple_train = pd.DataFrame(columns=['user_id', 'item_id', 'score'])
        df_triple_train.to_csv('data/ml-1m/k_fold/data_' + str(i) + '.csv', index=False)
        L_data['data' + str(i)] = L_data['data' + str(i)].astype('int')
        for index in L_data['data' + str(i)].index:
            for col in L_data['data' + str(i)].columns:
                if L_data['data' + str(i)].loc[index, col] != 0:
                    train_slice = pd.DataFrame({'user_id': [index], 'item_id': [col],
                                                'score': [L_data['data' + str(i)].loc[index, col]]})
                    train_slice.to_csv('data/ml-1m/k_fold/data_' + str(i) + '.csv', index=False, header=False,
                                       mode='a')


if __name__ == '__main__':
    print('start')
    # # 将三元组格式的数据转换为矩阵
    # rating_data_matrix = create_rating_matrix(rating_data_triple)
    # rating_data_matrix.to_csv('data/ml-1m/rating_matrix.csv')
    # # 训练集测试集分割，分割比例为80:20
    # split_matrix()
    # # 创建k折交叉验证数据
    # train_data_triple = pd.read_csv('data/ml-1m/train_test/train_80_triple.csv')
    # train_data_matrix = create_rating_matrix(train_data_triple)
    # k_fold_matrix(k=5, data=train_data_matrix)
