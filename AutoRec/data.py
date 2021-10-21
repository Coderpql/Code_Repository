from random import sample
import numpy as np
import pandas as pd
import math

np.random.seed(1000)


# 训练集测试集分割(三元组)
def split_triple(data_path, percent, train_path, test_path):
    # 读取数据
    data = pd.read_csv(data_path)
    # 创建训练、测试数据表格
    data_train = pd.DataFrame(columns=data.columns, dtype=np.int32)
    data_test = pd.DataFrame(columns=data.columns, dtype=np.int32)

    # 获取用户列表
    set_users = set(list(data['user_id']))

    # 数据分割
    for user in set_users:
        print('Split, user_id:', user)
        data_block = data.loc[data['user_id'].isin([user])]
        n = data_block.values.shape[0]
        num_rating = math.ceil(n * percent)
        if num_rating > n:
            num_rating = n
        total_pos = list(data_block.index)
        # 随机抽取训练数据
        index_train = sample(total_pos, num_rating)
        # 抽取完训练数据，剩余数据作为测试数据
        index_test = [x for x in total_pos if x not in index_train]
        # 将分割出来的训练数据加入训练集
        data_train_slice = data_block.loc[index_train, :]
        data_train = pd.concat([data_train, data_train_slice])
        # 将分割出来的测试数据加入测试集
        data_test_slice = data_block.loc[index_test, :]
        data_test = pd.concat([data_test, data_test_slice])

    # 数据填充
    # 补全空列
    train_item = data_train['item_id']
    test_item = data_test['item_id']

    train_item = list(train_item)
    test_item = list(test_item)

    train_item = set(train_item)
    test_item = set(test_item)

    # 获取测试集含有训练集不含有的用户集合
    set_i = test_item - train_item

    if len(set_i) != 0:
        for i in set_i:
            data_fill = data_test.loc[data_test['item_id'].isin([i])]
            n = data_fill.values.shape[0]
            pos = np.random.randint(0, n)
            index = data_fill.index[pos]
            data_slice = pd.DataFrame({'user_id': [data_fill.loc[index, 'user_id']],
                                       'item_id': [data_fill.loc[index, 'item_id']],
                                       'score': [data_fill.loc[index, 'score']]})
            data_train = pd.concat([data_train, data_slice])
            data_test = data_test.drop(index=index)
    data_is_ok(data_train, data_test)
    # 存储数据
    print('Store data to ' + train_path + '......')
    data_train.to_csv(train_path, index=False)
    print('Store data to ' + test_path + '......')
    data_test.to_csv(test_path, index=False)

    return data_train, data_test


# 判断是否有空行或者空列
def data_is_ok(data_train, data_test):
    # 数据检测
    train_user = data_train['user_id']
    test_user = data_test['user_id']

    train_user = list(train_user)
    test_user = list(test_user)

    train_user = set(train_user)
    test_user = set(test_user)

    if len(test_user - train_user) != 0:
        print('row is error')
        print('missing user_id:', test_user - train_user)
    else:
        print('row is ok')

    train_item = data_train['item_id']
    test_item = data_test['item_id']

    train_item = list(train_item)
    test_item = list(test_item)

    train_item = set(train_item)
    test_item = set(test_item)

    if len(test_item - train_item) != 0:
        print('column is error')
        print('missing item_id:', test_item - train_item)
    else:
        print('column is ok')


# 创建评分矩阵
def create_rating_matrix(data):
    # 构建评分矩阵
    table_select = pd.pivot_table(data, index=['user_id'], columns=['item_id'], values=['score'])
    # 删除多余索引
    table_select.columns = table_select.columns.droplevel(0)
    table_select = table_select.reset_index()
    table_select = pd.concat([pd.DataFrame(data=table_select.index.tolist(),
                                           columns=[table_select.index.name],
                                           index=table_select.index.tolist()), table_select], axis=1)
    col_list = table_select.columns.tolist()
    col_list.remove(None)
    table_select = table_select.loc[:, col_list]
    table_select = table_select.set_index('user_id')
    table_select = table_select.fillna(0.0)
    return table_select.astype('int')


# 分割交叉验证数据
def k_fold_triple(data_path, k, k_path):
    # 读取数据
    data = pd.read_csv(data_path)
    # 创建分割数据的dataframe
    L_data = locals()
    for i in range(k):
        L_data['data' + str(i)] = pd.DataFrame(columns=data.columns, dtype=np.int32)

    # 设置分割比例
    percent = 1 / float(k)

    # 获取用户列表
    set_users = set(list(data['user_id']))

    # 数据分割
    for user in set_users:
        print('Split_k_fold, user_id:', user)
        data_block = data.loc[data['user_id'].isin([user])]
        n = data_block.values.shape[0]
        num_rating = math.ceil(n * percent)
        if num_rating > n:
            num_rating = n
        total_item = list(data_block.index)
        for i in range(k):
            if total_item != 0:
                if i == k - 1 or len(total_item) <= num_rating:
                    choose_item = total_item
                else:
                    choose_item = sample(total_item, num_rating)
                total_item = [x for x in total_item if x not in choose_item]
                # 数据分割
                data_slice = data_block.loc[choose_item, :]
                L_data['data' + str(i)] = pd.concat([L_data['data' + str(i)], data_slice])
    # 数据存储
    for i in range(k):
        L_data['data' + str(i)].to_csv(k_path + str(i) + '.csv', index=False)


# 对k折交叉验证数据进行组合
def get_data_k_fold(path, test_id, k):
    data_train = None
    data_test = None
    print('===================数据拼接:' + str(test_id) + '===================')
    # 读取数据
    flag = 0
    for i in range(k):
        if i == test_id:
            data_test = pd.read_csv(path + str(i) + '.csv')
        if flag == 0:
            data_train = pd.read_csv(path + str(i) + '.csv')
        else:
            data = pd.read_csv(path + str(i) + '.csv')
            data_train = pd.concat([data_train, data])

    # 数据补全
    # 补全空行
    train_user = data_train['user_id']
    test_user = data_test['user_id']

    train_user = list(train_user)
    test_user = list(test_user)

    train_user = set(train_user)
    test_user = set(test_user)

    # 获取测试集含有训练集不含有的用户集合
    set_u = test_user - train_user

    if len(set_u) != 0:
        for u in set_u:
            data_fill = data_test.loc[data_test['user_id'].isin([u])]
            n = data_fill.values.shape[0]
            pos = np.random.randint(0, n)
            index = data_fill.index[pos]
            data_slice = pd.DataFrame({'user_id': [data_fill.loc[index, 'user_id']],
                                       'item_id': [data_fill.loc[index, 'item_id']],
                                       'score': [data_fill.loc[index, 'score']]})
            data_train = pd.concat([data_train, data_slice])
            data_test = data_test.drop(index=index)

    # 补全空列
    train_item = data_train['item_id']
    test_item = data_test['item_id']

    train_item = list(train_item)
    test_item = list(test_item)

    train_item = set(train_item)
    test_item = set(test_item)

    # 获取测试集含有训练集不含有的用户集合
    set_i = test_item - train_item

    if len(set_i) != 0:
        for i in set_i:
            data_fill = data_test.loc[data_test['item_id'].isin([i])]
            n = data_fill.values.shape[0]
            pos = np.random.randint(0, n)
            index = data_fill.index[pos]
            data_slice = pd.DataFrame({'user_id': [data_fill.loc[index, 'user_id']],
                                       'item_id': [data_fill.loc[index, 'item_id']],
                                       'score': [data_fill.loc[index, 'score']]})
            data_train = pd.concat([data_train, data_slice])
            data_test = data_test.drop(index=index)
    data_is_ok(data_train, data_test)

    print('数据拼接完成:' + str(test_id) + '！')

    # 创建评分矩阵数据以及mask
    print('===================创建评分矩阵和mask矩阵===================')
    # train
    print('训练集创建......')
    m_train = create_rating_matrix(data_train)
    m_mask_train = m_train.mask(m_train != 0, 1)
    print('训练集创建完成！')

    # test
    print('测试集创建......')
    m_test = pd.DataFrame(index=m_train.index, columns=m_train.columns, dtype='int')
    m_test = m_test.fillna(0)
    for row in data_test.itertuples():
        user_id = row.user_id
        item_id = row.item_id
        score = row.score
        m_test.loc[user_id, item_id] = score
    m_mask_test = m_test.mask(m_test != 0, 1)
    print('测试集创建完成！')

    m_train = m_train.values
    m_test = m_test.values
    m_mask_train = m_mask_train.values
    m_mask_test = m_mask_test.values

    return m_train, m_mask_train, m_test, m_mask_test


# 获取数据
def get_data(path, ratio, domain_name):
    # 加载数据
    # print('===================加载数据===================')
    # path_data = path + 'ratings.dat'
    # print('Load data from ' + path_data + '......')
    # names_col = ['user_id', 'item_id', 'score', 'time']
    # rating_triple = pd.read_table(path_data, sep='::', header=None, names=names_col, engine='python')
    # rating_triple = rating_triple.loc[:, ['user_id', 'item_id', 'score']]
    # 将三元组数转存为csv格式
    path_rating_triple = path + domain_name + '.csv'
    # rating_triple.to_csv(path_rating_triple, index=False)
    print('加载完成！')
    # 数据集分割
    path_train_triple = path + 'train_triple.csv'
    path_test_triple = path + 'test_triple.csv'
    print('===================分割数据===================')
    train_data_triple, test_data_triple = split_triple(path_rating_triple, ratio, path_train_triple, path_test_triple)
    print('分割完成！')

    # 创建评分矩阵数据以及mask
    print('===================创建评分矩阵和mask矩阵===================')
    # train
    print('训练集创建......')
    m_train = create_rating_matrix(train_data_triple)
    m_mask_train = m_train.mask(m_train != 0, 1)
    print('训练集创建完成！')

    # test
    print('测试集创建......')
    m_test = pd.DataFrame(index=m_train.index, columns=m_train.columns, dtype='int')
    m_test = m_test.fillna(0)
    for row in test_data_triple.itertuples():
        user_id = row.user_id
        item_id = row.item_id
        score = row.score
        m_test.loc[user_id, item_id] = score
    m_mask_test = m_test.mask(m_test != 0, 1)
    print('测试集创建完成！')

    m_train = m_train.values
    m_test = m_test.values
    m_mask_train = m_mask_train.values
    m_mask_test = m_mask_test.values

    return m_train, m_mask_train, m_test, m_mask_test
