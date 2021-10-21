import pandas as pd
import numpy as np
import math
from random import sample

np.random.seed(100)


# 加载数据
def load_data(path):
    names_col = ['user_id', 'item_id', 'score', 'time']
    df_ratings = pd.read_csv(path, names=names_col)
    df_ratings = df_ratings.loc[:, ['user_id', 'item_id', 'score']]
    return df_ratings


# 筛选数据
def data_filter(data, num_rating):
    data_group = data.groupby(['user_id']).count().item_id
    list_user = []
    for i in data_group.keys():
        if data_group[i] > num_rating:
            list_user.append(i)
    data = data.loc[data['user_id'].isin(list_user)]
    return data


# 训练集测试集分割(三元组)
def split_triple(data, percent, train_path, test_path):
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


# 获取目标域辅助域共同用户
def get_common_user(l_data):
    user_set = None
    flag = 0
    for data in l_data:
        users = set(data['user_id'].tolist())
        if flag == 0:
            user_set = users
        else:
            user_set = user_set.intersection(users)
        flag += 1
    print('共同用户数：', len(user_set))
    return user_set


def process_CD_data(path_data, combinations):
    for comb in combinations:
        print(comb)
        target = comb[0]
        source = comb[1]
        path_target = path_data + target + '.csv'
        path_source = path_data + source + '.csv'
        # 数据加载与筛选
        names_col = ['user_id', 'item_id', 'score', 'time']
        data_target = pd.read_csv(path_target, names=names_col)
        data_target = data_filter(data_target, 5)
        data_source = pd.read_csv(path_source, names=names_col)
        data_source = data_filter(data_source, 5)
        l_data = [data_target, data_source]
        # 获取共同用户
        common_users = list(get_common_user(l_data))
        # 按照共同用户筛选数据
        data_target = data_target.loc[data_target['user_id'].isin(list(common_users))]
        data_source = data_source.loc[data_source['user_id'].isin(list(common_users))]
        data_target.to_csv(path_data + target + '_.csv', index=False, header=False)
        data_source.to_csv(path_data + source + '_.csv', index=False, header=False)


# 获取数据
def get_data_autoRec(path_data, path, ratio, domain, is_process=False, mode='train'):
    path_train_triple = path + 'train/' + domain + '_train_triple.csv'
    path_test_triple = path + 'test/' + domain + '_test_triple.csv'

    if is_process:
        # 加载数据
        print('===================加载数据===================')
        rating_data_triple = load_data(path_data)
        # 数据筛选 规则：每个用户至少观看5部电影
        # rating_data_triple = data_filter(rating_data_triple, 5)
        print('加载完成！')
        # 数据集分割
        print('===================分割数据===================')
        train_data_triple, test_data_triple = split_triple(rating_data_triple, ratio, path_train_triple,
                                                           path_test_triple)
        print('分割完成！')

    else:
        train_data_triple = pd.read_csv(path_train_triple)
        test_data_triple = pd.read_csv(path_train_triple)

    # 创建评分矩阵数据以及mask
    print('===================创建评分矩阵和mask矩阵===================')
    # train
    print('训练集创建......')
    m_train = create_rating_matrix(train_data_triple)
    print('训练集创建完成！')

    # test
    print('测试集创建......')
    m_test = pd.DataFrame(index=list(set(list(test_data_triple['user_id']))), columns=m_train.columns, dtype='int')
    m_test = m_test.fillna(0)
    for row in test_data_triple.itertuples():
        user_id = row.user_id
        item_id = row.item_id
        score = row.score
        m_test.loc[user_id, item_id] = score
    print('测试集创建完成！')

    m_train = m_train.values
    m_test = m_test.values

    if mode == 'train':
        print(m_train.shape)
        return m_train
    else:
        print(m_test.shape)
        return m_test


# 获取数据
def get_data_UDARec(path_data, path, target, source, ratio, mode='train', is_process=False):
    path_train_triple = path + 'target/train/' + target + '_train_triple.csv'
    path_test_triple = path + 'target/test/' + target + '_test_triple.csv'
    path_source_triple = path + 'source/' + source + '_triple.csv'
    path_target = path_data + target + '_.csv'
    path_source = path_data + source + '_.csv'

    if is_process:
        # 数据加载
        data_target = load_data(path_target)
        data_source = load_data(path_source)
        # 数据集分割
        print('===================分割数据===================')
        train_target_triple, test_target_triple = split_triple(data_target, ratio, path_train_triple, path_test_triple)
        data_source.to_csv(path_source_triple, index=False)
    else:
        # 数据加载
        train_target_triple = pd.read_csv(path_train_triple)
        test_target_triple = pd.read_csv(path_test_triple)
        data_source = pd.read_csv(path_source_triple)

    # 创建评分矩阵数据以及mask
    print('===================创建评分矩阵和mask矩阵===================')
    # train
    print('训练集创建......')
    m_target_train = create_rating_matrix(train_target_triple)
    m_source_train = create_rating_matrix(data_source)
    print('训练集创建完成！')

    # test
    print('测试集创建......')
    m_target_test = pd.DataFrame(index=list(set(test_target_triple['user_id'].tolist())), columns=m_target_train.columns, dtype='int')
    m_target_test = m_target_test.fillna(0)
    for row in test_target_triple.itertuples():
        user_id = row.user_id
        item_id = row.item_id
        score = row.score
        m_target_test.loc[user_id, item_id] = score
    print('测试集创建完成！')

    # 添加目标域辅助域标识
    m_target_train['flag'] = np.ones((len(m_target_train.index), 1))
    m_target_test['flag'] = np.ones((len(m_target_test.index), 1))
    m_source_train['flag'] = np.zeros((len(m_target_train.index), 1))

    m_target_train = m_target_train.values
    m_target_test = m_target_test.values
    m_source_train = m_source_train.values

    if mode == 'train':
        return m_target_train, m_source_train
    else:
        return m_target_test, m_source_train
