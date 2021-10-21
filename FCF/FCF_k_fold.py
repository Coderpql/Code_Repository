import numpy as np
import pandas as pd


# 对k折交叉验证数据进行组合
def get_data(test_id, k):
    data_train = None
    data_test = None
    # 读取数据
    flag = 0
    for i in range(k):
        if i == test_id:
            data_test = pd.read_csv('data/ml-1m/k_fold/data_' + str(i) + '.csv')
        if flag == 0:
            data_train = pd.read_csv('data/ml-1m/k_fold/data_' + str(i) + '.csv')
        else:
            data = pd.read_csv('data/ml-1m/k_fold/data_' + str(i) + '.csv')
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


if __name__ == '__main__':
    for test_i in range(5):
        train_data, test_data = get_data(test_i, 5)
        print('test_id', test_i)
        print(train_data.values.shape)
        print(test_data.values.shape)
