import numpy as np
import pandas as pd
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error, recall_score


# 交叉验证
def cross_validate(k):
    kf = KFold(n_splits=k)
    C = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
    Gamma = [0.1, 0.3, 0.5, 0.7, 0.9, 1]
    li = ['C', 'Gamma', 'mae', 'acc']
    res = pd.DataFrame(columns=li)
    res.to_csv('cross_validation_res.csv', index=False)
    for c in C:
        for gamma in Gamma:
            MAE = 0
            ACC = 0
            for train_index, test_index in kf.split(X_train):
                # 分割数据
                train_x, train_y = X_train[train_index], Y_train[train_index]
                test_x, test_y = X_train[test_index], Y_train[test_index]

                # 创建模型
                model = make_pipeline(StandardScaler(), svm.SVC(C=c, gamma=gamma))

                # 训练模型
                model.fit(train_x, train_y)

                # 测试模型
                y_pred = model.predict(test_x)

                # 模型评价计算
                MAE += mean_absolute_error(test_y, y_pred)
                ACC += accuracy_score(test_y, y_pred)

            MAE = float(MAE)/k
            ACC /= k
            print('C: %.2f, Gamma: %.2f, mae: %.4f, acc: %.4f' % (c, gamma, MAE, ACC))
            # 保存本轮结果
            result = pd.DataFrame({'C': [c], 'Gamma': [gamma], 'mae': [MAE], 'acc': [ACC]})
            result.to_csv('cross_validation_res.csv', index=False, header=False, mode='a')


if __name__ == '__main__':
    # load data
    iris_data = datasets.load_iris()
    X = iris_data.data
    Y = iris_data.target

    # split train_data and test_data
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=100)

    # k折交叉验证
    # cross_validate(5)

    # create model
    # model_linear = make_pipeline(StandardScaler(), svm.SVC(kernel='linear', C=1))
    model_rbf = make_pipeline(StandardScaler(), svm.SVC(C=3, gamma=0.3))

    # training
    # model_linear.fit(X_train, Y_train)
    model_rbf.fit(X_train, Y_train)

    # testing
    Y_pred = model_rbf.predict(X_test)

    # calculate accuracy
    accuracy_rate = accuracy_score(Y_test, Y_pred)
    rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))
    mae = mean_absolute_error(Y_test, Y_pred)
    recall = recall_score(Y_test, Y_pred, average='macro')
    print('accuracy_rate:%f, rmse:%f, mse:%f, recall:%f' % (accuracy_rate, rmse, mae, recall))
