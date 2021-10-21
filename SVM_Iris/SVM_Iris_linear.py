import numpy as np
import pandas as pd
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error, recall_score
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_classification


# 输出结果
def plot_res(x_test, y_pred):
    t1 = np.array([[0, 0]])
    t2 = np.array([[0, 0]])
    t3 = np.array([[0, 0]])

    for i in range(x_test.shape[0]):
        if y_pred[i] == 0:
            t1 = np.append(t1, x_test[i, :].reshape(1, -1), axis=0)
        elif y_pred[i] == 1:
            t2 = np.append(t2, x_test[i, :].reshape(1, -1), axis=0)
        else:
            t3 = np.append(t3, x_test[i, :].reshape(1, -1), axis=0)

    t1 = t1[1:, :]
    t2 = t2[1:, :]
    t3 = t3[1:, :]

    # 绘制生成的数据
    plt.figure(figsize=(6, 5))

    plt.scatter(t1[:, 0], t1[:, 1], c='y')
    plt.scatter(t2[:, 0], t2[:, 1], c='r')
    plt.scatter(t3[:, 0], t3[:, 1], c='b')

    plt.title('Test Dataset')
    plt.xlabel('f1')
    plt.ylabel('f2')
    plt.savefig('res/linear/Test Dataset.png', dpi=300)
    plt.show()


# 绘制超平面
def plot_hyperplane(clf, x_test, y_pred, state):
    # 画出建立的超平面
    w = clf.coef_[0]  # 取得w值，w中是二维的
    a = -w[0] / w[1]  # 计算直线斜率
    xx = np.linspace(-4, 4)  # 随机产生连续x值
    yy = a * xx - (clf.intercept_[0]) / w[1]  # 根据随机x得到y值

    # 计算与直线相平行的两条直线
    b = clf.support_vectors_[0]
    yy_down = a * xx + (b[1] - a * b[0])
    b = clf.support_vectors_[-1]
    yy_up = a * xx + (b[1] - a * b[0])

    print('w:', w)
    print('a:', a)
    print('support_vectors:', clf.support_vectors_)
    print('clf.coef_', clf.coef_)

    plt.figure(figsize=(6, 5))

    # 画出三条直线
    plt.plot(xx, yy, 'k-')
    plt.plot(xx, yy_down, 'k--')
    plt.plot(xx, yy_up, 'k--')

    plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100, c="g")
    plt.scatter(x_test[:, 0], x_test[:, 1], c=y_pred)

    plt.axis('tight')

    if state:
        plt.title('The bold circle is the support vector')
        plt.savefig('res/linear/Linear_Boundary.png', dpi=300)
    else:
        plt.title('Test Res')
        plt.savefig('res/linear/Test_Res.png', dpi=300)
    plt.show()


if __name__ == '__main__':

    X = np.r_[np.random.randn(30, 2) - [2, 2], np.random.randn(30, 2) + [2, 2]]  # 随机生成左下方20个点，右上方20个点
    Y = [0] * 30 + [1] * 30

    print(X)
    print(Y)

    plt.figure(figsize=(6, 5))

    plt.scatter(X[:, 0], X[:, 1], c=Y)

    plt.title('Dataset')
    plt.xlabel('f1')
    plt.ylabel('f2')
    plt.savefig('res/linear/Dataset.png', dpi=300)
    plt.show()

    # split train_data and test_data
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=100)

    # create model
    model_linear = svm.SVC(kernel='linear', C=1)

    # training
    model_linear.fit(X_train, Y_train)

    plot_hyperplane(model_linear, X_train, Y_train, 1)

    plot_res(X_test, Y_test)

    # testing
    Y_pred = model_linear.predict(X_test)

    plot_hyperplane(model_linear, X_test, Y_pred, 0)

    # calculate accuracy
    accuracy_rate = accuracy_score(Y_test, Y_pred)
    rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))
    mae = mean_absolute_error(Y_test, Y_pred)
    recall = recall_score(Y_test, Y_pred, average='macro')
    print('accuracy_rate:%f, rmse:%f, mse:%f, recall:%f' % (accuracy_rate, rmse, mae, recall))

