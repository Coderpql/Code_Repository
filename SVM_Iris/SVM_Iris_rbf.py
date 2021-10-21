import numpy as np
import pandas as pd
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error, recall_score
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_classification, make_gaussian_quantiles
import mpl_toolkits.mplot3d
import math


def make_meshgrid(x, y, h=0.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, x_, y_, **params):
    Z = clf.predict(np.c_[x_.ravel(), y_.ravel()])
    Z = Z.reshape(x_.shape)
    return ax.contourf(x_, y_, Z, **params)


# 生成数据
def create_data():
    x, y = make_gaussian_quantiles(n_samples=1000)

    # 绘制生成的数据
    plt.figure(figsize=(6, 5))
    plt.scatter(x[:, 0], x[:, 1], c=y)
    plt.title('Dataset')
    plt.xlabel('f1')
    plt.ylabel('f2')
    plt.savefig('res/rbf/Dataset.png', dpi=300)
    plt.show()

    # 数据存储
    y = y.reshape(-1, 1)
    dataset = np.append(x, y, axis=1)
    dataset = pd.DataFrame(dataset)
    dataset.to_csv('data/data.csv', index=False)


# 输出结果
def plot_res(clf, x_test, y_test, y_pred):
    t1 = np.array([[0, 0]])
    t2 = np.array([[0, 0]])
    t3 = np.array([[0, 0]])

    for i in range(x_test.shape[0]):
        if y_test[i] == 0:
            t1 = np.append(t1, x_test[i, :].reshape(1, -1), axis=0)
        elif y_test[i] == 1:
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
    plt.savefig('res/rbf/Test Dataset.png', dpi=300)
    plt.show()

    # 绘制决策边界
    plt.subplots_adjust(wspace=.4, hspace=.4)
    X0, X1 = X_test[:, 0], X_test[:, 1]
    xx, yy = make_meshgrid(X0, X1)

    ax = plt.subplot()
    plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(X0, X1, c=y_test, cmap=plt.cm.coolwarm, s=20, edgecolor='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel('f1')
    ax.set_ylabel('f2')
    ax.set_title('RBF Boundary')
    plt.savefig('res/rbf/RBF_Boundary.png', dpi=300)
    plt.show()


def plot_rbf(clf, x_test):
    X0, X1 = x_test[:, 0], x_test[:, 1]
    xx, yy = make_meshgrid(X0, X1)
    ax = plt.subplot(111, projection='3d')
    # z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    z = (1 / 2 * math.pi * 3 ** 2) * np.exp(-(xx ** 2 + yy ** 2) / 2 * 3 ** 2)
    # z = np.exp(-(xx - yy) ** 2 / 2 * 3 ** 2)
    z = z.reshape(xx.shape)
    ax.plot_surface(xx, yy, z, rstride=1, cstride=1, cmap='rainbow', alpha=0.9)  # 绘面
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.savefig('res/rbf/RBF.png', dpi=300)
    plt.show()


if __name__ == '__main__':
    # create data
    create_data()
    # load data
    data = pd.read_csv('data/data.csv').values
    X = data[:, :-1]
    Y = data[:, -1:].ravel()
    Y = Y.astype('int')

    # split train_data and test_data
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=100)

    # create model
    model_rbf = svm.SVC(C=3, gamma=3)

    # training
    model_rbf.fit(X_train, Y_train)

    # testing
    Y_pred = model_rbf.predict(X_test)

    # calculate accuracy
    accuracy_rate = accuracy_score(Y_test, Y_pred)
    rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))
    mae = mean_absolute_error(Y_test, Y_pred)
    recall = recall_score(Y_test, Y_pred, average='macro')
    print('accuracy_rate:%f, rmse:%f, mae:%f, recall:%f' % (accuracy_rate, rmse, mae, recall))

    plot_res(model_rbf, X_test, Y_test, Y_pred)
    plot_rbf(model_rbf, X_test)
