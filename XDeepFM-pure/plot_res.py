import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np

config = {
    "font.family": 'Times New Roman',  # 设置字体类型
}
rcParams.update(config)


def CreatePic():

    # 通用设置
    matplotlib.rc('axes', facecolor='white')
    matplotlib.rc('figure', figsize=(20, 12))
    matplotlib.rc('axes', grid=False)

    # 创建图形
    plt.figure(1)
    ax1 = plt.subplot(1, 1, 1)

    plt.sca(ax1)
    # 用不同的颜色表示不同数据
    plt.plot(range(loss_train.shape[0]), loss_train, color="red", linewidth=3, linestyle="-", label='train_loss')
    # plt.plot(range(loss_test.shape[0]), loss_test, color="blue", linewidth=3, linestyle="-", label='test_loss')

    plt.xticks(fontsize=35)
    plt.yticks(fontsize=35)
    plt.title('LOSS', fontsize=40)
    plt.xlabel('Number of Iterations', fontsize=40)
    plt.ylabel('Loss', fontsize=35)
    plt.legend(loc="best", fontsize=35)
    plt.savefig(loss_name, dpi=600)
    plt.show()

    # plt.figure(2)
    # ax2 = plt.subplot(1, 1, 1)
    #
    # plt.sca(ax2)
    # # 用不同的颜色表示不同数据
    # plt.plot(range(mae_train.shape[0]), mae_train, color="red", linewidth=3, linestyle="-", label='train_mae')
    # plt.plot(range(mae_test.shape[0]), mae_test, color="blue", linewidth=3, linestyle="-", label='test_mae')
    #
    # plt.xticks(fontsize=35)
    # plt.yticks(fontsize=35)
    # plt.title('MAE', fontsize=40)
    # plt.xlabel('Number of Iterations', fontsize=40)
    # plt.ylabel('MAE', fontsize=35)
    # plt.legend(loc="best", fontsize=35)
    #
    # plt.savefig(mae_name, dpi=600)
    # plt.show()


if __name__ == '__main__':
    loss_name = 'log/LOSS.png'
    # mae_name = 'data/amazon/correlated/Cell_Phones_and_Accessories/log/MAE.png'
    loss_train = np.load('log/train_loss.npy')
    # loss_test = np.load('data/amazon/correlated/Cell_Phones_and_Accessories/log/test_loss.npy')
    # mae_train = np.load('data/amazon/correlated/' + domain_name + '/log/train_mae.npy')
    # mae_test = np.load('data/amazon/correlated/Cell_Phones_and_Accessories/log/test_mae.npy')
    CreatePic()
