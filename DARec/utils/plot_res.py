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
    plt.title(domain_name, fontsize=40)
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

    Domain_name = ['ratings_Apps_for_Android', 'ratings_Automotive', 'ratings_CDs_and_Vinyl', 'ratings_Movies_and_TV',
                   'ratings_Office_Products', 'ratings_Sports_and_Outdoors', 'ratings_Toys_and_Games',
                   'ratings_Video_Games']
    for domain_name in Domain_name:
        loss_name = '../res/log/AutoRec/' + domain_name + '_LOSS.pdf'
        # mae_name = '../res/log/AutoRec/' + domain_name + '_MAE.pdf'
        loss_train = np.load('../res/log/AutoRec/' + domain_name + '_train_loss.npy')
        # loss_test = np.load('../res/log/AutoRec/' + domain_name + '_test_loss.npy')
        # mae_train = np.load('../res/log/AutoRec/' + domain_name + '_train_mae.npy')
        # mae_test = np.load('../res/log/AutoRec/' + domain_name + '_test_mae.npy')
        CreatePic()
