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

    i = 0
    plt.sca(ax1)
    # 用不同的颜色表示不同数据
    plt.plot(range(mae_train.shape[0]), mae_train, color="red", linewidth=3, linestyle="-")
    plt.plot(range(mae_test.shape[0]), mae_test, color="blue", linewidth=3, linestyle="-")

    plt.xticks(fontsize=35)
    plt.yticks(fontsize=35)
    plt.title('MAE', fontsize=40)
    plt.xlabel('Number of Iterations', fontsize=40)
    plt.ylabel('MAE', fontsize=35)

    plt.savefig(name, dpi=600)
    plt.show()


if __name__ == '__main__':
    name = 'eval/MAE.png'
    mae_train = np.load('eval/mae_train.npy')
    mae_test = np.load('eval/mae_test.npy')
    CreatePic()
