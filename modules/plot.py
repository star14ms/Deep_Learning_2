import matplotlib.pyplot as plt
import numpy as np


def plot_loss_graph(train_loss, smooth=True, ylim=10):
    x = np.arange(len(train_loss))

    if smooth:
        plt.plot(x, smooth_curve(train_loss), f"-", label="loss")
    else:
        plt.plot(x, train_loss, f"-", label="loss")

    plt.xlabel("학습한 문장 수 (iterations)")
    plt.ylabel("손실 (Loss)")
    plt.ylim(0, ylim)
    plt.legend(loc='upper right')


def smooth_curve(x):
    """損失関数のグラフを滑らかにするために用いる

    参考：http://glowingpython.blogspot.jp/2012/02/convolution-with-numpy.html
    """
    window_len = 11
    s = np.r_[x[window_len-1:0:-1], x, x[-1:-window_len:-1]]
    w = np.kaiser(window_len, 2)
    y = np.convolve(w/w.sum(), s, mode='valid')
    return y[5:len(y)-5]