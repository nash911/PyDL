# ------------------------------------------------------------------------
# MIT License
#
# Copyright (c) [2021] [Avinash Ranganath]
#
# This code is part of the library PyDL <https://github.com/nash911/PyDL>
# This code is licensed under MIT license (see LICENSE.txt for details)
# ------------------------------------------------------------------------

import numpy as np
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt

from pydl.nn.layers import FC
from pydl.nn.nn import NN
from pydl.training.sgd import SGD
from pydl import conf


def main():
    mnist = fetch_openml('mnist_784')
    X = np.array(mnist.data, dtype=conf.dtype)
    y = np.array(mnist.target, dtype=np.int)
    K = np.max(y) + 1

    # plot first few images
    fig = plt.figure()
    for i, r in enumerate(np.random.randint(0, y.size, 9)):
        # define subplot
        plt.subplot(330 + 1 + i)
        # plot raw pixel data
        plt.imshow(np.reshape(X[r], (-1, 28)), cmap=plt.get_cmap('gray'))
        plt.title(str(y[r]))

    # show the figure
    plt.draw()
    plt.waitforbuttonpress(10)
    plt.close(fig)

    weight_scale = 0.01
    l1 = FC(X, num_neurons=400, bias=True, weight_scale=weight_scale, xavier=True,
            activation_fn='ReLU')
    l2 = FC(l1, num_neurons=K, bias=True, weight_scale=weight_scale, xavier=True,
            activation_fn='SoftMax')
    layers = [l1, l2]

    nn = NN(X, layers)
    sgd = SGD(nn, step_size=1e-2, reg_lambda=1e-1, train_size=60000, test_size=10000)
    sgd.train(X, y, normalize='pca', dims=0.97, shuffle=False, epochs=10000, plot='MNIST - SGD',
              log_freq=1)

    input("Press Enter to continue...")


if __name__ == '__main__':
    main()
