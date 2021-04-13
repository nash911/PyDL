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
from pydl.nn.layers import NN
from pydl.training.training import Training
from pydl import conf

def main():
    mnist = fetch_openml('mnist_784')
    X = np.array(mnist.data, dtype=conf.dtype)
    y = np.array(mnist.target, dtype=np.int)
    K = np.max(y) + 1

    # plot first few images
    for i, r in enumerate(np.random.randint(0, y.size, 9)):
        # define subplot
        plt.subplot(330 + 1 + i)
        # plot raw pixel data
        plt.imshow(np.reshape(X[r], (-1, 28)), cmap=plt.get_cmap('gray'))
        plt.title(str(y[r]))

    # show the figure
    plt.show()

    weight_range = (-0.01, 0.01)
    l1 = FC(X, num_neurons=1000, bias=True, weight_range=weight_range, activation_fn='ReLU')
    l2 = FC(l1, num_neurons=K, bias=True, weight_range=weight_range, activation_fn='SoftMax')
    layers = [l1, l2]

    nn = NN(X, layers)
    train = Training(nn, step_size=1e-2, reg_lambda=1e-1, train_size=60000, test_size=10000)
    train.train(X, y, normalize=True, shuffle=False, epochs=10000, plot=True, log_freq=1)

    input("Press Enter to continue...")


if __name__ == '__main__':
    main()
