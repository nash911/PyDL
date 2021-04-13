# ------------------------------------------------------------------------
# MIT License
#
# Copyright (c) [2021] [Avinash Ranganath]
#
# This code is part of the library PyDL <https://github.com/nash911/PyDL>
# This code is licensed under MIT license (see LICENSE.txt for details)
# ------------------------------------------------------------------------

import numpy as np
from sklearn import datasets

from pydl.nn.layers import FC
from pydl.nn.layers import NN
from pydl.training.training import Training
from pydl import conf

def main():
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    K = np.max(y) + 1

    # Data Stats
    print("Data Size: ", X.shape[0])
    print("Feature Size: ", X.shape[1])
    print("Data Min: ", np.min(X, axis=0))
    print("Data Max: ", np.max(X, axis=0))
    print("Data Range: ", np.max(X, axis=0) - np.min(X, axis=0))
    print("Data Mean: ", np.mean(X, axis=0))
    print("Data STD: ", np.std(X, axis=0))

    # SoftMax Cross Entropy
    l1 = FC(X, num_neurons=int(X.shape[-1]*2), bias=True, weight_range=(-1, 1), activation_fn='ReLU')
    l2 = FC(l1, num_neurons=int(X.shape[-1]), bias=True, weight_range=(-1, 1), activation_fn='Tanh')
    l3_a = FC(l2, num_neurons=K, bias=True, weight_range=(-1, 1), activation_fn='SoftMax')
    layers = [l1, l2, l3_a]

    nn = NN(X, layers)
    train = Training(nn, step_size=1e-3, reg_lambda=1e-2)
    train.train(X, y, normalize=True, epochs=50000, y_onehot=False, plot=True)

    # Sigmoid Cross Entropy
    l3_b = FC(l2, num_neurons=K, bias=True, weight_range=(-1, 1), activation_fn='Sigmoid')
    layers = [l1, l2, l3_b]

    nn = NN(X, layers)
    train = Training(nn, step_size=1e-3, reg_lambda=1e-2)
    train.train(X, y, normalize=True, epochs=50000, y_onehot=False, plot=True)

    input("Press Enter to continue...")


if __name__ == '__main__':
    main()
