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
from pydl.training.training import SGD
from pydl.training.training import Momentum
from pydl.training.training import RMSprop
from pydl.training.training import Adam
from pydl import conf

def main():
    breast_cancer = datasets.load_breast_cancer()
    X = breast_cancer.data
    y = breast_cancer.target
    y_onehot = np.zeros((y.size, 2))
    y_onehot[range(y.size), y] = 1
    K = np.max(y) + 1

    # Data Stats
    print("Data Size: ", X.shape[0])
    print("Feature Size: ", X.shape[1])
    print("Num. Classes: ", K)

    # Sigmoid Cross Entropy - Two Output Neurons
    l1 = FC(X, num_neurons=int(X.shape[-1]*2), bias=True, weight_scale=1.0, xavier=True,
            activation_fn='ReLU')
    l2 = FC(l1, num_neurons=int(X.shape[-1]), bias=True, weight_scale=1.0, xavier=True,
            activation_fn='Tanh')
    l3_a = FC(l2, num_neurons=2, bias=True, weight_scale=1.0, xavier=True, activation_fn='Sigmoid')
    layers = [l1, l2, l3_a]
    nn_a = NN(X, layers)

    # SGD
    sgd = SGD(nn_a, step_size=1e-3, reg_lambda=1e-2)
    sgd.train(X, y_onehot, normalize='mean', epochs=10000, y_onehot=False, plot='SGD')

    # RMSprop
    nn_a.reinitialize_network()
    rms = RMSprop(nn_a, step_size=1e-3, beta=0.9, reg_lambda=1e-2)
    rms.train(X, y, normalize='mean', epochs=10000, y_onehot=False, plot='RMSprop')


    # Sigmoid Cross Entropy - Single Output Neurons
    l1 = FC(X, num_neurons=int(X.shape[-1]*2), bias=True, weight_scale=1.0, xavier=True,
            activation_fn='ReLU')
    l2 = FC(l1, num_neurons=int(X.shape[-1]), bias=True, weight_scale=1.0, xavier=True,
            activation_fn='Tanh')
    l3_b = FC(l2, num_neurons=1, bias=True, weight_scale=1.0, xavier=True, activation_fn='Sigmoid')
    layers = [l1, l2, l3_b]
    nn_b = NN(X, layers)

    # Momentum
    momentum = Momentum(nn_b, step_size=1e-3, mu=0.5, reg_lambda=1e-2)
    momentum.train(X, y, normalize='mean', epochs=10000, y_onehot=False, plot='Momentum')

    # Adam
    nn_b.reinitialize_network()
    adam = Adam(nn_b, step_size=1e-3, beta_1=0.9, beta_2=0.999, reg_lambda=1e-2)
    adam.train(X, y_onehot, normalize='mean', epochs=10000, y_onehot=False, plot='Adam')

    input("Press Enter to continue...")


if __name__ == '__main__':
    main()
