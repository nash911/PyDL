# ------------------------------------------------------------------------
# MIT License
#
# Copyright (c) [2021] [Avinash Ranganath]
#
# This code is part of the library PyDL <https://github.com/nash911/PyDL>
# This code is licensed under MIT license (see LICENSE.txt for details)
# ------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

from pydl.nn.layers import FC
from pydl.nn.layers import NN
from pydl.training.training import SGD
from pydl.training.training import Momentum
from pydl.training.training import RMSprop
from pydl import conf

def main():
    N = 100 # number of points per class
    D = 2 # dimensionality
    K = 3 # number of classes
    X = np.zeros((N*K,D)) # data matrix (each row = single example)
    y = np.zeros(N*K, dtype='uint8') # class labels
    for j in range(K):
        ix = range(N*j,N*(j+1))
        r = np.linspace(0.0,1,N) # radius
        t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        y[ix] = j

    # Data Stats
    print("Data Size: ", X.shape[0])
    print("Feature Size: ", X.shape[1])
    print("Data Min: ", np.min(X, axis=0))
    print("Data Max: ", np.max(X, axis=0))
    print("Data Range: ", np.max(X, axis=0) - np.min(X, axis=0))
    print("Data Mean: ", np.mean(X, axis=0))
    print("Data STD: ", np.std(X, axis=0))

    # Visualize Data:
    fig = plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
    plt.draw()
    plt.waitforbuttonpress(0)
    plt.close(fig)

    # SoftMax Cross Entropy - SGD
    l1_a = FC(X, num_neurons=int(100), bias=True, activation_fn='ReLU')
    l2_a = FC(l1_a, num_neurons=K, bias=True, activation_fn='SoftMax')
    layers_a = [l1_a, l2_a]

    nn_a = NN(X, layers_a)
    sgd = SGD(nn_a, step_size=1e-2, reg_lambda=1e-3)
    sgd.train(X, y, normalize='mean', batch_size=256, epochs=20000, y_onehot=False,
              plot='SGD - Softmax')

    # Sigmoid Cross Entropy - Momentum
    l1_b = FC(X, num_neurons=int(100), bias=True, activation_fn='Tanh')
    l2_b = FC(l1_b, num_neurons=K, bias=True, activation_fn='Sigmoid')
    layers_b = [l1_b, l2_b]

    nn_b = NN(X, layers_b)
    momentum = Momentum(nn_b, mu=0.5, step_size=1e-2, reg_lambda=1e-3)
    momentum.train(X, y, batch_size=256, epochs=20000, y_onehot=False, plot='Momentum - Sigmoid')

    # Sigmoid Cross Entropy - RMSprop
    l1_c = FC(X, num_neurons=int(100), bias=True, activation_fn='ReLU')
    l2_c = FC(l1_c, num_neurons=K, bias=True, activation_fn='Sigmoid')
    layers_c = [l1_c, l2_c]

    nn_c = NN(X, layers_c)
    rms = RMSprop(nn_c, beta=0.9, step_size=1e-2, reg_lambda=1e-3)
    rms.train(X, y, batch_size=256, epochs=20000, y_onehot=False, plot='RMSprop - Sigmoid')

    input("Press Enter to continue...")


if __name__ == '__main__':
    main()
