# ------------------------------------------------------------------------
# MIT License
#
# Copyright (c) [2021] [Avinash Ranganath]
#
# This code is part of the library PyDL <https://github.com/nash911/PyDL>
# This code is licensed under MIT license (see LICENSE.txt for details)
# ------------------------------------------------------------------------

from abc import ABC, abstractmethod
import numpy as np

from pydl import conf


class Training(ABC):
    """An abstract class defining the interface of Training Algorithms.
    Args:
        name (str): Name of the training algorithm.
    """

    def __init__(self, nn=None, name=None):
        self._nn = nn
        self._name = name

        self._train_X =  self._train_y = self._test_X = self._test_y = None
        self._neg_ln_prob = None


    def split_data(self, X, y, train_size=70, test_size=30):
        sample_size = len(X)
        train_size = train_size / 100.0
        test_size = test_size / 100.0

        np.random.seed(0)
        order = np.random.permutation(sample_size)
        X = X[order]
        y = y[order]

        # Convert labels to one-hot vector
        y_onehot = np.zeros((y.size, y.max()+1))
        y_onehot[np.arange(y.size), y] = 1

        train_X = np.array(X[:int(train_size * sample_size)], dtype=conf.dtype)
        train_y = np.array(y_onehot[:int(train_size * sample_size)], dtype=conf.dtype)
        test_X = np.array(X[int(test_size * sample_size):], dtype=conf.dtype)
        test_y = np.array(y_onehot[int(test_size * sample_size):], dtype=conf.dtype)

        return train_X, train_y, test_X, test_y


    def loss(self, X, y, prob=None, summed=True):
        if prob is None:
            class_prob = self._nn.forward(X)
        else:
            # sum_probs = np.sum(prob, axis=-1, keepdims=True)
            # ones = np.ones_like(sum_probs)
            class_prob = prob

        # -ln(σ(z))
        self._neg_ln_prob = -np.log(class_prob)

        # Cross-Entropy Cost Fn.
        if summed:
            return np.sum(y * self._neg_ln_prob) / y.shape[0]
        else:
            return np.sum(y * self._neg_ln_prob, axis=-1)


    def loss_gradient(self, X, y, prob=None):
        if prob is None:
            class_prob = self._nn.forward(X)
        else:
            # sum_probs = np.sum(prob, axis=-1, keepdims=True)
            # ones = np.ones_like(sum_probs)
            class_prob = prob

        return (y * (-1.0 / class_prob)) / y.shape[0]


    def train(self, X, y):
        self._train_X, self._train_y, self._test_X, self._test_y = self.split_data(X, y)


class SGD(Training):
    def __init__(self, nn=None, name=None):
        super().__init__(nn=nn, name=name)
