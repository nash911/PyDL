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
import matplotlib.pyplot as plt

from pydl import conf


class Training(ABC):
    """An abstract class defining the interface of Training Algorithms.
    Args:
        name (str): Name of the training algorithm.
    """

    def __init__(self, nn=None, step_size=1e-2, reg_lambda=1e-4, train_size=70, test_size=30,
                 name=None):
        self._nn = nn
        self._step_size = step_size
        self._lambda = reg_lambda
        self._train_size = train_size / 100.0
        self._test_size = test_size / 100.0
        self._name = name

        self._train_X =  self._train_y = self._test_X = self._test_y = None
        self._class_prob = None
        self._neg_ln_prob = None


    def split_data(self, X, y, train_size=None, test_size=None):
        sample_size = len(X)

        if train_size is None:
            train_size = self._train_size
        if test_size is None:
            test_size = self._test_size / 100.0

        order = np.random.permutation(sample_size)
        X = X[order]
        y = y[order]

        # Convert labels to one-hot vector
        if self._nn is None:
            y_onehot = np.zeros((y.size, y.max()+1))
        else:
            y_onehot = np.zeros((y.size, self._nn.num_classes))
        y_onehot[np.arange(y.size), y] = 1

        train_X = np.array(X[:int(train_size * sample_size)], dtype=conf.dtype)
        train_y = np.array(y_onehot[:int(train_size * sample_size)], dtype=conf.dtype)
        test_X = np.array(X[int(train_size * sample_size):], dtype=conf.dtype)
        test_y = np.array(y_onehot[int(train_size * sample_size):], dtype=conf.dtype)

        return train_X, train_y, test_X, test_y


    def loss(self, X, y, prob=None, summed=True):
        if prob is None:
            self._class_prob = self._nn.forward(X)
        else:
            # sum_probs = np.sum(prob, axis=-1, keepdims=True)
            # ones = np.ones_like(sum_probs)
            self._class_prob = prob

        # -ln(σ(z))
        self._neg_ln_prob = -np.log(self._class_prob)

        if self._nn is not None and self._lambda > 0:
            nn_weights = self._nn.weights
            regularization_loss = 0
            for w in nn_weights:
                regularization_loss += np.sum(w * w)
            regularization_loss *= (0.5 * self._lambda)
        else:
            regularization_loss = 0

        # Cross-Entropy Cost Fn.
        if summed:
            return (np.sum(y * self._neg_ln_prob) / y.shape[0]) + regularization_loss
        else:
            return np.sum(y * self._neg_ln_prob, axis=-1) + regularization_loss


    def loss_gradient(self, X, y, prob=None):
        if prob is not None:
            # sum_probs = np.sum(prob, axis=-1, keepdims=True)
            # ones = np.ones_like(sum_probs)
            self._class_prob = prob
        elif self._class_prob is None:
            self._class_prob = self._nn.forward(X)

        loss_grad = (y * (-1.0 / self._class_prob)) / y.shape[0]
        self._class_prob = None

        return loss_grad


    def evaluate(self, X, y):
        test_prob = self._nn.forward(X)
        pred = np.argmax(test_prob, axis=-1)
        pred_onehot = np.zeros((pred.size, self._nn.num_classes))
        pred_onehot[np.arange(pred.size), pred] = 1
        pred_diff = np.sum(np.fabs(y - pred_onehot), axis=-1) / 2.0
        accuracy = (1.0 - np.mean(pred_diff)) * 100.0
        return accuracy


    def learning_curve_plot(self, fig, axs, train_loss, test_loss, accuracy):
        axs[0].clear()
        axs[0].plot(list(range(len(train_loss))), train_loss, color='red', label='Train Loss')
        axs[0].plot(list(range(len(test_loss))), test_loss, color='blue', label='Test Loss')
        axs[0].set(ylabel='Loss')
        axs[0].legend(loc='upper right')

        axs[1].clear()
        axs[1].plot(list(range(len(accuracy))), accuracy, color='green', label='Test Accuracy')
        axs[1].set(ylabel='Accuracy(%)')
        axs[1].legend(loc='lower right')

        plt.show(block=False)
        plt.pause(0.01)


    def train(self, X, y, batch_size=256, epochs=100, plot=True):
        self._train_X, self._train_y, self._test_X, self._test_y = \
            self.split_data(X, y)
        num_batches = int(np.ceil(self._train_X.shape[0] /  batch_size))

        if plot:
            fig, axs = plt.subplots(2, sharey=False, sharex=True)
        train_loss = list()
        test_loss = list()
        accuracy = list()

        for e in range(epochs):
            for i in range(num_batches):
                start = int(batch_size * i)
                if i == num_batches-1:
                    end = self._train_X.shape[0]
                else:
                    end = start + batch_size

                train_l = self.loss(self._train_X[start:end], self._train_y[start:end])
                loss_grad = self.loss_gradient(self._train_X[start:end], self._train_y[start:end])
                _ = self._nn.backward(loss_grad, self._lambda)

                for l in self._nn.layers:
                    l.weights += -self._step_size * l.weights_grad
                    if l.bias is not None:
                        l.bias += -self._step_size * l.bias_grad

                if e % 1000 == 0:
                    test_l = self.loss(self._test_X, self._test_y)
                    accur = self.evaluate(self._test_X, self._test_y)
                    print("Epoch-%d - Training Loss: %.4f - Test Loss: %.4f - Accuracy: %.4f" %
                          (e, train_l, test_l, accur))

                    if plot:
                        train_loss.append(train_l)
                        test_loss.append(test_l)
                        accuracy.append(accur)
                        self.learning_curve_plot(fig, axs, train_loss, test_loss, accuracy)



class SGD(Training):
    def __init__(self, nn=None, name=None):
        super().__init__(nn=nn, name=name)
