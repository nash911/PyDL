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
import sys
import time

from pydl import conf


class Training(ABC):
    """An abstract class defining the interface of Training Algorithms.
    Args:
        name (str): Name of the training algorithm.
    """

    def __init__(self, nn=None, step_size=1e-2, reg_lambda=1e-4, train_size=70, test_size=30,
                 activatin_type=None, name=None):
        self._nn = nn
        self._step_size = step_size
        self._lambda = reg_lambda
        self._name = name

        if activatin_type is not None:
            self._activatin_type = activatin_type
        elif nn is not None:
            self._activatin_type = self._nn.layers[-1].activation
        else:
            sys.exit("Error: Please specify activation_type for instance of class Training")

        if train_size + test_size == 100:
            self._train_size = train_size / 100.0
            self._test_size = test_size / 100.0
        else:
            self._train_size = int(train_size)
            self._test_size = int(test_size)

        self._train_X =  self._train_y = self._test_X = self._test_y = None
        self._class_prob = None


    def mean_normalize(self, X, mean=None, std=None):
        if mean is None:
            mean = np.mean(X, axis=0, keepdims=True)

        if std is None:
            std = np.std(X, axis=0, keepdims=True)
            std[std == 0] = 1

        mean_centered = X - mean
        normalized = mean_centered/std
        return mean, std, normalized


    def reduce_data_dimensions(self, X, dims=None, mean=None, U=None, S=None, N=None, whiten=False):
        if mean is None:
            mean = np.mean(X, axis=0, keepdims=True)

        if N is None:
            N = X.shape[0]

        X -= mean
        cov = np.matmul(X.T, X) / N

        if U is None or S is None:
            U, S, V = np.linalg.svd(cov)

        if type(dims) is float and dims <= 1.0:
            cumulative_ratio = (np.cumsum(S)/np.sum(S)).tolist()
            min_ratio = min([i for i in cumulative_ratio if i > dims])
            dims = cumulative_ratio.index(min_ratio)
            print("PCA - Reduced to %d dimensions, while retaining %.2f percent information" %
                  (dims+1, (min_ratio)*100.0))
            dims += 1
        elif type(dims) is int and dims >= 1:
            print("PCA - Reduced to %d dimensions, while retaining %.2f percent information" %
                  (dims, ((np.cumsum(S)/np.sum(S))[dims-1])*100.0))


        # Project data on to orthogonal subspace
        X_reduced = np.matmul(X, U[:, :dims])

        if whiten:
            X_reduced = X_reduced / np.sqrt(S[:dims] + 1e-4)

        return mean, U, S, dims, X_reduced


    def shuffle_split_data(self, X, y, shuffle=True, train_size=None, test_size=None,
                           y_onehot=False):
        sample_size = len(X)

        if train_size is None:
            train_size = self._train_size
        else:
            train_size = train_size / 100.0
        if test_size is None:
            test_size = self._test_size
        else:
            test_size = test_size / 100.0

        if shuffle:
            order = np.random.permutation(sample_size)
            X = X[order]
            y = y[order]

        if y_onehot and len(y.shape) == 1:
            # Convert labels to one-hot vector
            if self._nn is None:
                labels = np.zeros((y.size, y.max()+1))
            else:
                labels = np.zeros((y.size, self._nn.num_classes))
            labels[np.arange(y.size), y] = 1
        else:
            labels = y

        split_idx = int(train_size * sample_size) if train_size < 1 else train_size

        train_X = np.array(X[:split_idx], dtype=conf.dtype)
        train_y = np.array(labels[:split_idx], dtype=np.int32)
        test_X = np.array(X[split_idx:], dtype=conf.dtype)
        test_y = np.array(labels[split_idx:], dtype=np.int32)

        return train_X, train_y, test_X, test_y


    def loss(self, X, y, prob=None):
        if 'softmax' in self._activatin_type.lower():
            return self.softmax_cross_entropy_loss(X, y, prob)
        elif 'sigmoid' in self._activatin_type.lower():
            return self.sigmoid_cross_entropy_loss(X, y, prob)
        else:
            sys.exit("Error: Unknown activation_type: ", self._activation_fn)


    def loss_gradient(self, X, y, prob=None):
        if 'softmax' in self._activatin_type.lower():
            return self.softmax_cross_entropy_gradient(X, y, prob)
        elif 'sigmoid' in self._activatin_type.lower():
            return self.sigmoid_cross_entropy_gradient(X, y, prob)
        else:
            sys.exit("Error: Unknown activation_type: ", self._activation_fn)


    def softmax_cross_entropy_loss(self, X, y, prob=None):
        if prob is None:
            self._class_prob = self._nn.forward(X)
        else:
            # sum_probs = np.sum(prob, axis=-1, keepdims=True)
            # ones = np.ones_like(sum_probs)
            self._class_prob = prob

        # -ln(σ(z))
        neg_ln_prob = -np.log(self._class_prob)

        if self._nn is not None and self._lambda > 0:
            nn_weights = self._nn.weights
            regularization_loss = 0
            for w in nn_weights:
                regularization_loss += np.sum(w * w)
            regularization_loss *= (0.5 * self._lambda)
        else:
            regularization_loss = 0

        if len(y.shape) == 2: # y --> OneHot Representation
            # Convert y to class labels
            y = np.argmax(y, axis=-1)

        # Cross-Entropy Cost Fn.
        return np.mean(neg_ln_prob[range(y.size), y]) + regularization_loss


    def softmax_cross_entropy_gradient(self, X, y, prob=None):
        if prob is not None:
            # sum_probs = np.sum(prob, axis=-1, keepdims=True)
            # ones = np.ones_like(sum_probs)
            self._class_prob = prob
        elif self._class_prob is None:
            self._class_prob = self._nn.forward(X)

        if len(y.shape) == 2: # y --> OneHot Representation
            # Convert y to class labels
            y = np.argmax(y, axis=-1)

        loss_grad = np.zeros_like(self._class_prob)
        loss_grad[range(y.size), y] = (-1.0 / self._class_prob[range(y.size), y]) / y.shape[0]

        self._class_prob = None
        return loss_grad


    def sigmoid_cross_entropy_loss(self, X, y, prob=None):
        if prob is None:
            self._class_prob = self._nn.forward(X)
        else:
            self._class_prob = prob

        with np.errstate(divide='ignore'):
            # -ln(σ(z))
            neg_ln_prob = -np.log(self._class_prob)

            # -ln(1 - σ(z))
            neg_ln_one_mns_prob = -np.log(1.0 - self._class_prob)

        if len(y.shape) == 1: # y --> Class labels
            neg_ln_one_mns_prob[range(y.size), y] = neg_ln_prob[range(y.size), y]
            logistic_probs = neg_ln_one_mns_prob
        elif len(y.shape) == 2: # y --> OneHot Representation
            neg_ln_prob = np.nan_to_num(neg_ln_prob)
            neg_ln_one_mns_prob = np.nan_to_num(neg_ln_one_mns_prob)
            neg_ln_prob *= y
            neg_ln_one_mns_prob *= (1.0 - y)
            logistic_probs = neg_ln_prob + neg_ln_one_mns_prob

        if self._nn is not None and self._lambda > 0:
            nn_weights = self._nn.weights
            regularization_loss = 0
            for w in nn_weights:
                regularization_loss += np.sum(w * w)
            regularization_loss *= (0.5 * self._lambda)
        else:
            regularization_loss = 0

        # Logistic Cost Fn.
        return (np.sum(logistic_probs) / y.shape[0]) + regularization_loss


    def sigmoid_cross_entropy_gradient(self, X, y, prob=None):
        if prob is not None:
            self._class_prob = prob
        elif self._class_prob is None:
            self._class_prob = self._nn.forward(X)

        if len(y.shape) == 1: # y --> Class labels
            loss_grad = (1.0 / (1.0 - self._class_prob))
            loss_grad[range(y.size), y] = -1.0 / self._class_prob[range(y.size), y]
        elif len(y.shape) == 2: # y --> OneHot Representation
            neg_ln_prob_grad = np.nan_to_num(-1.0 / self._class_prob) * y
            neg_ln_one_mns_prob_grad = np.nan_to_num(1.0 / (1.0 - self._class_prob)) * (1 - y)
            loss_grad = neg_ln_prob_grad + neg_ln_one_mns_prob_grad

        self._class_prob = None
        return loss_grad / y.shape[0]


    def evaluate(self, X, y):
        test_prob = self._nn.forward(X)
        pred = np.argmax(test_prob, axis=-1)

        if len(y.shape) == 1: # y --> Class labels
            accuracy = np.mean(pred == y) * 100.0
        elif len(y.shape) == 2: # y --> OneHot Representation
            pred_onehot = np.zeros((pred.size, self._nn.num_classes))
            pred_onehot[np.arange(pred.size), pred] = 1
            pred_diff = np.sum(np.fabs(y - pred_onehot), axis=-1) / 2.0
            accuracy = (1.0 - np.mean(pred_diff)) * 100.0
        return accuracy


    def print_log(self, epoch, plot, fig, axs, train_l, train_loss, test_loss, train_accuracy,
                  test_accuracy):
        test_l = self.loss(self._test_X, self._test_y)
        train_accur = self.evaluate(self._train_X, self._train_y)
        test_accur = self.evaluate(self._test_X, self._test_y)
        print("Epoch-%d - Training Loss: %.4f - Test Loss: %.4f - Train Accuracy: %.4f - Test Accuracy: %.4f" %
              (epoch, train_l, test_l, train_accur, test_accur))

        if plot:
            train_loss.append(train_l)
            test_loss.append(test_l)
            train_accuracy.append(train_accur)
            test_accuracy.append(test_accur)
            self.learning_curve_plot(fig, axs, train_loss, test_loss, train_accuracy,
                                     test_accuracy)

    def learning_curve_plot(self, fig, axs, train_loss, test_loss, train_accuracy, test_accuracy):
        x_values = list(range(len(train_loss)))

        axs[0].clear()
        axs[0].plot(x_values, train_loss, color='red', label='Train Loss')
        axs[0].plot(x_values, test_loss, color='blue', label='Test Loss')
        axs[0].set(ylabel='Loss')
        axs[0].legend(loc='upper right')

        axs[1].clear()
        axs[1].plot(x_values, train_accuracy, color='red', label='Train Accuracy')
        axs[1].plot(x_values, test_accuracy, color='blue', label='Test Accuracy')
        axs[1].set(ylabel='Accuracy(%)')
        axs[1].legend(loc='lower right')

        plt.show(block=False)
        plt.pause(0.01)


    def train(self, X, y, normalize=None, dims=None, shuffle=True, batch_size=256, epochs=100,
              y_onehot=False, plot=True, log_freq=1000):
        start_time = time.time()

        # Shuffle and split data into train and test sets
        self._train_X, self._train_y, self._test_X, self._test_y = \
            self.shuffle_split_data(X, y, shuffle=shuffle, y_onehot=y_onehot)
        num_batches = int(np.ceil(self._train_X.shape[0] /  batch_size))

        if normalize is not None and any(str in normalize.lower() for str in ['mean', 'zero',
                                                                              'center']):
            # Mean Normalize data
            mean, std, self._train_X = self.mean_normalize(self._train_X)
            _, _, self._test_X = self.mean_normalize(self._test_X, mean, std)
        elif normalize is not None and 'pca' in normalize.lower():
            # Project data on to principle componnts using SVD
            mean, U, S, dims, self._train_X = self.reduce_data_dimensions(self._train_X, dims=dims)
            _, _, _, _, self._test_X = \
                self.reduce_data_dimensions(self._test_X, dims=dims, mean=mean, U=U, S=S,
                                            N=self._train_X.shape[0])
            self._nn.layers[0].reinitialize_weights(inputs=self._train_X)

        if plot:
            fig, axs = plt.subplots(2, sharey=False, sharex=True)
        else:
            fig = axs = None

        train_loss = list()
        test_loss = list()
        train_accuracy = list()
        test_accuracy = list()

        init_train_l = self.loss(self._train_X, self._train_y)
        self.print_log(0, plot, fig, axs, init_train_l, train_loss, test_loss, train_accuracy,
                       test_accuracy)

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
                self.update_network()

            if (e+1) % log_freq == 0:
                self.print_log(e+1, plot, fig, axs, train_l, train_loss, test_loss, train_accuracy,
                               test_accuracy)

        end_time = time.time()
        print("\nTraining Time: %.2f(s)" % (end_time-start_time))


class SGD(Training):
    def __init__(self, nn=None, step_size=1e-2, reg_lambda=1e-4, train_size=70, test_size=30,
                 activatin_type=None, name=None):
        super().__init__(nn=nn, step_size=step_size, reg_lambda=reg_lambda, train_size=train_size,
                         test_size=test_size, activatin_type=activatin_type, name=name)

    def update_network(self):
        for l in self._nn.layers:
            l.weights += -self._step_size * l.weights_grad
            if l.bias is not None:
                l.bias += -self._step_size * l.bias_grad
