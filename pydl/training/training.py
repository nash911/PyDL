# ------------------------------------------------------------------------
# MIT License
#
# Copyright (c) [2021] [Avinash Ranganath]
#
# This code is part of the library PyDL <https://github.com/nash911/PyDL>
# This code is licensed under MIT license (see LICENSE.txt for details)
# ------------------------------------------------------------------------

from abc import ABC
import numpy as np
import matplotlib.pyplot as plt
import sys
import warnings

from pydl import conf


class Training(ABC):
    """An abstract class defining the interface of Training Algorithms.

    Args:
        name (str): Name of the training algorithm.
    """

    def __init__(self, nn=None, step_size=1e-2, reg_lambda=0, train_size=70, test_size=30,
                 activatin_type=None, regression=False, name=None):
        self._nn = nn
        self._step_size = step_size
        self._lambda = reg_lambda
        self._regression = regression
        self._name = name

        if self._regression:
            # Regression task, so setting nn activation_type to None
            self._activatin_type = None
        elif activatin_type is not None:
            self._activatin_type = activatin_type
        elif nn is not None:
            self._activatin_type = self._nn.layers[-1].activation
        else:
            sys.exit("Error: Please specify activation_type for instance of class Training")

        self._binary_classification = False
        if nn is not None and not self._regression:
            if self._nn.num_output_neurons == 1:
                if self._activatin_type.lower() == 'sigmoid':
                    self._binary_classification = True
                else:
                    sys.exit("Error: For binary classification task with a single output neuron, " +
                             "please choose 'Sigmoid' activation function instead of " +
                             self._activatin_type.lower())
            elif self._nn.num_output_neurons == 2:
                print("WARNING: Network with two output neurons. Instead choose a single output " +
                      "neuron with Sigmoid activation function, for binary classification.")

        if train_size + test_size == 100:
            self._train_size = train_size / 100.0
            self._test_size = test_size / 100.0
        else:
            self._train_size = int(train_size)
            self._test_size = int(test_size)

        self._X_raw = None
        self._train_X = self._train_y = self._test_X = self._test_y = None
        self._X_mean = self._X_std = None
        self._X_min = self._X_max = None
        self._scale_range = np.array([-1, 1], dtype=conf.dtype)

        self._class_prob = None
        self._prediction_delta = None

    def train_size(self, X):
        data_size = X.shape[0]
        return int(data_size * self._train_size) if self._train_size <= 1.0 else \
            self._train_size

    def mean_normalize(self, X, mean=None, std=None):
        if mean is None:
            if self._X_mean is not None:
                warnings.warn("\nWARNING! Calculating data mean for normalization, while " +
                              "precalculated X-mean exists!")
            self._X_mean = np.mean(X, axis=0, keepdims=True)
            mean = self._X_mean

        if std is None:
            if self._X_std is not None:
                warnings.warn("\nWARNING! Calculating data std for normalization, while " +
                              "precalculated X-std exists!")
            self._X_std = np.std(X, axis=0, keepdims=True)
            self._X_std[self._X_std == 0] = 1
            std = self._X_std

        mean_centered = X - mean
        normalized = mean_centered / std
        return mean, std, normalized

    def invert_mean_normalization(self, X):
        if self._X_mean is None or self._X_std is None:
            sys.exit("Error: Mean/Std of the data is not calculated.")

        unnormalized = (X * self._X_std) + self._X_mean
        return unnormalized

    def min_max_normalize(self, X, min=None, max=None):
        if min is None:
            if self._X_min is not None:
                warnings.warn("\nWARNING! Calculating data min for normalization, while " +
                              "precalculated X-min exists!")
            self._X_min = np.min(X, axis=0, keepdims=True)
            min = self._X_min

        if max is None:
            if self._X_max is not None:
                warnings.warn("\nWARNING! Calculating data max for normalization, while " +
                              "precalculated X-max exists!")
            self._X_max = np.max(X, axis=0, keepdims=True)
            max = self._X_max

        a = self._scale_range[0]
        b = self._scale_range[1]

        normalized = a + (((X - min) * (b - a)) / (max - min))
        return min, max, normalized

    def invert_min_max_normalization(self, X):
        if self._X_min is None or self._X_max is None:
            sys.exit("Error: Min/Max of the data is not calculated.")

        a = self._scale_range[0]
        b = self._scale_range[1]
        unnormalized = (((X - a) * (self._X_max - self._X_min)) / (b - a)) + self._X_min
        return unnormalized

    def invert_normalization(self, X, normalize=None):
        if normalize.lower() == 'mean':
            inverted_X = self.invert_mean_normalization(X)
        elif 'min' in normalize.lower() and 'max' in normalize.lower():
            inverted_X = self.invert_min_max_normalization(X)

        return inverted_X

    def convert_to_difference(self, X, step=1):
        self._X_raw = X
        return X[step:] - X[:-step]

    def invert_difference(self, X_raw, diff):
        return X_raw + diff

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
            cumulative_ratio = (np.cumsum(S) / np.sum(S)).tolist()
            min_ratio = min([i for i in cumulative_ratio if i > dims])
            dims = cumulative_ratio.index(min_ratio)
            print("PCA - Reduced to %d dimensions, while retaining %.2f percent information" %
                  (dims + 1, (min_ratio) * 100.0))
            dims += 1
        elif type(dims) is int and dims >= 1:
            print("PCA - Reduced to %d dimensions, while retaining %.2f percent information" %
                  (dims, ((np.cumsum(S) / np.sum(S))[dims - 1]) * 100.0))

        # Project data on to orthogonal subspace
        X_reduced = np.matmul(X, U[:, :dims])

        if whiten:
            X_reduced = X_reduced / np.sqrt(S[:dims] + 1e-4)

        return mean, U, S, dims, X_reduced

    def shuffle_split_data(self, X, y, shuffle=True, train_size=None, test_size=None,
                           y_onehot=False):
        sample_size = X.shape[0]

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

        if self._binary_classification and y_onehot:
            sys.exit("Error: For binary classification task, class labels will not be " +
                     "represented as One-Hot vectors. Set 'y_onehot' parameter to False in the " +
                     "function call Training.shuffle_split_data().")

        if self._regression:
            labels = np.reshape(y, newshape=(-1, 1))
        elif self._binary_classification:
            if len(y.shape) == 1:
                labels = np.reshape(y, newshape=(-1, 1))
            elif len(y.shape) == 2:
                labels = np.reshape(np.argmax(y, axis=-1), newshape=(-1, 1))
        elif y_onehot and len(y.shape) == 1:
            # Convert labels to one-hot vector
            if self._nn is None:
                labels = np.zeros((y.size, y.max() + 1))
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

    def loss(self, X, y, y_raw=None, pred_diff=None, inference=False, prob=None, normalize=None):
        if self._regression:
            return self.mse_loss(X, y, y_raw, pred_diff, inference, pred=prob, normalize=normalize)
        else:
            if 'softmax' in self._activatin_type.lower():
                return self.softmax_cross_entropy_loss(X, y, inference, prob)
            elif 'sigmoid' in self._activatin_type.lower():
                return self.sigmoid_cross_entropy_loss(X, y, inference, prob)
            else:
                sys.exit("Error: Unknown activation_type: ", self._activation_fn)

    def loss_gradient(self, X, y, inference=False, prob=None):
        if self._regression:
            return self.mse_gradient(X, y, inference)
        else:
            if 'softmax' in self._activatin_type.lower():
                return self.softmax_cross_entropy_gradient(X, y, inference, prob)
            elif 'sigmoid' in self._activatin_type.lower():
                return self.sigmoid_cross_entropy_gradient(X, y, inference, prob)
            else:
                sys.exit("Error: Unknown activation_type: ", self._activation_fn)

    def mse_loss(self, X, y, y_raw=None, pred_diff=None, inference=False, pred=None,
                 normalize=None):
        if pred is None:
            prediction = self._nn.forward(X, inference)
        else:
            prediction = pred

        if inference:
            # Invert Normalization on labels (y) and prediction
            if normalize is not None:
                y = self.invert_normalization(y, normalize)
                prediction = self.invert_normalization(prediction, normalize)

            # Invert data differencing on labels (y)
            if y_raw is not None:
                y = y_raw

            # Invert data differencing on prediction
            if pred_diff is not None:
                prediction = self.invert_difference(pred_diff, prediction)

        #        1  m
        # MSE = --- ∑ (ŷᵢ - yᵢ)²
        #       2m  i
        self._prediction_delta = y - prediction
        data_loss = 0.5 * np.mean(np.square(self._prediction_delta))

        if not inference and self._nn is not None and self._lambda > 0:
            nn_weights = self._nn.weights
            regularization_loss = 0
            for w in nn_weights:
                if w is not None:
                    regularization_loss += np.sum(w * w)
            regularization_loss *= (0.5 * self._lambda)
        else:
            regularization_loss = 0

        # MSE Cost Fn.
        return data_loss + regularization_loss

    def mse_gradient(self, X, y, inference=False):
        if self._prediction_delta is None:
            _ = self._nn.forward(X, inference)

        # ∂MSE      1
        # ---- = - --- (ŷᵢ - yᵢ)
        #  ∂yᵢ      m
        loss_grad = -self._prediction_delta / self._prediction_delta.shape[0]

        self._prediction_delta = None
        return loss_grad

    def softmax_cross_entropy_loss(self, X, y, inference=False, prob=None):
        if prob is None:
            self._class_prob = self._nn.forward(X, inference)
        else:
            # sum_probs = np.sum(prob, axis=-1, keepdims=True)
            # ones = np.ones_like(sum_probs)
            self._class_prob = prob

        with np.errstate(divide='ignore'):
            # -ln(σ(z))
            neg_ln_prob = -np.log(self._class_prob)
            neg_ln_prob = np.nan_to_num(neg_ln_prob)

        if self._nn is not None and self._lambda > 0:
            nn_weights = self._nn.weights
            regularization_loss = 0
            for w in nn_weights:
                if w is not None:
                    regularization_loss += np.sum(w * w)
            regularization_loss *= (0.5 * self._lambda)
        else:
            regularization_loss = 0

        if len(y.shape) == 2:  # y --> OneHot Representation
            # Convert y to class labels
            y = np.argmax(y, axis=-1)

        # Cross-Entropy Cost Fn.
        #          m k               1
        # L(p,y) = ∑ ∑-ln(pᵢ) * yᵢ + - λ∑(θⱼ)²
        #            i               2  j
        return np.mean(neg_ln_prob[range(y.size), y]) + regularization_loss

    def softmax_cross_entropy_gradient(self, X, y, inference=False, prob=None):
        if prob is not None:
            # sum_probs = np.sum(prob, axis=-1, keepdims=True)
            # ones = np.ones_like(sum_probs)
            self._class_prob = prob
        elif self._class_prob is None:
            self._class_prob = self._nn.forward(X, inference)

        if len(y.shape) == 2:  # y --> OneHot Representation
            # Convert y to class labels
            y = np.argmax(y, axis=-1)

        loss_grad = np.zeros_like(self._class_prob)
        loss_grad[range(y.size), y] = \
            np.nan_to_num(-1.0 / self._class_prob[range(y.size), y]) / y.shape[0]

        self._class_prob = None
        return loss_grad

    def sigmoid_cross_entropy_loss(self, X, y, inference=False, prob=None):
        if prob is None:
            self._class_prob = self._nn.forward(X, inference)
        else:
            self._class_prob = prob

        with np.errstate(divide='ignore'):
            # -ln(σ(z))
            neg_ln_prob = -np.log(self._class_prob)

            # -ln(1 - σ(z))
            neg_ln_one_mns_prob = -np.log(1.0 - self._class_prob)

        if len(y.shape) == 1:  # y --> Class labels
            neg_ln_one_mns_prob[range(y.size), y] = neg_ln_prob[range(y.size), y]
            logistic_probs = neg_ln_one_mns_prob
        elif len(y.shape) == 2:  # y --> OneHot Representation
            neg_ln_prob = np.nan_to_num(neg_ln_prob)
            neg_ln_one_mns_prob = np.nan_to_num(neg_ln_one_mns_prob)
            if self._binary_classification:
                logistic_probs = (y * neg_ln_prob) + ((1 - y) * neg_ln_one_mns_prob)
            else:
                neg_ln_prob *= y
                neg_ln_one_mns_prob *= (1 - y)
                logistic_probs = neg_ln_prob + neg_ln_one_mns_prob

        if self._nn is not None and self._lambda > 0:
            nn_weights = self._nn.weights
            regularization_loss = 0
            for w in nn_weights:
                if w is not None:
                    regularization_loss += np.sum(w * w)
            regularization_loss *= (0.5 * self._lambda)
        else:
            regularization_loss = 0

        # Logistic Cost Fn.
        return (np.sum(logistic_probs) / y.shape[0]) + regularization_loss

    def sigmoid_cross_entropy_gradient(self, X, y, inference=False, prob=None):
        if prob is not None:
            self._class_prob = prob
        elif self._class_prob is None:
            self._class_prob = self._nn.forward(X, inference)

        if len(y.shape) == 1:  # y --> Class labels
            loss_grad = (1.0 / (1.0 - self._class_prob))
            loss_grad[range(y.size), y] = -1.0 / self._class_prob[range(y.size), y]
        elif len(y.shape) == 2:  # y --> OneHot Representation
            if self._binary_classification:
                loss_grad = (y * np.nan_to_num(-1.0 / self._class_prob)) + \
                            ((1 - y) * np.nan_to_num(1.0 / (1.0 - self._class_prob)))
            else:
                neg_ln_prob_grad = np.nan_to_num(-1.0 / self._class_prob) * y
                neg_ln_one_mns_prob_grad = np.nan_to_num(1.0 / (1.0 - self._class_prob)) * (1 - y)
                loss_grad = neg_ln_prob_grad + neg_ln_one_mns_prob_grad

        self._class_prob = None
        return loss_grad / y.shape[0]

    def batch_loss(self, X, y, y_raw=None, batch_size=None, inference=True, normalize=None,
                   hidden_state=None, log_freq=1):
        if batch_size is None:
            batch_size = X.shape[0]
        num_batches = int(np.ceil(X.shape[0] / batch_size))

        # Reset hidden state of RNN layers, if any
        self.reset_recurrent_layers()

        batch_l = list()
        for i in range(num_batches):
            if log_freq < 0:
                print("Inference -- Batch: %d" % (i))
            start = int(batch_size * i)
            if i == num_batches - 1:
                end = X.shape[0]
            else:
                end = start + batch_size

            batch_l.append(self.loss(X[start:end], y[start:end],
                                     (None if y_raw is None else y_raw[(start + 1):(end + 1)]),
                                     (None if y_raw is None else y_raw[(start):(end)]),
                                     inference, normalize=normalize))

            self.reset_recurrent_layers(hidden_state=hidden_state)

        return np.mean(batch_l)

    def evaluate(self, X, y, batch_size=None, inference=True, hidden_state=None):
        if batch_size is None:
            batch_size = X.shape[0]
        num_batches = int(np.ceil(X.shape[0] / batch_size))

        # Reset hidden state of RNN layers, if any
        self.reset_recurrent_layers()

        test_prob = list()
        for i in range(num_batches):
            start = int(batch_size * i)
            if i == num_batches - 1:
                end = X.shape[0]
            else:
                end = start + batch_size
            test_prob.append(self._nn.forward(X[start:end], inference))
            self.reset_recurrent_layers(hidden_state=hidden_state)

        test_prob = np.vstack(test_prob)

        if self._binary_classification:
            pred = np.array(test_prob >= 1 - test_prob, dtype=np.int32)
        else:
            pred = np.argmax(test_prob, axis=-1)

        if self._binary_classification or len(y.shape) == 1:  # y --> Class labels
            accuracy = np.mean(pred == y) * 100.0
        elif len(y.shape) == 2:  # y --> OneHot Representation
            pred_onehot = np.zeros((pred.size, self._nn.num_classes))
            pred_onehot[np.arange(pred.size), pred] = 1
            pred_diff = np.sum(np.fabs(y - pred_onehot), axis=-1) / 2.0
            accuracy = (1.0 - np.mean(pred_diff)) * 100.0
        return accuracy

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
        axs[1].set(xlabel='Epoch')
        axs[1].legend(loc='upper right' if self._regression else'lower right')

        plt.show(block=False)
        plt.pause(0.01)

    def print_log(self, epoch, plot, fig, axs, batch_size, train_l, epochs_list, train_loss,
                  test_loss, train_accuracy, test_accuracy, log_freq=1, normalize=None,
                  data_diff=False):
        if self._train_y is None or self._test_y is None:
            train_X = self._train_X[:-1]
            train_y = self._train_X[1:]

            test_X = self._test_X[:-1]
            test_y = self._test_X[1:]

            # Set hidden state of a recurrent unit to the ...
            hidden_state = 'previous_state'
        else:
            train_X = self._train_X
            train_y = self._train_y

            test_X = self._test_X
            test_y = self._test_y

            hidden_state = None

        if data_diff:
            train_size = self.train_size(self._X_raw[:-1])
            train_y_raw = self._X_raw[1:(train_size + 1)]
            test_y_raw = self._X_raw[train_size:]
        else:
            train_y_raw = None
            test_y_raw = None

        test_l = self.batch_loss(test_X, test_y, test_y_raw, batch_size, inference=True,
                                 normalize=normalize, hidden_state=hidden_state, log_freq=log_freq)
        if self._regression:
            train_l = \
                self.batch_loss(train_X, train_y, train_y_raw, batch_size, inference=True,
                                normalize=normalize, hidden_state=hidden_state, log_freq=log_freq)
            train_accur = np.sqrt(train_l)
            test_accur = np.sqrt(test_l)
        else:  # Classification
            train_accur = self.evaluate(train_X, train_y, batch_size, inference=True)
            test_accur = self.evaluate(test_X, test_y, batch_size, inference=True)

        # Store training logs
        epochs_list.append(epoch)
        train_loss.append(train_l)
        test_loss.append(test_l)
        train_accuracy.append(train_accur)
        test_accuracy.append(test_accur)

        # Print training logs
        print(("Epoch-%d - Training Loss: %.4f - Test Loss: %.4f - Train Accuracy: %.4f - " +
               "Test Accuracy: %.4f") % (epoch, train_l, test_l, train_accur, test_accur))

        if plot:
            self.learning_curve_plot(fig, axs, train_loss, test_loss, train_accuracy,
                                     test_accuracy)
