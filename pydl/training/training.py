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
import time
from collections import OrderedDict

from pydl import conf


class Training(ABC):
    """An abstract class defining the interface of Training Algorithms.

    Args:
        name (str): Name of the training algorithm.
    """

    def __init__(self, nn=None, step_size=1e-2, reg_lambda=1e-4, train_size=70, test_size=30,
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

        self._train_X = self._train_y = self._test_X = self._test_y = None
        self._class_prob = None
        self._prediction_delta = None

    def mean_normalize(self, X, mean=None, std=None):
        if mean is None:
            mean = np.mean(X, axis=0, keepdims=True)

        if std is None:
            std = np.std(X, axis=0, keepdims=True)
            std[std == 0] = 1

        mean_centered = X - mean
        normalized = mean_centered / std
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

    def loss(self, X, y, inference=False, prob=None):
        if self._regression:
            return self.mse_loss(X, y, inference, pred=prob)
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

    def mse_loss(self, X, y, inference=False, pred=None):
        if pred is None:
            prediction = self._nn.forward(X, inference)
        else:
            prediction = pred

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

    def batch_loss(self, X, y, batch_size=None, inference=True, log_freq=1):
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
            batch_l.append(self.loss(X[start:end], y[start:end], inference))
            self.reset_recurrent_layers(hidden_state='previous_state')

        return np.mean(batch_l)

    def evaluate(self, X, y, batch_size=None, inference=True):
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
            self.reset_recurrent_layers(hidden_state='previous_state')

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

    def fit_test_data(self, X, fig, axs, batch_size=None, inference=True):
        if batch_size is None:
            batch_size = X.shape[0]
        num_batches = int(np.ceil(X.shape[0] / batch_size))

        # # Reset hidden state of RNN layers, if any
        # self.reset_recurrent_layers()

        test_pred = list()
        for i in range(num_batches):
            start = int(batch_size * i)
            if i == num_batches - 1:
                end = X.shape[0]
            else:
                end = start + batch_size
            test_pred.append(self._nn.forward(X[start:end], inference))
            self.reset_recurrent_layers(hidden_state='previous_state')
        test_pred = np.concatenate(test_pred, axis=0)

        colors = ['red', 'blue', 'green', 'purple', 'orange']
        axs.clear()
        for c in range(X.shape[-1]):
            axs.plot(X[1:, c], linestyle='-', color=colors[c])
            axs.plot(test_pred[:-1, c], '.', color=colors[c])

        plt.show(block=False)
        plt.pause(0.01)

    def reset_recurrent_layers(self, hidden_state=None):
        for layer in self._nn.layers:
            if layer.type in ['RNN_Layer', 'LSTM_Layer']:
                layer.reset_internal_states(hidden_state)

    def print_log(self, epoch, plot, fig, axs, batch_size, train_l, epochs_list, train_loss,
                  test_loss, train_accuracy, test_accuracy, log_freq=1):
        test_l = self.batch_loss(self._test_X, self._test_y, batch_size, inference=True,
                                 log_freq=log_freq)
        if self._regression:
            train_l = self.batch_loss(self._train_X, self._train_y, batch_size, inference=True,
                                      log_freq=log_freq)
            train_accur = np.sqrt(train_l)
            test_accur = np.sqrt(test_l)
        else:  # Classification
            train_accur = self.evaluate(self._train_X, self._train_y, batch_size, inference=True)
            test_accur = self.evaluate(self._test_X, self._test_y, batch_size, inference=True)

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

    def print_log_rnn(self, epoch, plot, fig, axs, batch_size, train_l, epochs_list, train_loss,
                      test_loss, train_accuracy, test_accuracy, log_freq=1):
        test_l = self.batch_loss(self._test_X[:-1], self._test_X[1:], batch_size, inference=True,
                                 log_freq=log_freq)
        if self._regression:
            train_l = self.batch_loss(self._train_X[:-1], self._train_X[1:], batch_size,
                                      inference=True, log_freq=log_freq)
            train_accur = np.sqrt(train_l)
            test_accur = np.sqrt(test_l)
        else:  # Classification
            train_accur = self.evaluate(self._train_X[:-1], self._train_X[1:], batch_size,
                                        inference=True)
            test_accur = self.evaluate(self._test_X[:-1], self._test_X[1:], batch_size,
                                       inference=True)

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

    def prepare_data(self, X, y, normalize=None, dims=None, shuffle=True, batch_size=256,
                     y_onehot=False):
        # Shuffle and split data into train and test sets
        self._train_X, self._train_y, self._test_X, self._test_y = \
            self.shuffle_split_data(X, y, shuffle=shuffle, y_onehot=y_onehot)

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

        print("Training Data:\n", self._train_X.shape)
        print("Test Data:\n", self._test_X.shape)

    def prepare_character_data(self, X):
        # Encode data to OneHot
        data_size = len(X)
        unique_chars = list(set(X))
        K = len(unique_chars)

        self._char_to_idx = {ch: i for i, ch in enumerate(unique_chars)}
        self._idx_to_char = {i: ch for i, ch in enumerate(unique_chars)}

        train_size = int(data_size * self._train_size)
        test_size = data_size - train_size

        self._train_X = np.zeros((train_size, K), dtype=conf.dtype)
        for i, d in enumerate(X[:train_size]):
            self._train_X[i, self._char_to_idx[d]] = 1

        self._test_X = np.zeros((test_size, K), dtype=conf.dtype)
        for i, d in enumerate(X[train_size:]):
            self._test_X[i, self._char_to_idx[d]] = 1

        for k, v in self._idx_to_char.items():
            print("%d: %s" % (k, v))

        print("Training Data:\n", self._train_X.shape)
        print("Test Data:\n", self._test_X.shape)

    def prepare_regression_data(self, X, y=None):
        data_size = X.shape[0]
        train_size = int(data_size * self._train_size)

        self._train_X = X[:train_size]
        self._test_X = X[train_size:]

        if y is not None:
            self._train_y = y[:train_size]
            self._test_y = y[train_size:]

        print("Training Data:\n", self._train_X.shape)
        print("Test Data:\n", self._test_X.shape)

    def generate_character_sequence(self, sample_length=100, temperature=1.0):
        sampled_char = self._idx_to_char[np.random.randint(self._nn.num_classes)]
        while not (sampled_char.isalpha() and sampled_char.isupper()):
            sampled_char = self._idx_to_char[np.random.randint(self._nn.num_classes)]

        sampled_text = [sampled_char]
        for n in range(sample_length):
            # Set the previous sampled character (O_t-1) as current input (I_t) to the network,
            # encodes as a OneHot vector
            input = np.zeros((1, self._nn.num_classes), dtype=conf.dtype)
            input[0, self._char_to_idx[sampled_char]] = 1

            # Forward propagate throug the network
            for layer in self._nn.layers:
                if layer.type in ['RNN_Layer', 'LSTM_Layer']:
                    if n == 0:
                        # Reset hidden state at the beginning of sequence generation
                        layer.reset_internal_states()
                    input = np.copy(layer.forward(input, inference=True)[1])
                    # Set previous hidden state (h_t-1) to current hidden state output (h_t)
                    layer.reset_internal_states('previous_state')
                else:
                    probs = layer.forward(input, inference=True, temperature=temperature)
                    sampled_idx = np.random.choice(range(self._nn.num_classes),
                                                   p=probs.reshape(-1))
                    sampled_char = self._idx_to_char[sampled_idx]
                    sampled_text.append(sampled_char)

        return sampled_text

    def generate_cont_time_series_data(self, fig, axs, sample_length=500):
        input = self._train_X[-1].reshape(1, -1)

        sampled_data = [input]
        for n in range(sample_length):
            for layer in self._nn.layers:
                if layer.type in ['RNN_Layer', 'LSTM_Layer']:
                    input = np.copy(layer.forward(input, inference=True)[1])
                    # Set previous hidden state (h_t-1) to current hidden state output (h_t)
                    layer.reset_internal_states('previous_state')
                else:
                    input = layer.forward(input, inference=True)
                    sampled_data.append(input)

        sampled_data = np.concatenate(sampled_data, axis=0)
        data = np.vstack((self._train_X, sampled_data))
        train_size = self._train_X.shape[0]

        colors = ['red', 'blue', 'green', 'purple', 'orange']
        axs.clear()
        for c in range(data.shape[-1]):
            axs.plot(list(range(train_size)), data[:train_size, c], linestyle='-', color=colors[c])
            axs.plot(list(range(train_size, data.shape[0])), data[train_size:, c], linestyle=':',
                     color=colors[c])

        plt.show(block=False)
        plt.pause(0.01)

        return

    def train(self, batch_size=256, epochs=1000, plot=None, log_freq=100):
        start_time = time.time()

        if plot is not None:
            fig, axs = plt.subplots(2, sharey=False, sharex=True)
            fig.suptitle(plot, fontsize=20)
        else:
            fig = axs = None

        epochs_list = list()
        train_loss = list()
        test_loss = list()
        train_accuracy = list()
        test_accuracy = list()
        num_batches = int(np.ceil(self._train_X.shape[0] / batch_size))

        init_train_l = self.batch_loss(self._train_X, self._train_y, batch_size, inference=False,
                                       log_freq=log_freq)
        self.print_log(0, plot, fig, axs, batch_size, init_train_l, epochs_list, train_loss,
                       test_loss, train_accuracy, test_accuracy, log_freq)

        for e in range(epochs):
            for i in range(num_batches):
                if log_freq < 0:
                    print("Epoch: %d -- Batch: %d" % (e, i))
                start = int(batch_size * i)
                if i == num_batches - 1:
                    end = self._train_X.shape[0]
                else:
                    end = start + batch_size

                train_l = self.loss(self._train_X[start:end], self._train_y[start:end],
                                    inference=False)
                loss_grad = self.loss_gradient(self._train_X[start:end], self._train_y[start:end])
                _ = self._nn.backward(loss_grad, self._lambda)
                self.update_network(e + 1)

            if (e + 1) % np.abs(log_freq) == 0:
                self.print_log((e + 1), plot, fig, axs, batch_size, train_l, epochs_list,
                               train_loss, test_loss, train_accuracy, test_accuracy, log_freq)

        training_logs_dict = OrderedDict()
        training_logs_dict['epochs'] = epochs_list
        training_logs_dict['train_loss'] = train_loss
        training_logs_dict['test_loss'] = test_loss
        training_logs_dict['train_accuracy'] = train_accuracy
        training_logs_dict['test_accuracy'] = test_accuracy

        end_time = time.time()
        print("\nTraining Time: %.2f(s)" % (end_time - start_time))

        return training_logs_dict

    def train_rnn(self, batch_size=256, epochs=1000, sample_length=100, temperature=1.0, plot=None,
                  fit_test_data=False, log_freq=100):
        start_time = time.time()

        if plot is not None:
            fig, axs = plt.subplots(2, sharey=False, sharex=True)
            fig.suptitle(plot, fontsize=20)
        else:
            fig = axs = None

        if fit_test_data:
            fig_2, axs_2 = plt.subplots(1, sharey=False, sharex=False)
            fig_3, axs_3 = plt.subplots(1, sharey=False, sharex=False)

        epochs_list = list()
        train_loss = list()
        test_loss = list()
        train_accuracy = list()
        test_accuracy = list()
        num_batches = int(np.ceil((self._train_X.shape[0] - 1) / batch_size))

        init_train_l = self.batch_loss(self._train_X[:-1], self._train_X[1:], batch_size,
                                       inference=False, log_freq=log_freq)
        self.print_log_rnn(0, plot, fig, axs, batch_size, init_train_l, epochs_list, train_loss,
                           test_loss, train_accuracy, test_accuracy, log_freq)

        for e in range(epochs):
            for i in range(num_batches):
                if log_freq < 0:
                    print("Epoch: %d -- Batch: %d" % (e, i))
                startX = int(batch_size * i)
                startY = startX + 1
                if i == num_batches - 1:
                    endX = self._train_X.shape[0] - 1
                    endY = endX + 1
                else:
                    endX = startX + batch_size
                    endY = endX + 1

                train_l = self.loss(self._train_X[startX:endX], self._train_X[startY:endY],
                                    inference=False)
                loss_grad = self.loss_gradient(self._train_X[startX:endX],
                                               self._train_X[startY:endY])
                _ = self._nn.backward(loss_grad, self._lambda)
                self.update_network(e + 1)

            if e % 5 == 0:
                if self._regression:
                    if fit_test_data:
                        # Generate continuous time series data starting from the last training
                        # data point
                        sampled_text = \
                            self.generate_cont_time_series_data(fig_2, axs_2, sample_length)
                else:
                    # Sample from the model by setting an initial seed
                    sampled_text = self.generate_character_sequence(sample_length, temperature)
                    print('\n', ''.join(sampled_text), '\n')

            if (e + 1) % np.abs(log_freq) == 0:
                self.print_log_rnn((e + 1), plot, fig, axs, batch_size, train_l, epochs_list,
                                   train_loss, test_loss, train_accuracy, test_accuracy, log_freq)
                if self._regression and fit_test_data:
                    self.fit_test_data(self._test_X, fig_3, axs_3, batch_size, inference=True)

            # End of an epoch - Reset RNN layers
            self.reset_recurrent_layers()

        training_logs_dict = OrderedDict()
        training_logs_dict['epochs'] = epochs_list
        training_logs_dict['train_loss'] = train_loss

        end_time = time.time()
        print("\nTraining Time: %.2f(s)" % (end_time - start_time))

        return training_logs_dict


class SGD(Training):
    def __init__(self, nn=None, step_size=1e-2, reg_lambda=1e-4, train_size=70, test_size=30,
                 activatin_type=None, regression=False, name=None):
        super().__init__(nn=nn, step_size=step_size, reg_lambda=reg_lambda, train_size=train_size,
                         test_size=test_size, activatin_type=activatin_type, regression=regression,
                         name=name)

    def train(self, X, y, normalize=None, dims=None, shuffle=True, batch_size=256, epochs=1000,
              y_onehot=False, plot=None, log_freq=100):
        self.prepare_data(X=X, y=y, normalize=normalize, dims=dims, shuffle=shuffle,
                          batch_size=batch_size, y_onehot=y_onehot)

        training_logs_dict = super().train(batch_size=batch_size, epochs=epochs, plot=plot,
                                           log_freq=log_freq)
        return training_logs_dict

    def train_rnn(self, X, y=None, batch_size=256, epochs=1000, sample_length=100, temperature=1.0,
                  plot=None, fit_test_data=False, log_freq=100):
        if self._regression:
            self.prepare_regression_data(X)
        else:
            self.prepare_character_data(X)

        training_logs_dict = super().train_rnn(batch_size=batch_size, epochs=epochs, plot=plot,
                                               sample_length=sample_length, temperature=temperature,
                                               fit_test_data=fit_test_data, log_freq=log_freq)
        return training_logs_dict

    def update_network(self, t=None):
        for layer in self._nn.layers:
            if layer.type in ['FC_Layer', 'Convolution_Layer', 'RNN_Layer', 'LSTM_Layer']:
                layer.weights += -self._step_size * layer.weights_grad
                if layer.bias is not None:
                    layer.bias += -self._step_size * layer.bias_grad
            layer.reset()


class Momentum(Training):
    def __init__(self, nn=None, step_size=1e-2, mu=0.5, reg_lambda=1e-4, train_size=70,
                 test_size=30, activatin_type=None, regression=False, name=None):
        super().__init__(nn=nn, step_size=step_size, reg_lambda=reg_lambda, train_size=train_size,
                         test_size=test_size, activatin_type=activatin_type, regression=regression,
                         name=name)
        self._mu = mu
        self._vel = list()

    def init_momentum_velocity(self):
        # Initialize momentum velocities to zero
        for layer in self._nn.layers:
            if layer.type in ['FC_Layer', 'Convolution_Layer', 'RNN_Layer', 'LSTM_Layer']:
                v = {'w': np.zeros_like(layer.weights),
                     'b': np.zeros_like(layer.bias)}
            else:
                v = None
            self._vel.append(v)

    def train(self, X, y, normalize=None, dims=None, shuffle=True, batch_size=256, epochs=1000,
              y_onehot=False, plot=None, log_freq=100):
        self.prepare_data(X=X, y=y, normalize=normalize, dims=dims, shuffle=shuffle,
                          batch_size=batch_size, y_onehot=y_onehot)
        self.init_momentum_velocity()

        training_logs_dict = super().train(batch_size=batch_size, epochs=epochs, plot=plot,
                                           log_freq=log_freq)
        return training_logs_dict

    def train_rnn(self, X, y=None, batch_size=256, epochs=10000, sample_length=100, temperature=1.0,
                  plot=None, fit_test_data=False, log_freq=100):
        if self._regression:
            self.prepare_regression_data(X)
        else:
            self.prepare_character_data(X)

        self.init_momentum_velocity()

        training_logs_dict = super().train_rnn(batch_size=batch_size, epochs=epochs, plot=plot,
                                               sample_length=sample_length, temperature=temperature,
                                               fit_test_data=fit_test_data, log_freq=log_freq)
        return training_logs_dict

    def update_network(self, t=None):
        for layer, v in zip(self._nn.layers, self._vel):
            if layer.type in ['FC_Layer', 'Convolution_Layer', 'RNN_Layer', 'LSTM_Layer']:
                v['w'] = (v['w'] * self._mu) - (self._step_size * layer.weights_grad)
                v['b'] = (v['b'] * self._mu) - (self._step_size * layer.bias_grad)

                layer.weights += v['w']
                layer.bias += v['b']
            layer.reset()


class RMSprop(Training):
    def __init__(self, nn=None, step_size=1e-2, beta=0.999, reg_lambda=1e-4, train_size=70,
                 test_size=30, activatin_type=None, regression=False, name=None):
        super().__init__(nn=nn, step_size=step_size, reg_lambda=reg_lambda, train_size=train_size,
                         test_size=test_size, activatin_type=activatin_type, regression=regression,
                         name=name)
        self._beta = beta
        self._cache = list()

    def init_rmsprop_cache(self):
        # Initialize cache to zero
        for layer in self._nn.layers:
            if layer.type in ['FC_Layer', 'Convolution_Layer', 'RNN_Layer', 'LSTM_Layer']:
                c = {'w': np.zeros_like(layer.weights),
                     'b': np.zeros_like(layer.bias)}
            else:
                c = None
            self._cache.append(c)

    def train(self, X, y, normalize=None, dims=None, shuffle=True, batch_size=256, epochs=1000,
              y_onehot=False, plot=None, log_freq=100):
        self.prepare_data(X=X, y=y, normalize=normalize, dims=dims, shuffle=shuffle,
                          batch_size=batch_size, y_onehot=y_onehot)
        self.init_rmsprop_cache()

        training_logs_dict = super().train(batch_size=batch_size, epochs=epochs, plot=plot,
                                           log_freq=log_freq)
        return training_logs_dict

    def train_rnn(self, X, y=None, batch_size=256, epochs=10000, sample_length=100, temperature=1.0,
                  plot=None, fit_test_data=False, log_freq=100):
        if self._regression:
            self.prepare_regression_data(X)
        else:
            self.prepare_character_data(X)

        self.init_rmsprop_cache()

        training_logs_dict = super().train_rnn(batch_size=batch_size, epochs=epochs, plot=plot,
                                               sample_length=sample_length, temperature=temperature,
                                               fit_test_data=fit_test_data, log_freq=log_freq)
        return training_logs_dict

    def update_network(self, t=None):
        for layer, c in zip(self._nn.layers, self._cache):
            if layer.type in ['FC_Layer', 'Convolution_Layer', 'RNN_Layer', 'LSTM_Layer']:
                c['w'] = (self._beta * c['w']) + ((1 - self._beta) * (layer.weights_grad**2))
                c['b'] = (self._beta * c['b']) + ((1 - self._beta) * (layer.bias_grad**2))

                layer.weights += -(self._step_size / (np.sqrt(c['w']) + 1e-6)) * layer.weights_grad
                layer.bias += -(self._step_size / (np.sqrt(c['b']) + 1e-6)) * layer.bias_grad
            layer.reset()


class Adam(Training):
    def __init__(self, nn=None, step_size=1e-2, beta_1=0.9, beta_2=0.999, reg_lambda=1e-4,
                 train_size=70, test_size=30, activatin_type=None, regression=False, name=None):
        super().__init__(nn=nn, step_size=step_size, reg_lambda=reg_lambda, train_size=train_size,
                         test_size=test_size, activatin_type=activatin_type, regression=regression,
                         name=name)
        self._beta_1 = beta_1
        self._beta_2 = beta_2
        self._m = list()
        self._v = list()

    def init_adam_moment(self):
        # Initialize cache to zero
        for layer in self._nn.layers:
            if layer.type in ['FC_Layer', 'Convolution_Layer', 'RNN_Layer', 'LSTM_Layer']:
                m = {'w': np.zeros_like(layer.weights),
                     'b': np.zeros_like(layer.bias)}

                v = {'w': np.zeros_like(layer.weights),
                     'b': np.zeros_like(layer.bias)}
            else:
                m = None
                v = None
            self._m.append(m)
            self._v.append(v)

    def train(self, X, y, normalize=None, dims=None, shuffle=True, batch_size=256, epochs=10000,
              y_onehot=False, plot=None, log_freq=100):
        self.prepare_data(X=X, y=y, normalize=normalize, dims=dims, shuffle=shuffle,
                          batch_size=batch_size, y_onehot=y_onehot)
        self.init_adam_moment()

        training_logs_dict = super().train(batch_size=batch_size, epochs=epochs, plot=plot,
                                           log_freq=log_freq)
        return training_logs_dict

    def train_rnn(self, X, y=None, batch_size=256, epochs=10000, sample_length=100, temperature=1.0,
                  plot=None, fit_test_data=False, log_freq=100):
        if self._regression:
            self.prepare_regression_data(X)
        else:
            self.prepare_character_data(X)

        self.init_adam_moment()

        training_logs_dict = super().train_rnn(batch_size=batch_size, epochs=epochs, plot=plot,
                                               sample_length=sample_length, temperature=temperature,
                                               fit_test_data=fit_test_data, log_freq=log_freq)
        return training_logs_dict

    def update_network(self, t):
        for layer, m, v in zip(self._nn.layers, self._m, self._v):
            if layer.type in ['FC_Layer', 'Convolution_Layer', 'RNN_Layer', 'LSTM_Layer']:
                # First order moment update
                m['w'] = (self._beta_1 * m['w']) + ((1 - self._beta_1) * layer.weights_grad)
                m['b'] = (self._beta_1 * m['b']) + ((1 - self._beta_1) * layer.bias_grad)

                # Second order moment update
                v['w'] = (self._beta_2 * v['w']) + ((1 - self._beta_2) * (layer.weights_grad**2))
                v['b'] = (self._beta_2 * v['b']) + ((1 - self._beta_2) * (layer.bias_grad**2))

                # Update Weights
                m_hat_w = m['w'] / (1.0 - self._beta_1**t)
                v_hat_w = v['w'] / (1.0 - self._beta_2**t)
                layer.weights += -(self._step_size * m_hat_w) / (np.sqrt(v_hat_w) + 1e-8)

                # Update Bias
                m_hat_b = m['b'] / (1.0 - self._beta_1**t)
                v_hat_b = v['b'] / (1.0 - self._beta_2**t)
                layer.bias += -(self._step_size * m_hat_b) / (np.sqrt(v_hat_b) + 1e-8)

            layer.reset()
