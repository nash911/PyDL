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
import time
from collections import OrderedDict

from pydl.training.training import Training
from pydl import conf


class RecurrentTraining(Training):
    """Class for training recurrent NN architectures like RNN, LSTM, and GRU.

    Args:
        name (str): Name of the training algorithm.
    """

    def __init__(self, nn=None, step_size=1e-2, reg_lambda=0, train_size=70, test_size=30,
                 activatin_type=None, regression=False, name=None):
        super().__init__(nn=nn, step_size=step_size, reg_lambda=reg_lambda, train_size=train_size,
                         test_size=test_size, activatin_type=activatin_type, regression=regression,
                         name=name)

    def reset_recurrent_layers(self, hidden_state=None):
        for layer in self._nn.layers:
            if layer.type in ['RNN_Layer', 'LSTM_Layer', 'GRU_Layer']:
                layer.reset_internal_states(hidden_state)

    def fit_test_data(self, X, fig, axs, batch_size=None, normalize=None, inference=True):
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
            prediction = self._nn.forward(X[start:end], inference)
            if normalize is not None:
                if normalize.lower() == 'mean':
                    prediction = self.invert_mean_normalization(prediction)
                elif 'min' in normalize.lower() and 'max' in normalize.lower():
                    prediction = self.invert_min_max_normalization(prediction)
            test_pred.append(prediction)
            self.reset_recurrent_layers(hidden_state='previous_state')
        test_pred = np.concatenate(test_pred, axis=0)

        colors = ['red', 'blue', 'green', 'purple', 'orange']
        axs.clear()

        if normalize is not None:
            if normalize.lower() == 'mean':
                unnorm_X = self.invert_mean_normalization(X)
            elif 'min' in normalize.lower() and 'max' in normalize.lower():
                unnorm_X = self.invert_min_max_normalization(X)
        else:
            unnorm_X = X

        for c in range(X.shape[-1]):
            axs.plot(unnorm_X[1:, c], linestyle='-', color=colors[c])
            axs.plot(test_pred[:-1, c], '.', color=colors[c])

        plt.show(block=False)
        plt.pause(0.01)

    def prepare_character_data(self, X, y=None):
        # Encode data to OneHot
        data_size = len(X)
        unique_chars = list(set(X))
        K = len(unique_chars)

        self._char_to_idx = {ch: i for i, ch in enumerate(unique_chars)}
        self._idx_to_char = {i: ch for i, ch in enumerate(unique_chars)}

        train_size = int(data_size * self._train_size) if self._train_size <= 1.0 else \
            self._train_size
        test_size = data_size - train_size

        self._train_X = np.zeros((train_size, K), dtype=conf.dtype)
        for i, d in enumerate(X[:train_size]):
            self._train_X[i, self._char_to_idx[d]] = 1

        self._test_X = np.zeros((test_size, K), dtype=conf.dtype)
        for i, d in enumerate(X[train_size:]):
            self._test_X[i, self._char_to_idx[d]] = 1

        for k, v in self._idx_to_char.items():
            print("%d: %s" % (k, v))

    def prepare_sequence_data(self, X, y=None, normalize=None):
        try:
            data_size = X.shape[0]
            train_size = int(data_size * self._train_size) if self._train_size <= 1.0 else \
                self._train_size
        except AttributeError:
            # Character data
            self.prepare_character_data(X)

        if y is not None:
            self._train_X = X[:train_size]
            self._test_X = X[train_size:]

            self._train_y = y[:train_size]
            self._test_y = y[train_size:]

            if self._binary_classification:
                if len(self._train_y.shape) == 1:
                    self._train_y = np.reshape(self._train_y, newshape=(-1, 1))
                    self._test_y = np.reshape(self._test_y, newshape=(-1, 1))
                elif len(self._train_y.shape) == 2 and self._train_y.shape[1] > 1:
                    self._train_y = np.reshape(np.argmax(self._train_y, axis=-1), newshape=(-1, 1))
                    self._test_y = np.reshape(np.argmax(self._test_y, axis=-1), newshape=(-1, 1))

            if normalize is not None:
                if normalize.lower() == 'mean':
                    # Mean Normalize data
                    mean, std, self._train_X = self.mean_normalize(self._train_X)
                    _, _, self._test_X = self.mean_normalize(self._test_X, mean, std)
                elif 'min' in normalize.lower() and 'max' in normalize.lower():
                    # Min-Max Normalize data
                    min, max, self._train_X = self.min_max_normalize(self._train_X)
                    _, _, self._test_X = self.min_max_normalize(self._test_X, min, max)
        else:
            if self._regression:
                self._train_X = X[:train_size]
                self._test_X = X[train_size:]

                if normalize is not None:
                    if normalize.lower() == 'mean':
                        # Mean Normalize data
                        mean, std, self._train_X = self.mean_normalize(self._train_X)
                        _, _, self._test_X = self.mean_normalize(self._test_X, mean, std)
                    elif 'min' in normalize.lower() and 'max' in normalize.lower():
                        # Min-Max Normalize data
                        min, max, self._train_X = self.min_max_normalize(self._train_X)
                        _, _, self._test_X = self.min_max_normalize(self._test_X, min, max)
            else:
                self.prepare_character_data(X)

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
                if layer.type in ['RNN_Layer', 'LSTM_Layer', 'GRU_Layer']:
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

    def generate_cont_time_series_data(self, fig, axs, normalize=None, sample_length=500):
        if normalize is not None:
            if normalize.lower() == 'mean':
                train_X = self.invert_mean_normalization(self._train_X)
            elif 'min' in normalize.lower() and 'max' in normalize.lower():
                train_X = self.invert_min_max_normalization(self._train_X)
        else:
            train_X = self._train_X

        input = train_X[-1].reshape(1, -1)

        sampled_data = [input]
        for n in range(sample_length):
            for layer in self._nn.layers:
                if layer.type in ['RNN_Layer', 'LSTM_Layer', 'GRU_Layer']:
                    input = np.copy(layer.forward(input, inference=True)[1])
                    # Set previous hidden state (h_t-1) to current hidden state output (h_t)
                    layer.reset_internal_states('previous_state')
                else:
                    input = layer.forward(input, inference=True)
                    if normalize is not None:
                        if normalize.lower() == 'mean':
                            unnormalized_output = self.invert_mean_normalization(input)
                        elif 'min' in normalize.lower() and 'max' in normalize.lower():
                            unnormalized_output = self.invert_min_max_normalization(input)
                    else:
                        unnormalized_output = input
                    sampled_data.append(unnormalized_output)

        sampled_data = np.concatenate(sampled_data, axis=0)
        data = np.vstack((train_X, sampled_data))
        train_size = train_X.shape[0]

        colors = ['red', 'blue', 'green', 'purple', 'orange']
        axs.clear()
        for c in range(data.shape[-1]):
            axs.plot(list(range(train_size)), data[:train_size, c], linestyle='-', color=colors[c])
            axs.plot(list(range(train_size, data.shape[0])), data[train_size:, c], linestyle=':',
                     color=colors[c])

        plt.show(block=False)
        plt.pause(0.01)

        return

    def train_recurrent(self, batch_size=256, epochs=1000, sample_length=100, normalize=None,
                        temperature=1.0, plot=None, fit_test_data=False, log_freq=100):
        start_time = time.time()

        if plot is not None:
            fig, axs = plt.subplots(2, sharey=False, sharex=True)
            fig.suptitle(plot, fontsize=20)
        else:
            fig = axs = None

        if fit_test_data:
            fig_2, axs_2 = plt.subplots(1, sharey=False, sharex=False)
            fig_3, axs_3 = plt.subplots(1, sharey=False, sharex=False)

        if self._train_y is None or self._test_y is None:
            train_X = self._train_X[:-1]
            train_y = self._train_X[1:]
        else:
            train_X = self._train_X
            train_y = self._train_y

        epochs_list = list()
        train_loss = list()
        test_loss = list()
        train_accuracy = list()
        test_accuracy = list()
        num_batches = int(np.ceil(train_X.shape[0] / batch_size))

        init_train_l = self.batch_loss(train_X, train_y, batch_size, inference=False,
                                       hidden_state='previous_state', log_freq=log_freq)
        self.print_log(0, plot, fig, axs, batch_size, init_train_l, epochs_list, train_loss,
                       test_loss, train_accuracy, test_accuracy, log_freq)

        for e in range(epochs):
            for i in range(num_batches):
                if log_freq < 0:
                    print("Epoch: %d -- Batch: %d" % (e, i))

                if self._train_y is None:
                    # Case where temporal dependency exists in the full batch on data
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
                else:
                    # Case where temporal dependency does not exist between individual data points,
                    # but only within a single data popint (eg: Image Captioning)
                    start = int(batch_size * i)
                    if i == num_batches - 1:
                        end = self._train_X.shape[0]
                    else:
                        end = start + batch_size

                    train_l = self.loss(self._train_X[start:end], self._train_y[start:end],
                                        inference=False)
                    loss_grad = \
                        self.loss_gradient(self._train_X[start:end], self._train_y[start:end])

                _ = self._nn.backward(loss_grad, self._lambda)
                self.update_network(e + 1)

            if e % 5 == 0:
                if self._regression:
                    if fit_test_data:
                        # Generate continuous time series data starting from the last training
                        # data point
                        sampled_text = \
                            self.generate_cont_time_series_data(fig_2, axs_2, normalize,
                                                                sample_length)
                else:
                    if self._train_y is None:
                        # Sample from the model by setting an initial seed
                        sampled_text = self.generate_character_sequence(sample_length, temperature)
                        print('\n', ''.join(sampled_text), '\n')

            if (e + 1) % np.abs(log_freq) == 0:
                self.print_log((e + 1), plot, fig, axs, batch_size, train_l, epochs_list,
                               train_loss, test_loss, train_accuracy, test_accuracy, log_freq)
                if self._regression and fit_test_data:
                    self.fit_test_data(self._test_X, fig_3, axs_3, batch_size, normalize,
                                       inference=True)

            # End of an epoch - Reset RNN layers
            self.reset_recurrent_layers()

        training_logs_dict = OrderedDict()
        training_logs_dict['epochs'] = epochs_list
        training_logs_dict['train_loss'] = train_loss
        training_logs_dict['test_loss'] = test_loss
        training_logs_dict['train_accuracy'] = train_accuracy
        training_logs_dict['test_accuracy'] = test_accuracy

        end_time = time.time()
        print("\nTraining Time: %.2f(s)" % (end_time - start_time))

        # if plot is not None:
        #     plt.close(fig)

        return training_logs_dict
