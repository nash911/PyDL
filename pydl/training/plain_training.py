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
import json

from pydl.training.training import Training


class PlainTraining(Training):
    """Class for training non-recurrent NN architectures like FC and CNN.

    Args:
        name (str): Name of the training algorithm.
    """

    def __init__(self, nn=None, step_size=1e-2, reg_lambda=0, train_size=70, test_size=30,
                 activatin_type=None, regression=False, save=False, name=None):
        super().__init__(nn=nn, step_size=step_size, reg_lambda=reg_lambda, train_size=train_size,
                         test_size=test_size, activatin_type=activatin_type, regression=regression,
                         save=save, name=name)

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
            # Project data on to principle components using SVD
            mean, U, S, dims, self._train_X = self.reduce_data_dimensions(self._train_X, dims=dims)
            _, _, _, _, self._test_X = \
                self.reduce_data_dimensions(self._test_X, dims=dims, mean=mean, U=U, S=S,
                                            N=self._train_X.shape[0])
            self._nn.layers[0].reinitialize_weights(inputs=self._train_X)

        if self._save:
            self._save_dict['data'] = self.save_data_params()

        print("Training Data:\n", self._train_X.shape)
        print("Test Data:\n", self._test_X.shape)

    def train(self, batch_size=256, epochs=1000, plot=None, log_freq=100, model_file=None):
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

        init_train_l = self.batch_loss(self._train_X, self._train_y, batch_size=batch_size,
                                       inference=False, log_freq=log_freq)
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

            if self._save and (e + 1) % np.abs(log_freq) == 0:
                self._save_dict['nn'] = self._nn.save()
                with open(model_file, 'w') as mf:
                    json.dump(self._save_dict, mf)

        if self._save and (e + 1) % np.abs(log_freq) != 0:
            # Save the latest model after the final epoch of training
            self._save_dict['nn'] = self._nn.save()
            with open(model_file, 'w') as mf:
                json.dump(self._save_dict, mf)

        training_logs_dict = OrderedDict()
        training_logs_dict['epochs'] = epochs_list
        training_logs_dict['train_loss'] = train_loss
        training_logs_dict['test_loss'] = test_loss
        training_logs_dict['train_accuracy'] = train_accuracy
        training_logs_dict['test_accuracy'] = test_accuracy

        end_time = time.time()
        print("\nTraining Time: %.2f(s)" % (end_time - start_time))

        return training_logs_dict
