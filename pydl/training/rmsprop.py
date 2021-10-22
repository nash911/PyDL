# ------------------------------------------------------------------------
# MIT License
#
# Copyright (c) [2021] [Avinash Ranganath]
#
# This code is part of the library PyDL <https://github.com/nash911/PyDL>
# This code is licensed under MIT license (see LICENSE.txt for details)
# ------------------------------------------------------------------------

import numpy as np

from pydl.training.plain_training import PlainTraining
from pydl.training.recurrent_training import RecurrentTraining


class RMSprop(PlainTraining, RecurrentTraining):
    def __init__(self, nn=None, step_size=1e-2, beta=0.999, reg_lambda=0, train_size=70,
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

                try:  # A Recurrent Layer (Eg: RNN or LSTM layer)
                    if layer.tune_internal_states:
                        c['h'] = np.zeros_like(layer.init_hidden_state)

                        try:  # LSTM layer
                            c['c'] = np.zeros_like(layer.init_cell_state)
                        except AttributeError:  # non-LSTM rcurrent layer
                            pass
                except AttributeError:  # Non-Recurrent Layer (FC or CNN layer)
                    pass
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

    def train_recurrent(self, X, y=None, batch_size=256, epochs=10000, sample_length=100,
                        normalize=False, temperature=1.0, plot=None, fit_test_data=False,
                        log_freq=100):
        self.prepare_sequence_data(X, y, normalize)

        self.init_rmsprop_cache()

        training_logs_dict = \
            super().train_recurrent(batch_size=batch_size, epochs=epochs, plot=plot,
                                    sample_length=sample_length, temperature=temperature,
                                    fit_test_data=fit_test_data, log_freq=log_freq)
        return training_logs_dict

    def update_network(self, t=None):
        for layer, c in zip(self._nn.layers, self._cache):
            if layer.type in ['FC_Layer', 'Convolution_Layer', 'RNN_Layer', 'LSTM_Layer']:
                c['w'] = (self._beta * c['w']) + ((1 - self._beta) * (layer.weights_grad**2))
                c['b'] = (self._beta * c['b']) + ((1 - self._beta) * (layer.bias_grad**2))

                # Update Weights and Bias
                layer.weights += -(self._step_size / (np.sqrt(c['w']) + 1e-6)) * layer.weights_grad
                layer.bias += -(self._step_size / (np.sqrt(c['b']) + 1e-6)) * layer.bias_grad

                # Update Cell State
                if layer.cell_state_grad is not None:
                    c['c'] = \
                        (self._beta * c['c']) + ((1 - self._beta) * (layer.cell_state_grad**2))
                    layer.init_cell_state += -(self._step_size / (np.sqrt(c['c']) + 1e-6)) * \
                        layer.cell_state_grad

                # Update Hidden State
                if layer.hidden_state_grad is not None:
                    c['h'] = \
                        (self._beta * c['h']) + ((1 - self._beta) * (layer.hidden_state_grad**2))
                    layer.init_hidden_state += -(self._step_size / (np.sqrt(c['h']) + 1e-6)) * \
                        layer.hidden_state_grad
            layer.reset()
