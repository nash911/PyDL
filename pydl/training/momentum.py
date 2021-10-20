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


class Momentum(PlainTraining, RecurrentTraining):
    def __init__(self, nn=None, step_size=1e-2, mu=0.5, reg_lambda=0, train_size=70, test_size=30,
                 activatin_type=None, regression=False, name=None):
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

                try:  # A Recurrent Layer (Eg: RNN or LSTM layer)
                    if layer.tune_internal_states:
                        v['h'] = np.zeros_like(layer.init_hidden_state)
                except AttributeError:  # Non-Recurrent Layer (FC or CNN layer)
                    pass
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

    def train_recurrent(self, X, y=None, batch_size=256, epochs=10000, sample_length=100,
                        normalize=False, temperature=1.0, plot=None, fit_test_data=False,
                        log_freq=100):
        self.prepare_sequence_data(X, y, normalize)

        self.init_momentum_velocity()

        training_logs_dict = \
            super().train_recurrent(batch_size=batch_size, epochs=epochs, plot=plot,
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

                if layer.hidden_state_grad is not None:
                    v['h'] = (v['h'] * self._mu) - (self._step_size * layer.hidden_state_grad)
                    layer.init_hidden_state += v['h']

            layer.reset()
