# ------------------------------------------------------------------------
# MIT License
#
# Copyright (c) [2021] [Avinash Ranganath]
#
# This code is part of the library PyDL <https://github.com/nash911/PyDL>
# This code is licensed under MIT license (see LICENSE.txt for details)
# ------------------------------------------------------------------------

import numpy as np
from collections import OrderedDict

from pydl.training.plain_training import PlainTraining
from pydl.training.recurrent_training import RecurrentTraining


class Adam(PlainTraining, RecurrentTraining):
    def __init__(self, nn=None, step_size=1e-2, beta_1=0.9, beta_2=0.999, reg_lambda=0,
                 train_size=70, test_size=30, activatin_type=None, regression=False, save=False,
                 name=None):
        super().__init__(nn=nn, step_size=step_size, reg_lambda=reg_lambda, train_size=train_size,
                         test_size=test_size, activatin_type=activatin_type, regression=regression,
                         save=save, name=name)
        self._beta_1 = beta_1
        self._beta_2 = beta_2
        self._m = list()
        self._v = list()

        if save:
            training_dict = OrderedDict()
            training_dict['step_size'] = float(step_size)
            training_dict['beta_1'] = float(beta_1)
            training_dict['beta_2'] = float(beta_2)
            training_dict['reg_lambda'] = float(reg_lambda)
            training_dict['train_size'] = int(train_size) if train_size > 1.0 else float(train_size)
            training_dict['test_size'] = int(test_size) if test_size > 1.0 else float(test_size)
            training_dict['regression'] = regression
            self._save_dict['training'] = training_dict

    def init_adam_moment(self):
        # Initialize cache to zero
        for layer in self._nn.layers:
            if layer.type in ['FC_Layer', 'Convolution_Layer', 'RNN_Layer', 'LSTM_Layer',
                              'GRU_Layer']:
                m = {'w': np.zeros_like(layer.weights),
                     'b': np.zeros_like(layer.bias)}

                v = {'w': np.zeros_like(layer.weights),
                     'b': np.zeros_like(layer.bias)}

                try:  # A Recurrent Layer (Eg: RNN/LSTM/GRU layer)
                    if layer.tune_internal_states:
                        m['h'] = np.zeros_like(layer.init_hidden_state)
                        v['h'] = np.zeros_like(layer.init_hidden_state)

                        try:  # LSTM layer
                            m['c'] = np.zeros_like(layer.init_cell_state)
                            v['c'] = np.zeros_like(layer.init_cell_state)
                        except AttributeError:  # non-LSTM recurrent layer
                            pass
                except AttributeError:  # Non-Recurrent Layer (FC or CNN layer)
                    pass
            else:
                m = None
                v = None
            self._m.append(m)
            self._v.append(v)

    def train(self, X, y, normalize=None, dims=None, shuffle=True, batch_size=256, epochs=10000,
              y_onehot=False, plot=None, log_freq=100, model_file=None):
        self.prepare_data(X=X, y=y, normalize=normalize, dims=dims, shuffle=shuffle,
                          batch_size=batch_size, y_onehot=y_onehot)
        self.init_adam_moment()

        training_logs_dict = super().train(batch_size=batch_size, epochs=epochs, plot=plot,
                                           log_freq=log_freq, model_file=model_file)
        return training_logs_dict

    def train_recurrent(self, X, y=None, batch_size=256, epochs=10000, sample_length=100,
                        normalize=None, data_diff=False, temperature=1.0, plot=None,
                        fit_test_data=False, log_freq=100):
        self.prepare_sequence_data(X, y, normalize, data_diff)

        self.init_adam_moment()

        training_logs_dict = \
            super().train_recurrent(batch_size=batch_size, epochs=epochs, plot=plot,
                                    sample_length=sample_length, normalize=normalize,
                                    data_diff=data_diff, temperature=temperature,
                                    fit_test_data=fit_test_data, log_freq=log_freq)
        return training_logs_dict

    def update_network(self, t):
        for layer, m, v in zip(self._nn.layers, self._m, self._v):
            if layer.type in ['FC_Layer', 'Convolution_Layer', 'RNN_Layer', 'LSTM_Layer',
                              'GRU_Layer']:
                # First order moment update
                m['w'] = (self._beta_1 * m['w']) + ((1 - self._beta_1) * layer.weights_grad)
                m['b'] = (self._beta_1 * m['b']) + ((1 - self._beta_1) * layer.bias_grad)
                if layer.cell_state_grad is not None:
                    m['c'] = \
                        (self._beta_1 * m['c']) + ((1 - self._beta_1) * layer.cell_state_grad)
                if layer.hidden_state_grad is not None:
                    m['h'] = \
                        (self._beta_1 * m['h']) + ((1 - self._beta_1) * layer.hidden_state_grad)

                # Second order moment update
                v['w'] = (self._beta_2 * v['w']) + ((1 - self._beta_2) * (layer.weights_grad**2))
                v['b'] = (self._beta_2 * v['b']) + ((1 - self._beta_2) * (layer.bias_grad**2))
                if layer.cell_state_grad is not None:
                    v['c'] = (self._beta_2 * v['c']) + \
                             ((1 - self._beta_2) * (layer.cell_state_grad**2))
                if layer.hidden_state_grad is not None:
                    v['h'] = (self._beta_2 * v['h']) + \
                             ((1 - self._beta_2) * (layer.hidden_state_grad**2))

                # Update Weights
                m_hat_w = m['w'] / (1.0 - self._beta_1**t)
                v_hat_w = v['w'] / (1.0 - self._beta_2**t)
                layer.weights += -(self._step_size * m_hat_w) / (np.sqrt(v_hat_w) + 1e-8)

                # Update Bias
                m_hat_b = m['b'] / (1.0 - self._beta_1**t)
                v_hat_b = v['b'] / (1.0 - self._beta_2**t)
                layer.bias += -(self._step_size * m_hat_b) / (np.sqrt(v_hat_b) + 1e-8)

                # Update Cell State
                if layer.cell_state_grad is not None:
                    m_hat_c = m['c'] / (1.0 - self._beta_1**t)
                    v_hat_c = v['c'] / (1.0 - self._beta_2**t)
                    layer.init_cell_state += \
                        -(self._step_size * m_hat_c) / (np.sqrt(v_hat_c) + 1e-8)

                # Update Hidden State
                if layer.hidden_state_grad is not None:
                    m_hat_h = m['h'] / (1.0 - self._beta_1**t)
                    v_hat_h = v['h'] / (1.0 - self._beta_2**t)
                    layer.init_hidden_state += \
                        -(self._step_size * m_hat_h) / (np.sqrt(v_hat_h) + 1e-8)

            # Reset Layer
            layer.reset()
