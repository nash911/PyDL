# ------------------------------------------------------------------------
# MIT License
#
# Copyright (c) [2022] [Avinash Ranganath]
#
# This code is part of the library PyDL <https://github.com/nash911/PyDL>
# This code is licensed under MIT license (see LICENSE.txt for details)
# ------------------------------------------------------------------------

import numpy as np

from pydl.optimizer.optimizer import Optimizer


class Adam(Optimizer):
    def __init__(self, nn, step_size=1e-2, beta_1=0.9, beta_2=0.999, name=None):
        super().__init__(step_size=step_size, name=name)

        self._beta_1 = beta_1
        self._beta_2 = beta_2
        self._m = list()
        self._v = list()

        self.init_adam_moment(nn)

    def init_adam_moment(self, nn):
        # Initialize cache to zero
        for layer in nn.layers:
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

    def update_network(self, nn, t):
        for layer, m, v in zip(nn.layers, self._m, self._v):
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
                # layer.weights += np.clip(-(self._step_size * m_hat_w) /
                #                          (np.sqrt(v_hat_w) + 1e-8), -1, 1)

                # Update Bias
                m_hat_b = m['b'] / (1.0 - self._beta_1**t)
                v_hat_b = v['b'] / (1.0 - self._beta_2**t)
                layer.bias += -(self._step_size * m_hat_b) / (np.sqrt(v_hat_b) + 1e-8)
                # layer.bias += np.clip(-(self._step_size * m_hat_b) /
                #                       (np.sqrt(v_hat_b) + 1e-8), -1, 1)

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
