# ------------------------------------------------------------------------
# MIT License
#
# Copyright (c) [2022] [Avinash Ranganath]
#
# This code is part of the library PyDL <https://github.com/nash911/PyDL>
# This code is licensed under MIT license (see LICENSE.txt for details)
# ------------------------------------------------------------------------

from pydl.optimizer.optimizer import Optimizer


class SGD(Optimizer):
    def __init__(self, step_size=1e-4, name=None):
        super().__init__(step_size=step_size, name=name)

    def update_network(self, nn, t=None):
        for layer in nn.layers:
            if layer.type in ['FC_Layer', 'Convolution_Layer', 'RNN_Layer', 'LSTM_Layer',
                              'GRU_Layer']:
                # Update Weights
                layer.weights += -self._step_size * layer.weights_grad
                # layer.weights += np.clip(-self._step_size * layer.weights_grad, -1, 1)

                # Update Bias
                if layer.bias is not None:
                    layer.bias += -self._step_size * layer.bias_grad
                    # layer.bias += np.clip(-self._step_size * layer.bias_grad, -1, 1)

                # Update Cell State
                if layer.cell_state_grad is not None:
                    layer.init_cell_state += -self._step_size * layer.cell_state_grad

                # Update Hidden State
                if layer.hidden_state_grad is not None:
                    layer.init_hidden_state += -self._step_size * layer.hidden_state_grad
            layer.reset()
