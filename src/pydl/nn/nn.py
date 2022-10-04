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


class NN:
    """The Neural Network Class."""

    def __init__(self, inputs, layers=[], name=None):
        self._inputs = inputs
        self._layers = layers
        self._network_out = None
        self._name = name

    @property
    def num_classes(self):
        return 2 if self._layers[-1].num_neurons == 1 else self._layers[-1].num_neurons

    @property
    def num_output_neurons(self):
        return self._layers[-1].num_neurons

    @property
    def weights(self):
        return [layer.weights for layer in self._layers]

    @property
    def layers(self):
        return self._layers

    def reinitialize_network(self):
        for layer in self._layers:
            layer.reinitialize_weights()

    def forward(self, inputs, inference=False):
        layer_inp = inputs
        for layer in self._layers:
            layer_out = layer.forward(layer_inp, inference=inference)
            if type(layer_out) is OrderedDict:  # If the current layer is RNN/LSTM/GRU
                # Stack layer outputs as dicts into a numpy array
                if layer.architecture_type == 'many_to_many':
                    seq_len = len(layer_out)
                    layer_inp = np.vstack([layer_out[t] for t in range(1, seq_len + 1)])
                else:  # 'many_to_one'
                    layer_inp = list(layer_out.values())[-1]
            else:
                layer_inp = layer_out
        self._network_out = layer_out
        return self._network_out

    def backward(self, inp_grad, reg_lambda=0, inputs=None):
        if self._network_out is None:
            _ = self.forward(inputs, inference=False)

        layer_inp_grad = inp_grad
        for layer in reversed(self._layers):
            if layer.type in ['RNN_Layer', 'LSTM_Layer', 'GRU_Layer'] and \
               type(layer_inp_grad) is not OrderedDict:
                # If the current layer is RNN, while the previous layer was not
                inp_grad_dict = OrderedDict()
                if layer.architecture_type == 'many_to_many':
                    for t, grad in enumerate(layer_inp_grad, start=1):
                        inp_grad_dict[t] = grad
                else:  # 'many_to_one'
                    inp_grad_dict[list(layer.output.keys())[-1]] = layer_inp_grad
                layer_inp_grad = inp_grad_dict
            elif layer.type not in ['RNN_Layer', 'LSTM_Layer', 'GRU_Layer'] and \
                    type(layer_inp_grad) is OrderedDict:
                # If the current layer is not RNN, while the previous layer was
                seq_len = len(layer_inp_grad)
                layer_inp_grad = np.vstack([layer_inp_grad[t] for t in range(1, seq_len + 1)])

            layer_out_grad = layer.backward(layer_inp_grad, reg_lambda)
            layer_inp_grad = layer_out_grad
        self._network_out = None
        return layer_out_grad

    def update_weights(self, alpha):
        for layer in self._layers:
            layer.update_weights(alpha)
        return
