# ------------------------------------------------------------------------
# MIT License
#
# Copyright (c) [2021] [Avinash Ranganath]
#
# This code is part of the library PyDL <https://github.com/nash911/PyDL>
# This code is licensed under MIT license (see LICENSE.txt for details)
# ------------------------------------------------------------------------

import numpy as np

from pydl import conf


class NN:
    """The Neural Network Class
    """

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
        return [l.weights for l in self._layers]


    @property
    def layers(self):
        return self._layers


    def reinitialize_network(self):
        for l in self._layers:
            l.reinitialize_weights()


    def forward(self, inputs, inference=False):
        layer_inp = inputs
        for l in self._layers:
            layer_out = l.forward(layer_inp, inference=inference)
            layer_inp = layer_out
        self._network_out = layer_out
        return self._network_out


    def backward(self, inp_grad, reg_lambda=0, inputs=None):
        if self._network_out is None:
            _ = self.forward(inputs, inference=False)

        layer_inp_grad = inp_grad
        for l in reversed(self._layers):
            layer_out_grad = l.backward(layer_inp_grad, reg_lambda)
            layer_inp_grad = layer_out_grad
        self._network_out = None
        return layer_out_grad


    def update_weights(self, alpha):
        for l in self._layers:
            l.update_weights(alpha)
        return
