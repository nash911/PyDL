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

from pydl.nn.layers import FC
from pydl.nn.conv import Conv
from pydl.nn.pool import Pool
from pydl.nn.batchnorm import BatchNorm
from pydl import conf


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

    def save(self):
        nn_dict = OrderedDict()

        for i, layer in enumerate(self._layers):
            nn_dict['Layer-' + str(i + 1)] = layer.save()

        return nn_dict

    def load(self, nn_dict):
        inputs = self._inputs
        for k, v in nn_dict.items():
            if v['type'] == 'FC_Layer':
                inputs = self.load_FC(inputs, v)
            elif v['type'] == 'Convolution_Layer':
                inputs = self.load_conv(inputs, v)
            elif v['type'] == 'Pooling_Layer':
                inputs = self.load_pool(inputs, v)

            try:
                if v['batchnorm'] is not False:
                    self._layers[-1].batchnorm = self.load_batchnorm(v['batchnorm'])
            except KeyError:
                pass

    def load_batchnorm(self, bn_dict):
        batchnorm = BatchNorm(gamma=np.array(bn_dict['gamma']), beta=np.array(bn_dict['beta']),
                              momentum=float(bn_dict['momentum']), name=bn_dict['name'])
        if bn_dict['avg_mean'] is not None:
            batchnorm.avg_mean = np.array(bn_dict['avg_mean'])
        if bn_dict['avg_var'] is not None:
            batchnorm.avg_var = np.array(bn_dict['avg_var'])
        return batchnorm

    def load_FC(self, inputs, layer_dict):
        fc_layer = FC(inputs, weights=np.array(layer_dict['weights'], dtype=conf.dtype),
                      bias=np.array(layer_dict['bias'], dtype=conf.dtype), batchnorm=False,
                      activation_fn=layer_dict['activation_fn'], dropout=None if
                      layer_dict['dropout'] is None else float(layer_dict['dropout']),
                      name=layer_dict['name'])
        self._layers.append(fc_layer)
        return fc_layer

    def load_conv(self, inputs, layer_dict):
        conv_layer = Conv(inputs, zero_padding=tuple(layer_dict['zero_padding']),
                          stride=layer_dict['stride'], weights=np.array(layer_dict['weights'],
                          dtype=conf.dtype), bias=np.array(layer_dict['bias'], dtype=conf.dtype),
                          activation_fn=layer_dict['activation_fn'],
                          force_adjust_output_shape=layer_dict['force_adjust_output_shape'],
                          dropout=None if layer_dict['dropout'] is None else
                          float(layer_dict['dropout']), name=layer_dict['name'])

        self._layers.append(conv_layer)
        return conv_layer

    def load_pool(self, inputs, layer_dict):
        pool_layer = Pool(inputs, receptive_field=tuple(layer_dict['receptive_field']),
                          padding=layer_dict['padding'], stride=tuple(layer_dict['stride']),
                          pool=layer_dict['pool'], name=layer_dict['name'])

        self._layers.append(pool_layer)
        return pool_layer
