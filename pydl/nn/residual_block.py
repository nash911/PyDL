# ------------------------------------------------------------------------
# MIT License
#
# Copyright (c) [2021] [Avinash Ranganath]
#
# This code is part of the library PyDL <https://github.com/nash911/PyDL>
# This code is licensed under MIT license (see LICENSE.txt for details)
# ------------------------------------------------------------------------

import numpy as np
import sys
import warnings

from pydl.nn.layers import Layer
from pydl.nn.activations import Linear
from pydl.nn.activations import Sigmoid
from pydl.nn.activations import Tanh
from pydl.nn.activations import SoftMax
from pydl.nn.activations import ReLU
from pydl.nn.conv import Conv
from pydl import conf

activations = {'linear' : Linear,
               'sigmoid' : Sigmoid,
               'tanh' : Tanh,
               'softmax' : SoftMax,
               'relu' : ReLU
              }

class ResidualBlock(Layer):
    """The Residual Block Class
    """

    def __init__(self, skip_connect, conv_layers, activation_fn='ReLU', name='Res_Block'):
        super().__init__(name=name)
        self._type = 'Res_Block'
        self._skip_connect = skip_connect
        self._block_layers = conv_layers
        self._block_out_activation_fn = activations[activation_fn.lower()]()

        self._block_layers[-1].activation = 'Linear'

        skip_size = self._skip_connect.shape[1:]
        block_out_shape = self._block_layers[-1].shape[1:]

        if skip_size != block_out_shape:
            self._skip_convolution = \
                Conv(self._skip_connect, receptive_field=(1,1), num_filters=block_out_shape[0],
                     zero_padding=0, stride=self._block_layers[0].stride, name='Skip_Conv',
                     weight_scale=1.0, xavier=True, activation_fn='Linear', batchnorm=True)
        else:
            self._skip_convolution = None


    # Getters
    # -------
    @property
    def shape(self):
        # (None, num_filters, output_height, output_width) of the last (convolution) layer
        return self._block_layers[-1].shape

    @property
    def layers(self):
        return self._block_layers

    @property
    def skip_convolution(self):
        return self._skip_convolution


    def forward(self, inputs, inference=None):
        layer_inp = inputs
        for l in self._block_layers:
            layer_out = l.forward(layer_inp, inference=inference)
            layer_inp = layer_out

        if self._skip_convolution is not None:
            skip_input = self._skip_convolution.forward(inputs, inference=inference)
        else:
            skip_input = inputs

        block_out = self._block_out_activation_fn.forward(layer_out + skip_input)
        return block_out


    def backward(self, inp_grad, reg_lambda=0, inputs=None):
        if len(inp_grad.shape) > 2: # The proceeding layer is a Convolution/Pooling layer
            pass
        else: # The proceeding layer is a FC layer
            # Reshape incoming gradients accordingly
            inp_grad = inp_grad.reshape(-1, *self._out_shape[1:])

        # dy/dz: Gradient of the output of the layer w.r.t the logits 'z'
        block_out_activation_grad = self._block_out_activation_fn.backward(inp_grad)

        layer_inp_grad = block_out_activation_grad
        for l in reversed(self._block_layers):
            layer_out_grad = l.backward(layer_inp_grad, reg_lambda)
            layer_inp_grad = layer_out_grad

        if self._skip_convolution is not None:
            skip_grad = self._skip_convolution.backward(block_out_activation_grad, reg_lambda)
        else:
            skip_grad = block_out_activation_grad

        block_out_grad = layer_out_grad + skip_grad

        return block_out_grad


    def update_weights(self, alpha):
        pass
