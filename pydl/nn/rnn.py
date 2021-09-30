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

from pydl.nn.layers import Layer
from pydl.nn.activations import Linear
from pydl.nn.activations import Sigmoid
from pydl.nn.activations import Tanh
from pydl.nn.activations import SoftMax
from pydl.nn.activations import ReLU
from pydl.nn.dropout import Dropout
from pydl import conf

activations = {'linear': Linear,
               'sigmoid': Sigmoid,
               'tanh': Tanh,
               'softmax': SoftMax,
               'relu': ReLU
               }


class RNN(Layer):
    """The RNN Layer Class."""

    def __init__(self, inputs, num_neurons=None, weights=None, bias=True, seq_len=None, xavier=True,
                 weight_scale=1.0, activation_fn='Tanh', dropout=None, name=None):
        super().__init__(name=name)
        self._weights = OrderedDict()
        self._inputs = OrderedDict()
        self._output = OrderedDict()

        self._weights_grad = OrderedDict()
        self._out_grad = OrderedDict()

        self._type = 'RNN_Layer'
        self._num_neurons = num_neurons
        self._seq_len = seq_len
        self._inp_size = np.prod(inputs.shape[1:])
        self._weight_scale = weight_scale
        self._xavier = xavier
        self._has_bias = True if type(bias) == np.ndarray else bias
        self._activation_fn = [activations[activation_fn.lower()]() for _ in
                               range(self._seq_len + 1)]

        if weights is not None:
            # Assert that the weights dict has two values, each of which is a 2D array
            assert(len(weights) == 2)
            assert(len(weights['hidden'].shape) == 2 and len(weights['inp'].shape) == 2)

            if num_neurons is not None:
                assert(weights['hidden'].shape == (num_neurons, num_neurons) and
                       weights['inp'].shape[1] == num_neurons)
            else:
                assert(weights['hidden'].shape[0] == weights['hidden'].shape[1] and
                       weights['inp'].shape[1] == weights['hidden'].shape[0])
                self._num_neurons = weights['hidden'].shape[0]
            self._weights = weights
        else:
            self._weights['hidden'] = np.random.randn(self._num_neurons, self._num_neurons) * \
                weight_scale
            self._weights['inp'] = np.random.randn(self._inp_size, self._num_neurons) * weight_scale
            if xavier:
                # Apply Xavier Initialization
                if self._activation_fn[0].type.lower() == 'relu':
                    norm_fctr = np.sqrt((self._num_neurons + self._inp_size) / 2.0)
                else:
                    norm_fctr = np.sqrt(self._num_neurons + self._inp_size)
                self._weights['hidden'] /= norm_fctr
                self._weights['inp'] /= norm_fctr

        if type(bias) == np.ndarray:
            self._bias = bias
        elif bias:
            self._bias = np.zeros(self._num_neurons, dtype=conf.dtype)
        else:
            self._bias = None

        self._output[0] = np.zeros((1, self.num_neurons), dtype=conf.dtype)
        self.reset_gradients()

        if dropout is not None and dropout < 1.0:
            self._dropout = [Dropout(p=dropout, activation_fn=self._activation_fn[0].type) for _ in
                             range(self._seq_len + 1)]

    # Getters
    # -------
    @property
    def weights(self):
        return np.vstack((self._weights['hidden'], self._weights['inp']))

    @property
    def hidden_weights(self):
        return self._weights['hidden']

    @property
    def input_weights(self):
        return self._weights['inp']

    @property
    def shape(self):
        return (None, self._num_neurons)

    @property
    def size(self):
        return self.weights.size

    @property
    def num_neurons(self):
        return self._num_neurons

    @property
    def weights_grad(self):
        return np.vstack((self._weights_grad['hidden'], self._weights_grad['inp']))

    @property
    def hidden_weights_grad(self):
        return self._weights_grad['hidden']

    @property
    def input_weights_grad(self):
        return self._weights_grad['inp']

    # Setters
    # -------
    @weights.setter
    def weights(self, w):
        self.hidden_weights = w[:self.num_neurons, :]
        self.input_weights = w[self.num_neurons:, :]

    @hidden_weights.setter
    def hidden_weights(self, w):
        assert(w.shape == self._weights['hidden'].shape)
        self._weights['hidden'] = w

    @input_weights.setter
    def input_weights(self, w):
        assert(w.shape == self._weights['inp'].shape)
        self._weights['inp'] = w

    def reinitialize_weights(self, inputs=None, num_neurons=None):
        num_feat = self._inp_size if inputs is None else np.prod(inputs.shape[1:])
        num_neurons = self._num_neurons if num_neurons is None else num_neurons

        # Reinitialize weights
        self._weights['hidden'] = np.random.randn(num_neurons, num_neurons) * self._weight_scale
        self._weights['inp'] = np.random.randn(num_feat, num_neurons) * self._weight_scale
        if self._xavier:
            # Apply Xavier Initialization
            if self._activation_fn[0].type.lower() == 'relu':
                norm_fctr = np.sqrt((num_neurons + num_feat) / 2.0)
            else:
                norm_fctr = np.sqrt(num_neurons + num_feat)
            self._weights /= norm_fctr

        # Reset layer size
        self._inp_size = num_feat
        self._num_neurons = num_neurons

        if self._has_bias:
            self._bias = np.zeros(num_neurons, dtype=conf.dtype)

        self.reset_gradients()

    def reset_gradients(self):
        self._weights_grad['hidden'] = np.zeros_like(self._weights['hidden'])
        self._weights_grad['inp'] = np.zeros_like(self._weights['inp'])
        self._bias_grad = np.zeros_like(self._bias)
        self._out_grad = OrderedDict()

    def score_fn(self, inputs, weights=None):
        if weights is None:
            weighted_sum = np.matmul(inputs['h'], self._weights['hidden']) + \
                np.matmul(inputs['inp'], self._weights['inp'])
        else:
            weighted_sum = np.matmul(inputs['h'], weights['hidden']) + \
                np.matmul(inputs['inp'], weights['inp'])

        if self._has_bias:
            return weighted_sum + self._bias
        else:
            return weighted_sum

    def hidden_weight_gradients(self, inp_grad, inputs, reg_lambda=0):
        # dy/dwh: Gradient of the layer activation 'y' w.r.t the hidden weights 'Wh'
        hidden_weight_grad = inputs.reshape(-1, 1) * inp_grad

        if reg_lambda > 0:
            hidden_weight_grad += (reg_lambda * self._weights['hidden'])
        return hidden_weight_grad

    def input_weight_gradients(self, inp_grad, inputs, reg_lambda=0):
        # dy/dwx: Gradient of the layer activation 'y' w.r.t the input weights 'Wx'
        inp_weight_grad = inputs.reshape(-1, 1) * inp_grad

        if reg_lambda > 0:
            inp_weight_grad += (reg_lambda * self._weights['inp'])
        return inp_weight_grad

    def bias_gradients(self, inp_grad):
        if not self._has_bias:
            return None
        else:
            # dy/db: Gradient of the layer activation 'y' w.r.t the bias 'b'
            grad = inp_grad.reshape(-1)
            return grad

    def input_gradients(self, inp_grad, summed=True):
        # dy/dx: Gradient of the layer activation 'y' w.r.t the inputs 'X'
        out_grad = self._weights['inp'] * inp_grad

        if summed:
            out_grad = np.sum(out_grad, axis=-1, keepdims=False)
        return out_grad

    def hidden_gradients(self, inp_grad, summed=True):
        # dy/dx: Gradient of the layer activation 'y' w.r.t the hidden layer inputs 'H_t-1'
        hidden_grad = self._weights['hidden'] * inp_grad

        if summed:
            hidden_grad = np.sum(hidden_grad, axis=-1, keepdims=False)
        return hidden_grad

    def forward(self, inputs, inference=False, mask=None, temperature=1.0):
        if len(inputs.shape) > 2:  # Preceeding layer is a Convolution/Pooling layer or 3D inputs
            # Unroll inputs
            batch_size = inputs.shape[0]
            inputs = inputs.reshape(batch_size, -1)

        for t, inp in enumerate(inputs, start=1):
            # Store inputs in dict for backprop
            self._inputs[t] = inp.reshape(1, -1)

            # Sum of weighted inputs
            z = self.score_fn({'h': self._output[t - 1], 'inp': inp})

            # Nonlinearity Activation
            self._output[t] = self._activation_fn[t].forward(z)

            # Dropout
            if self._dropout is not None:
                if not inference:  # Training step
                    if self.dropout_mask is None:
                        drop_mask = None if mask is None else mask[t - 1]
                    else:
                        drop_mask = self.dropout_mask[t - 1]

                    # Apply Dropout Mask
                    self._output[t] = self._dropout[t].forward(self._output[t], drop_mask)
                else:  # Inference
                    if self._activation_fn[t].type in ['Sigmoid', 'Tanh', 'SoftMax']:
                        self._output[t] *= self.dropout[t].p
                    else:  # Activation Fn. ∈ {'Linear', 'ReLU'}
                        pass  # Do nothing - Inverse Dropout

        return self._output

    def backward(self, inp_grad, reg_lambda=0, inputs=None):
        hidden_grad = np.zeros((1, self.num_neurons), dtype=conf.dtype)
        for t in reversed(range(1, len(inp_grad) + 1)):
            # Add gradients from the next (t+1) hidden sequence and from the layer above (l+1)
            grad = hidden_grad + inp_grad[t]

            # Backpropagating through Dropout
            if self._dropout is not None:
                drop_grad = self._dropout[t].backward(grad)
            else:
                drop_grad = grad

            # dy/dz: Gradient of the layer output w.r.t the logits 'z'
            activation_grad = self._activation_fn[t].backward(drop_grad)

            # dy/dw: Gradient of the layer output w.r.t the weights 'wh' and 'wx'
            self._weights_grad['hidden'] += \
                self.hidden_weight_gradients(activation_grad, self._output[t - 1], reg_lambda)
            self._weights_grad['inp'] += \
                self.input_weight_gradients(activation_grad, self._inputs[t], reg_lambda)

            # dy/db: Gradient of the layer output w.r.t the bias
            if self._has_bias:
                self._bias_grad += self.bias_gradients(activation_grad)

            # dy/di: Gradient of the layer output w.r.t the layer's inputs 'i' and 'h_t-1'
            self._out_grad[t] = self.input_gradients(activation_grad)
            hidden_grad = self.hidden_gradients(activation_grad)

        return self._out_grad

    def update_weights(self, alpha):
        self._weights['hidden'] += -alpha * self._weights_grad['hidden']
        self._weights['inp'] += -alpha * self._weights_grad['inp']

        if self._has_bias:
            self._bias += -alpha * self._bias_grad

        self.reset()

    def reset(self):
        self.reset_gradients()
        self.reset_internal_states(hidden_state='previous_state')

    def reset_internal_states(self, hidden_state=None):
        try:
            if hidden_state.lower() == 'previous_state':
                hidden_state = list(self._output.values())[-1]
        except AttributeError:
            pass

        self._inputs = OrderedDict()
        self._output = OrderedDict()
        if hidden_state is None:
            self._output[0] = np.zeros((1, self.num_neurons), dtype=conf.dtype)
        else:
            self._output[0] = hidden_state
