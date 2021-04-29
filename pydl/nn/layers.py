# ------------------------------------------------------------------------
# MIT License
#
# Copyright (c) [2021] [Avinash Ranganath]
#
# This code is part of the library PyDL <https://github.com/nash911/PyDL>
# This code is licensed under MIT license (see LICENSE.txt for details)
# ------------------------------------------------------------------------

from abc import ABC, abstractmethod
import numpy as np

from pydl.nn.activations import Linear
from pydl.nn.activations import Sigmoid
from pydl.nn.activations import Tanh
from pydl.nn.activations import SoftMax
from pydl.nn.activations import ReLU
from pydl.nn.batchnorm import BatchNorm
from pydl.nn.dropout import Dropout
from pydl import conf

activations = {'linear' : Linear,
               'sigmoid' : Sigmoid,
               'tanh' : Tanh,
               'softmax' : SoftMax,
               'relu' : ReLU
              }


class Layer(ABC):
    """An abstract class defining the interface of the Layer.
    Args:
        name (str): Name of the layer.
    """

    def __init__(self, name=None):
        self._name = name

        self._inputs = None
        self._output = None

        self._weights = None
        self._bias = None

        self._weights_grad = None
        self._bias_grad = None
        self._out_grad = None

    # Getters
    # -------
    @property
    def weights(self):
        return self._weights

    @property
    def bias(self):
        return self._bias

    @property
    def name(self):
        return self._name

    @property
    def shape(self):
        return (None, self._num_neurons)

    @property
    def size(self):
        return self._weights.size

    @property
    def num_neurons(self):
        return self._num_neurons

    @property
    def activation(self):
        return self._activation_fn.type

    @property
    def has_batchnorm(self):
        return False if self._batchnorm is None else True

    @property
    def batchnorm(self):
        return self._batchnorm

    @property
    def dropout(self):
        return self._dropout

    @property
    def output(self):
        return self._output

    @property
    def weights_grad(self):
        return self._weights_grad

    @property
    def bias_grad(self):
        return self._bias_grad

    @property
    def out_grad(self):
        return self._out_grad

    # Setters
    # -------
    @weights.setter
    def weights(self, w):
        assert(w.shape == self._weights.shape)
        self._weights = w

    @bias.setter
    def bias(self, b):
        assert(b.shape == self._bias.shape)
        self._bias = b

    # Abstract Methods
    # ----------------
    @abstractmethod
    def forward(self, inputs, inference=False, mask=None):
        pass

    @abstractmethod
    def backward(self, inp_grad):
        pass


class FC(Layer):
    """The Hidden Layer Class
    """

    def __init__(self, inputs, num_neurons=None, weights=None, bias=True, weight_scale=1.0,
                 xavier=True, activation_fn='Sigmoid', batchnorm=False, dropout=None, name=None):
        super().__init__(name=name)
        self._inp_size = inputs.shape[-1]
        self._num_neurons = num_neurons
        self._weight_scale = weight_scale
        self._xavier = xavier
        self._has_bias = True if type(bias) == np.ndarray else bias
        self._activation_fn = activations[activation_fn.lower()]()

        if weights is not None:
            if num_neurons is not None:
                assert(weights.shape[-1] == num_neurons)
            else:
                self._num_neurons = weights.shape[-1]
            self._weights = weights
        else:
            self._weights = np.random.randn(self._inp_size, self._num_neurons) * weight_scale
            if xavier:
                # Apply Xavier Initialization
                if self._activation_fn.type.lower() == 'relu':
                    norm_fctr = np.sqrt(self._inp_size/2.0)
                else:
                    norm_fctr = np.sqrt(self._inp_size)
                self._weights /= norm_fctr

        if type(bias) == np.ndarray:
            self._bias = bias
        elif bias:
            self._bias = np.zeros(self._num_neurons, dtype=conf.dtype)
        else:
            self._bias = None

        if batchnorm:
            self._batchnorm = BatchNorm(feature_size=self._num_neurons)
        else:
            self._batchnorm = None

        if dropout is not None:
            self._dropout = Dropout(p=dropout, activation_fn=self._activation_fn.type)
        else:
            self._dropout = None


    def reinitialize_weights(self, inputs=None, num_neurons=None):
        num_feat = self._inp_size if inputs is None else inputs.shape[-1]
        num_neurons = self._num_neurons if num_neurons is None else num_neurons

        # Reinitialize weights
        self._weights = np.random.randn(num_feat, num_neurons) * self._weight_scale
        if self._xavier:
            # Apply Xavier Initialization
            if self._activation_fn.type.lower() == 'relu':
                norm_fctr = np.sqrt(num_feat/2.0)
            else:
                norm_fctr = np.sqrt(num_feat)
            self._weights /= norm_fctr

        # Reset layer size
        self._inp_size = num_feat
        self._num_neurons = num_neurons

        if self._has_bias:
            self._bias = np.zeros(num_neurons, dtype=conf.dtype)

        if self._batchnorm is not None:
            self._batchnorm.reinitialize_params(feature_size=num_neurons)


    def score_fn(self, inputs, weights=None):
        self._inputs = inputs
        if weights is None:
            weighted_sum = np.matmul(self._inputs, self._weights)
        else:
            weighted_sum = np.matmul(self._inputs, weights)

        if self._has_bias:
            return weighted_sum + self._bias
        else:
            return weighted_sum

    def weight_gradients(self, inp_grad, reg_lambda=0, inputs=None, summed=True):
        if inputs is not None:
            self._inputs = inputs
        assert(self._inputs is not None)

        # dy/dw: Gradient of the layer activation 'y' w.r.t the weights 'w'
        grad = self._inputs[:,:,np.newaxis] * inp_grad[:,np.newaxis,:]
        if summed:
            grad = np.sum(grad, axis=0, keepdims=False)

        if reg_lambda > 0:
            grad += (reg_lambda * self._weights)
        return grad

    def bias_gradients(self, inp_grad, summed=True):
        if not self._has_bias:
            return None
        else:
            # dy/db: Gradient of the layer activation 'y' w.r.t the bias 'b'
            grad = inp_grad
            if summed:
                grad = np.sum(grad, axis=0, keepdims=False)
            return grad

    def input_gradients(self, inp_grad, summed=True):
        # dy/dx: Gradient of the layer activation 'y' w.r.t the inputs 'X'
        grad = self._weights[np.newaxis,:,:] * inp_grad[:,np.newaxis,:]
        if summed:
            grad = np.sum(grad, axis=-1, keepdims=False)
        return grad

    def forward(self, inputs, inference=False, mask=None):
        self._inputs = inputs

        # Sum of weighted inputs
        z = self.score_fn(inputs)

        # Batchnorm
        if self._batchnorm is not None:
            z = self._batchnorm.forward(z)

        # Nonlinearity Activation
        self._output = self._activation_fn.forward(z)

        # Dropout
        if self._dropout is not None:
            if not inference: # Training step
                # Apply Dropout Mask
                self._output = self._dropout.forward(self._output, mask)
            else: # Inference
                 if self._activation_fn.type in ['Sigmoid', 'Tanh', 'SoftMax']:
                     self._output *= self.dropout.p
                 else: # Activation Fn. ∈ {'Linear', 'ReLU'}
                     pass # Do nothing - Inverse Dropout

        return self._output

    def backward(self, inp_grad, reg_lambda=0, inputs=None):
        if self._dropout is not None:
            drop_grad = self._dropout.backward(inp_grad)
        else:
            drop_grad = inp_grad

        # dy/dz: Gradient of the output of the layer w.r.t the logits 'z'
        activation_grad = self._activation_fn.backward(drop_grad)

        if self._batchnorm is not None:
            batch_grad = self._batchnorm.backward(activation_grad)
        else:
            batch_grad = activation_grad

        self._weights_grad = self.weight_gradients(batch_grad, reg_lambda, inputs)
        if self._has_bias:
            self._bias_grad = self.bias_gradients(batch_grad)

        self._out_grad = self.input_gradients(batch_grad)
        return self._out_grad

    def update_weights(self, alpha):
        if self._batchnorm is not None:
            self._batchnorm.update_params(alpha)

        self.weights += self._weights_grad * alpha
        if self._has_bias:
            self.bias += self._bias_grad * alpha


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
