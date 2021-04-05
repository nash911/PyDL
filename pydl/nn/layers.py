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

from pydl.nn.activations import Sigmoid
from pydl.nn.activations import SoftMax
from pydl.nn.activations import ReLU
from pydl import conf

activations = {'sigmoid' : Sigmoid,
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
    def output(self):
        return self._output

    @property
    def weights_grad(self):
        return self._weights_grad

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
    def forward(self, inputs):
        pass

    @abstractmethod
    def backward(self, inp_grad):
        pass


class FC(Layer):
    """The Hidden Layer Class
    """

    def __init__(self, inputs, num_neurons=None, weights=None, bias=False, activation_fn='Sigmoid',
                 name=None):
        super().__init__(name=name)
        self._inp_size = inputs.shape[-1]
        self._num_neurons = num_neurons
        self._has_bias = True if type(bias) == np.ndarray else bias
        self._activation_fn = activations[activation_fn.lower()]()

        if weights is not None:
            if num_neurons is not None:
                assert(weights.shape[-1] == num_neurons)
            else:
                self._num_neurons = weights.shape[-1]
            self._weights = weights
        else:
            self._weights = np.random.uniform(-1, 1, (self._inp_size, self._num_neurons))

        if type(bias) == np.ndarray:
            self._bias = bias
        elif bias:
            self._bias = np.random.uniform(-1, 1, (self._num_neurons))
        else:
            self._bias = None


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

    def forward(self, inputs):
        self._inputs = inputs
        self._output = self._activation_fn.forward(self.score_fn(inputs))
        return self._output

    def backward(self, inp_grad, reg_lambda=0, inputs=None):
        # dy/dz: Gradient of the output of the layer w.r.t the logits 'z'
        activation_grad = self._activation_fn.backward(inp_grad)

        self._weights_grad = self.weight_gradients(activation_grad, reg_lambda, inputs)
        if self._has_bias:
            self._bias_grad = self.bias_gradients(activation_grad)

        self._out_grad = self.input_gradients(activation_grad)
        return self._out_grad

    def update_weights(self, alpha):
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


    def forward(self, inputs):
        layer_inp = inputs
        for l in self._layers:
            layer_out = l.forward(layer_inp)
            layer_inp = layer_out
        self._network_out = layer_out
        return self._network_out


    def backward(self, inp_grad, reg_lambda=0, inputs=None):
        if self._network_out is None:
            _ = self.forward(inputs)

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
