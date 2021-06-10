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
from pydl.nn.batchnorm import BatchNorm
from pydl.nn.dropout import Dropout
from pydl import conf

activations = {'linear' : Linear,
               'sigmoid' : Sigmoid,
               'tanh' : Tanh,
               'softmax' : SoftMax,
               'relu' : ReLU
              }


class Conv(Layer):
    """The Convolution Layer Class
    """

    def __init__(self, inputs, receptive_field=None, num_filters=None, zero_padding=0, stride=1,
                 weights=None, bias=True, weight_scale=1.0, xavier=True, activation_fn='ReLU',
                 batchnorm=False, dropout=None, name='ConvLayer'):
        super().__init__(name=name)
        self._type = 'Convolution_Layer'
        self._inp_shape = inputs.shape[1:] # Input volume --> [depth, height, width]
        self._receptive_field = receptive_field # Filter's (height, width)
        self._num_filters = num_filters
        self._zero_padding = zero_padding
        self._stride = stride
        self._weight_scale = weight_scale
        self._xavier = xavier
        self._has_bias = True if type(bias) == np.ndarray else bias
        self._activation_fn = activations[activation_fn.lower()]()

        # Initialize Weights
        self.init_weights(weights)

        # Check if the hyperparameter are valid for the given input volume, and set output volume shape
        self.set_output_volume_shape()

        # Calculate unroll indices of input volume
        self.calculate_unroll_indices()

        # Initialize Bias
        if type(bias) == np.ndarray:
            assert(bias.size == self._num_filters)
            self._bias = bias.reshape(-1, 1)
        elif bias:
            self._bias = np.zeros((self._num_filters, 1), dtype=conf.dtype)
        else:
            self._bias = None

        if batchnorm:
            self._batchnorm = BatchNorm(feature_size=self._out_shape[1:])
        else:
            self._batchnorm = None

        if dropout is not None and dropout < 1.0:
            self._dropout = Dropout(p=dropout, activation_fn=self._activation_fn.type)
        else:
            self._dropout = None


    # Getters
    # -------
    @property
    def weights(self):
        return self._weights

    @property
    def bias(self):
        return self._bias

    @property
    def shape(self):
        # (None, num_filters, output_height, output_width)
        return self._out_shape

    @property
    def size(self):
        # (num_filters x output_height x output_width)
        return self._out_size

    @property
    def receptive_field(self):
        # (filter_height, filter_width)
        return self._receptive_field

    @property
    def filter_shape(self):
        # (input_volume_depth, filter_height, filter_width)
        return self._filter_shape

    @property
    def num_filters(self):
        return self._num_filters

    @property
    def zero_padding(self):
        return self._zero_padding

    @property
    def stride(self):
        return self._stride


    # Setters
    # -------
    @weights.setter
    def weights(self, w):
        assert(w.shape == self._weights.shape)
        self._weights = w

    @bias.setter
    def bias(self, b):
        try:
            assert(b.shape == self._bias.shape)
            self._bias = b
        except:
            assert(b.size == self._bias.size)
            self._bias = b.reshape(-1, 1)


    def init_weights(self, weights):
        # Initialize Weights
        # Weight Dimension - (num_filters, input_depth, filter_height, filter_width)
        if weights is not None:
            # The first dimension of weights should be the no. of filters in the layer
            if self._num_filters is not None:
                assert(weights.shape[0] == self._num_filters)
            else:
                self._num_filters = weights.shape[0]

            # Check filter depth - Should be the same as the depth of the input volume
            assert(weights.shape[1] == self._inp_shape[0])

            # The two innermost dimensions of the weight tensor are filter height (rows) and
            # width (cols) respectively --> Which is same as the dimensions of the receptive field
            if self._receptive_field is not None:
                assert(weights.shape[2:] == self._receptive_field)
            else:
                self._receptive_field = weights.shape[2:]

            # Filter_shape --> (input_volume_depth, filter_height, filter_width)
            self._filter_shape = tuple((self._inp_shape[0], *self._receptive_field))

            # Size of each filter in the layer --> (depth x height x width)
            self._filter_size = np.prod(self._filter_shape)

            self._weights = weights
        else:
            assert(self._receptive_field is not None and self._num_filters is not None)

            # Filter_shape --> (input_volume_depth, filter_height, filter_width)
            self._filter_shape = tuple((self._inp_shape[0], *self._receptive_field))

            # Size of each filter in the layer --> (depth x height x width)
            self._filter_size = np.prod(self._filter_shape)

            # Initialize weights from a normal distribution
            self._weights = np.random.randn(self._num_filters, *self._filter_shape) * \
                            self._weight_scale

            if self._xavier:
                # Apply Xavier Initialization
                if self._activation_fn.type.lower() == 'relu':
                    norm_fctr = np.sqrt(self._filter_size/2.0)
                else:
                    norm_fctr = np.sqrt(self._filter_size)
                self._weights /= norm_fctr


    def reinitialize_weights(self):
        # Initialize weights from a normal distribution
        self._weights = np.random.randn(self._num_filters, self._filter_shape) * self._weight_scale

        if self._xavier:
            # Apply Xavier Initialization
            if self._activation_fn.type.lower() == 'relu':
                norm_fctr = np.sqrt(self._filter_size/2.0)
            else:
                norm_fctr = np.sqrt(self._filter_size)
            self._weights /= norm_fctr

        if self._has_bias:
            self._bias = np.zeros((self._num_filters, 1), dtype=conf.dtype)

        if self._batchnorm is not None:
            self._batchnorm.reinitialize_params(feature_size=self._out_shape[1:])


    def set_output_volume_shape(self):
        i_h = self._inp_shape[1]
        i_w = self._inp_shape[2]
        f_h = self._receptive_field[0]
        f_w = self._receptive_field[1]

        # Calculate output volume's height and width:
        # (W - F + 2P)/S + 1 --> W: Input volume size | F: Receptive field size | P: Pad | S: Stride
        o_h = ((i_h - f_h + (2*self._zero_padding)) / self._stride) + 1
        o_w = ((i_w - f_w + (2*self._zero_padding)) / self._stride) + 1

        # Assert if the layer hyperparameter combination is valid for the layer's input shape
        if(o_h % 1 != 0 or o_w % 1 != 0):
            # print("Error\n Layer: %s\n Input shape: (%d, %d, %d)\n Receptive field: (%d, %d) \
            #        \n Zero passing: %d\n Stride: %d\n" %
            #       (self.name, self._inp_shape[0], self._inp_shape[1], self._inp_shape[2], f_h, f_w,
            #        self._zero_padding, self._stride))
            sys.exit("Error: Layer parameters (receptive fiels, stride, padding) does not fit " +
                     "the input volume.")
        elif(f_h > i_h or f_w > i_w):
            sys.exit("Error: Kernal size is greater than input size.")
        else:
            self._out_shape = tuple((None, self._num_filters, int(o_h), int(o_w)))
            self._out_size = np.prod(self._out_shape[1:])


    def calculate_unroll_indices(self): # Conv-Algo-5
        inp_d = self._inp_shape[0]
        inp_h = self._inp_shape[1]
        inp_w = self._inp_shape[2]
        ker_h = self._receptive_field[0]
        ker_w = self._receptive_field[1]

        window_row_inds = inp_h - (ker_h-1) + (2*self._zero_padding)
        window_col_inds = inp_w - (ker_w-1) + (2*self._zero_padding)

        out_h = int((inp_h - ker_h + (2*self._zero_padding)) / self._stride) + 1
        out_w = int((inp_w - ker_w + (2*self._zero_padding)) / self._stride) + 1

        r0 = np.repeat(np.arange(ker_h), ker_w)
        r1 = np.repeat(np.arange(window_row_inds, step=self._stride), out_w)

        c0 = np.tile(np.arange(ker_w), ker_h)
        c1 = np.tile(np.arange(window_col_inds, step=self._stride), out_h)

        r = r0.reshape(-1,1) + r1.reshape(1,-1)
        c = c0.reshape(-1,1) + c1.reshape(1,-1)

        self._row_inds = np.tile(r, reps=(inp_d, 1))
        self._col_inds = np.tile(c, reps=(inp_d, 1))

        self._slice_inds = np.tile(np.repeat(np.arange(inp_d), ker_h*ker_w), reps=(out_h*out_w, 1)).T


    def score_fn(self, inputs, weights=None):
        # Zero-pad input volume based on the setting
        if self._zero_padding > 0:
            pad = self._zero_padding
            padded_inputs = np.pad(inputs, ((0,0),(0,0),(pad,pad),(pad,pad)), 'constant')
        else:
            padded_inputs = inputs

        # Unroll input volume to shape: (batch, out_rows*out_cols, filter_size[dxhxw])
        self._unrolled_inputs = padded_inputs[:, self._slice_inds, self._row_inds, self._col_inds]

        # Unroll filter weights to shape: (num_filters, filter_size[dxhxw])
        if weights is None:
            weights_reshaped = self._weights.reshape(self._num_filters, -1)
        else:
            weights_reshaped = weights.reshape(self._num_filters, -1)

        # Weighted sum of each receptive field with every filter
        weighted_sum = np.matmul(weights_reshaped, self._unrolled_inputs)

        # Add the bias term
        if self._has_bias:
            weighted_sum += self._bias

        # Reshape output to it's correct volumetric shape: (batch, out_depth, out_height, out_width)
        weighted_sum_reshaped = weighted_sum.reshape(-1, *self._out_shape[1:])

        return weighted_sum_reshaped


    def weight_gradients(self, inp_grad, reg_lambda=0, inputs=None, summed=True):
        if inputs is not None:
            self._unrolled_inputs = inputs
        assert(self._unrolled_inputs is not None)

        batch = inp_grad.shape[0]

        # dy/dw: Gradient of the layer activation 'y' w.r.t the weights 'w'
        grad = self._unrolled_inputs[:,np.newaxis,:,:] * \
               inp_grad.reshape(batch, self._num_filters, 1, -1)

        if summed:
            grad = np.sum(grad, axis=(0,3), keepdims=False)
            grad = grad.reshape(self._num_filters, *self._filter_shape)

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
                grad = np.sum(grad, axis=(0,2,3), keepdims=False).reshape(-1, 1)
            return grad


    def input_gradients(self, inp_grad, summed=True):  # Conv-Bck-Algo-4
        # dy/dx: Gradient of the layer activation 'y' w.r.t the inputs 'X'
        batch_size = inp_grad.shape[0]
        inp_d = self._inp_shape[0]
        padded_inp_h = self._inp_shape[1] + (2*self._zero_padding)
        padded_inp_w = self._inp_shape[2] + (2*self._zero_padding)

        # weight * inp_grads
        w_inp_grad = self._weights.reshape(self._num_filters, -1, 1) * \
                     inp_grad.reshape(batch_size, self._num_filters, 1, -1)
        w_inp_grad = np.sum(w_inp_grad, axis=1, keepdims=False)

        batch_inds = np.arange(batch_size).reshape(-1, 1, 1)
        out_grads = np.zeros((batch_size, inp_d, padded_inp_h, padded_inp_w))
        np.add.at(out_grads, [batch_inds, self._slice_inds, self._row_inds, self._col_inds],
                  w_inp_grad)

        if self._zero_padding > 0:
            pad = self._zero_padding
            out_grads = out_grads[:,:,pad:-pad,pad:-pad]

        return out_grads


    def forward(self, inputs, inference=False, mask=None):
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
                self._output = self._dropout.forward(self._output, mask if self.dropout_mask is None
                                                     else self.dropout_mask)
            else: # Inference
                 if self._activation_fn.type in ['Sigmoid', 'Tanh', 'SoftMax']:
                     self._output *= self.dropout.p
                 else: # Activation Fn. ∈ {'Linear', 'ReLU'}
                     pass # Do nothing - Inverse Dropout

        return self._output


    def backward(self, inp_grad, reg_lambda=0, inputs=None):
        if len(inp_grad.shape) > 2: # The proceeding layer is a Convolution/Pooling layer
            pass
        else: # The proceeding layer is a FC layer
            # Reshape incoming gradients accordingly
            inp_grad = inp_grad.reshape(-1, *self._out_shape[1:])

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
