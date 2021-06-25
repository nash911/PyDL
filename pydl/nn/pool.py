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
from pydl import conf


class Pool(Layer):
    """The Pooling Layer Class
    """

    def __init__(self, inputs, receptive_field=None, padding='SAME', stride=None, pool='MAX',
                 name='Pooling_Layer'):
        super().__init__(name=name)
        self._type = 'Pooling_Layer'
        self._inp_shape = inputs.shape[1:] # Input volume --> [depth, height, width]

        if pool.upper() in ['MAX', 'AVG']:
            self._pool = pool.upper()
        else:
            sys.exit("Error: Unknown pooling opration: " + pool + " in layer: ", name)

        # Set receptive field dimensions
        if receptive_field is None:
            r_field_height = self._inp_shape[1]
            r_field_width = self._inp_shape[2]
        else:
            r_field_height = receptive_field[0]
            r_field_width = receptive_field[1]

        # Reduce receptive field to fit input shape, if larger
        if self._inp_shape[1] < r_field_height:
            warnings.warn("\nWARNING! PoolLayer - Receptive field height is larger than input " +
                          "height. Adjusting reveptive field height to fit input height.")
            r_field_height = self._inp_shape[1]
        if self._inp_shape[2] < r_field_width:
            warnings.warn("\nWARNING! PoolLayer - Receptive field width is larger than input " +
                          "width. Adjusting reveptive field width to fit input width.")
            r_field_width = self._inp_shape[2]
        self._receptive_field = tuple((r_field_height, r_field_width))

        # Set stride rows and columns
        if stride is None:
            stride_r = r_field_height
            stride_c = r_field_width
        elif type(stride) is int:
            stride_r = stride_c = stride
        elif type(stride) is tuple:
            stride_r = stride[0]
            stride_c = stride[1]

        # Reduce stride rows and cols to fit input shape, if larger
        if self._inp_shape[1] < stride_r or r_field_height < stride_r:
            warnings.warn("\nWARNING! PoolLayer - Stride rows is larger than input " +
                          "height and/or receptive field height. Adjusting stride rows accordingly.")
            stride_r = int(np.min([self._inp_shape[1], r_field_height]))
        if self._inp_shape[2] < stride_c or r_field_width < stride_c:
            warnings.warn("\nWARNING! PoolLayer - Stride cols is larger than input " +
                          "width. Adjusting stride cols to fit input width.")
            stride_c = int(np.min([self._inp_shape[2], r_field_width]))
        self._stride = tuple((stride_r, stride_c))

        self._pooling_mask = None

        # Check if the hyperparameter are valid for the given input volume, and set output volume shape
        self.set_output_volume_shape(padding)

        # Calculate unroll indices of input volume
        self.calculate_unroll_indices()


    # Getters
    # -------
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
    def padding(self):
        return self._padding

    @property
    def stride(self):
        return self._stride


    def set_output_volume_shape(self, padding):
        inp_d = self._inp_shape[0]
        inp_h = self._inp_shape[1]
        inp_w = self._inp_shape[2]
        ker_h = self._receptive_field[0]
        ker_w = self._receptive_field[1]

        pad_h = ((inp_h - ker_h) % self._stride[0])
        pad_w = ((inp_w - ker_w) % self._stride[1])

        if pad_h == 0 and pad_w == 0:
            pad_h_tuple = tuple((0,0))
            pad_w_tuple = tuple((0,0))
            self._padding = None
        else:
            if pad_h == 0:
                pad_h_tuple = tuple((0,0))
            else:
                pad_h_tuple = tuple((int(np.floor(pad_h/2)), int(np.ceil(pad_h/2))))

            if pad_w == 0:
                pad_w_tuple = tuple((0,0))
            else:
                pad_w_tuple = tuple((int(np.floor(pad_w/2)), int(np.ceil(pad_w/2))))

            self._padding = tuple((pad_h_tuple, pad_w_tuple))

        # Calculate output volume's height and width:
        # (W - F + P)/S + 1 --> W: Input volume size | F: Receptive field size | P: Pad | S: Stride
        o_h = ((inp_h - ker_h + np.sum(pad_h_tuple)) / self._stride[0]) + 1
        o_w = ((inp_w - ker_w + np.sum(pad_w_tuple)) / self._stride[1]) + 1

        self._out_shape = tuple((None, inp_d, int(o_h), int(o_w)))
        self._out_size = np.prod(self._out_shape[1:])


    def calculate_unroll_indices(self):
        inp_d = self._inp_shape[0]
        inp_h = self._inp_shape[1]
        inp_w = self._inp_shape[2]
        ker_h = self._receptive_field[0]
        ker_w = self._receptive_field[1]

        window_row_inds = inp_h - (ker_h-1) + (0 if self._padding is None else
                                               np.sum(self._padding[0]))
        window_col_inds = inp_w - (ker_w-1) + (0 if self._padding is None else
                                               np.sum(self._padding[1]))

        out_h = self._out_shape[2]
        out_w = self._out_shape[3]

        r0 = np.repeat(np.arange(ker_h), ker_w)
        r1 = np.repeat(np.arange(window_row_inds, step=self._stride[0]), out_w)

        c0 = np.tile(np.arange(ker_w), ker_h)
        c1 = np.tile(np.arange(window_col_inds, step=self._stride[1]), out_h)

        r = r1.reshape(-1,1) + r0.reshape(1,-1)
        c = c1.reshape(-1,1) + c0.reshape(1,-1)

        self._row_inds = r
        self._col_inds = c


    def input_gradients_max(self, inp_grad, summed=True): # Pool-grad Algo-3
        # dy/dx: Gradient of the layer activation 'y' w.r.t the inputs 'X'
        batch_size = inp_grad.shape[0]
        inp_dep = self._inp_shape[0]

        r = np.arange(self._pooling_mask.size) % (np.prod(inp_grad.shape[2:]))
        row = self._row_inds[r, self._pooling_mask].reshape(batch_size, *self._out_shape[1:])
        col = self._col_inds[r, self._pooling_mask].reshape(batch_size, *self._out_shape[1:])

        if self._padding is None:
            out_grad_shape = tuple((batch_size, *self._inp_shape))
        else:
            out_grad_rows_cols = np.array(self._inp_shape[1:]) + np.sum(np.array(self._padding),
                                                                        axis=-1)
            out_grad_shape = tuple((batch_size, inp_dep, *out_grad_rows_cols))
        out_grads = np.zeros(out_grad_shape)

        batch = np.arange(batch_size).reshape(-1, 1, 1, 1)
        dep = np.arange(inp_dep).reshape(1, -1, 1, 1)

        np.add.at(out_grads, [batch, dep, row, col], inp_grad)

        if self._padding is not None:
            pad_r = self._padding[0]
            pad_c = self._padding[1]

            if np.sum(pad_r) > 0:
                out_grads = out_grads[:,:,pad_r[0]:-pad_r[1],:]

            if np.sum(pad_c) > 0:
                out_grads = out_grads[:,:,:,pad_c[0]:-pad_c[1]]

        return out_grads


    def input_gradients_avg(self, inp_grad, summed=True): # Pool-grad Algo-3
        # dy/dx: Gradient of the layer activation 'y' w.r.t the inputs 'X'
        batch_size = inp_grad.shape[0]
        inp_dep = self._inp_shape[0]
        field_size = np.prod(self._receptive_field)

        # row = np.expand_dims(np.expand_dims(self._row_inds, axis=0), axis=0)
        # col = np.expand_dims(np.expand_dims(self._col_inds, axis=0), axis=0)
        row = self._row_inds[np.newaxis,np.newaxis,:,:]
        col = self._col_inds[np.newaxis,np.newaxis,:,:]
        batch = np.arange(batch_size).reshape(-1, 1, 1, 1)
        dep = np.arange(inp_dep).reshape(1, -1, 1, 1)

        if self._padding is None:
            out_grad_shape = tuple((batch_size, *self._inp_shape))
        else:
            out_grad_rows_cols = np.array(self._inp_shape[1:]) + np.sum(np.array(self._padding),
                                                                        axis=-1)
            out_grad_shape = tuple((batch_size, inp_dep, *out_grad_rows_cols))
        out_grads = np.zeros(out_grad_shape)

        grads = inp_grad.reshape(batch_size, inp_dep, -1, 1) * \
                np.array([1.0/field_size] * int(field_size)).reshape(1, 1, 1, -1)

        np.add.at(out_grads, [batch, dep, row, col], grads)

        if self._padding is not None:
            pad_r = self._padding[0]
            pad_c = self._padding[1]

            if np.sum(pad_r) > 0:
                out_grads = out_grads[:,:,pad_r[0]:-pad_r[1],:]

            if np.sum(pad_c) > 0:
                out_grads = out_grads[:,:,:,pad_c[0]:-pad_c[1]]

        return out_grads

    def forward_max(self, padded_inputs):
        ker_h = self._receptive_field[0]
        ker_w = self._receptive_field[1]

        # Unroll input volume to shape: (batch, out_rows*out_cols, filter_size[hxw])
        unrolled_inputs = padded_inputs[:, :, self._row_inds, self._col_inds]

        # Pooling mask of each receptive field, per slice over the entire batch
        self._pooling_mask = np.argmax(unrolled_inputs, axis=-1).reshape(-1)

        # Max pooling output reshaped to output size
        unroller_inp_reshaped = unrolled_inputs.reshape(-1, ker_h*ker_w)
        pooling_out = unroller_inp_reshaped[np.arange(unroller_inp_reshaped.shape[0]),
                                            self._pooling_mask]

        return pooling_out.reshape(-1, *self.shape[1:])


    def forward_avg(self, padded_inputs):
        ker_h = self._receptive_field[0]
        ker_w = self._receptive_field[1]

        # Unroll input volume to shape: (batch, out_rows*out_cols, filter_size[hxw])
        unrolled_inputs = padded_inputs[:, :, self._row_inds, self._col_inds]

        # Pooling mask of each receptive field, per slice over the entire batch
        avg_pool_out = np.mean(unrolled_inputs, axis=-1).reshape(-1, *self.shape[1:])

        return avg_pool_out


    def forward(self, inputs, inference=None):
        # Zero-pad input volume based on the setting
        if self._padding is not None:
            pad = self._padding
            padded_inputs = np.pad(inputs, ((0,0),(0,0),*self._padding), 'constant',
                                   constant_values=-np.inf if self._pool == 'MAX' else 0)
        else:
            padded_inputs = inputs

        if self._pool == 'MAX':
            pool_out = self.forward_max(padded_inputs)
        elif self._pool == 'AVG':
            pool_out = self.forward_avg(padded_inputs)

        return pool_out


    def backward(self, inp_grad, reg_lambda=0, inputs=None):
        if len(inp_grad.shape) > 2: # The proceeding layer is a Convolution/Pooling layer
            pass
        else: # The proceeding layer is a FC layer
            # Reshape incoming gradients accordingly
            inp_grad = inp_grad.reshape(-1, *self._out_shape[1:])

        if self._pool == 'MAX':
            out_grad = self.input_gradients_max(inp_grad)
            self._pooling_mask = None
        elif self._pool == 'AVG':
            out_grad = self.input_gradients_avg(inp_grad)

        return out_grad

    def update_weights(self, alpha):
        pass
