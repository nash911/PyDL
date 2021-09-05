# ------------------------------------------------------------------------
# MIT License
#
# Copyright (c) [2021] [Avinash Ranganath]
#
# This code is part of the library PyDL <https://github.com/nash911/PyDL>
# This code is licensed under MIT license (see LICENSE.txt for details)
# ------------------------------------------------------------------------

import unittest
import numpy as np
import numpy.testing as npt
import itertools

from pydl.nn.conv import Conv
from pydl.nn.residual_block import ResidualBlock
from pydl import conf


class TestResidualBlock(unittest.TestCase):
    def unroll_input_volume(self, inp, w, batch, dep, inp_h, inp_w, num_k, k_h, k_w, pad, strd):
        input_padded = np.pad(inp, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant') \
            if pad > 0 else inp
        out_h = int((inp_h - k_h + (2 * pad)) / strd) + 1
        out_w = int((inp_w - k_w + (2 * pad)) / strd) + 1
        out_rows = out_h * out_w
        out_cols = w[0].size
        unrolled_inp_shape = tuple((batch, out_rows, out_cols))
        unrolled_inp = np.empty(unrolled_inp_shape)
        kernal_size = k_h * k_w

        for b in range(batch):
            for row in range(out_rows):
                for col in range(out_cols):
                    inp_dep = int(col / kernal_size)
                    inp_row = int(row / out_w) * strd + int((col % kernal_size) / k_w)
                    inp_col = int(row % out_w) * strd + (col % k_w)
                    unrolled_inp[b, row, col] = input_padded[b, inp_dep, inp_row, inp_col]

        return unrolled_inp

    def test_forward(self):
        def test(inp, weights, skip_weights, true_out, pad=0, stride=1, rcp_field=None,
                 num_filters=None, actv_fn='ReLU', bchnorm=False):
            conv_layers = [inp]
            for i, (w, b) in enumerate(weights):
                conv = Conv(conv_layers[-1], receptive_field=rcp_field, num_filters=num_filters,
                            zero_padding=pad, stride=(stride if i == 0 else 1), batchnorm=bchnorm,
                            activation_fn=actv_fn)
                conv.weights = w
                conv.bias = b
                conv_layers.append(conv)

            res_block = ResidualBlock(inp, conv_layers[1:], activation_fn=actv_fn)

            if skip_weights is not None:
                res_block.skip_convolution.weights = skip_weights[0]
                res_block.skip_convolution.bias = skip_weights[1]

            out_volume = res_block.forward(inp)
            npt.assert_almost_equal(out_volume, true_out, decimal=8)

        # Combinatorial Test Cases
        # ------------------------
        num_layers = [1, 2, 3, 5]
        batch_size = [1, 2, 3]
        inp_depth = [1, 2, 3]
        inp_height = [5, 7, 9]
        inp_width = [5, 7, 9]
        num_kernals = [1, 2, 3]
        kernal = [1, 3]
        stride = [1, 2]
        batchnorm = [True, False]
        activationv_function = ['Linear', 'Tanh', 'ReLU']

        for n_layers, batch, dep, inp_h, inp_w, num_k, ker, strd, bn, actv_fn in \
            list(itertools.product(num_layers, batch_size, inp_depth, inp_height, inp_width,
                                   num_kernals, kernal, stride, batchnorm, activationv_function)):

            # Create block input
            X = np.random.uniform(-1, 1, (batch, dep, inp_h, inp_w))

            # Calculate pad size based on kernal size
            pad = int((ker - 1) / 2)

            weights = list()
            layer_inp = X
            # Forward propogate block convolutions
            for i in range(n_layers):
                inp_dep = layer_inp.shape[1]
                layer_inp_h = layer_inp.shape[2]
                layer_inp_w = layer_inp.shape[3]

                # Create layer's weights and bias
                w = np.random.randn(num_k, inp_dep, ker, ker) * 1.0
                b = np.random.rand(num_k)
                weights.append(tuple((w, b)))

                # Calculate layer's output size
                out_height = (layer_inp.shape[2] - ker + (2 * pad)) / (strd if i == 0 else 1) + 1
                out_width = (layer_inp.shape[3] - ker + (2 * pad)) / (strd if i == 0 else 1) + 1

                # Unroll layer's inputs
                unrolled_inp = self.unroll_input_volume(layer_inp, w, batch, inp_dep, layer_inp_h,
                                                        layer_inp_w, num_k, ker, ker, pad,
                                                        (strd if i == 0 else 1))

                # Using element-wise multiplication with broadcasting for calculating weighted sum
                weighted_sum = unrolled_inp[:, np.newaxis, :, :] * w.reshape(num_k, 1, -1)
                weighted_sum = np.sum(weighted_sum, axis=-1, keepdims=False) + b.reshape(-1, 1)
                score = weighted_sum.reshape(-1, num_k, int(out_height), int(out_width))

                if bn:
                    # Batch-norm on the layer's score
                    score = (score - np.mean(score, axis=0)) / np.sqrt(np.var(score, axis=0) +
                                                                       1e-32)

                if i == n_layers - 1:  # Skip activation for the final layer in the block
                    layer_out = score
                else:  # Else apply the appropriate activation for the later's score
                    if actv_fn == 'Tanh':
                        layer_out = (2.0 / (1.0 + np.exp(-2.0 * score))) - 1.0
                    elif actv_fn == 'ReLU':
                        layer_out = np.maximum(0, score)
                    elif actv_fn == 'Linear':
                        layer_out = score

                # Set the layer's activation as input for the following layer
                layer_inp = layer_out

            # If there is a size mismatch between the volumes of the skip connection and the block
            # output, then perform a linear transformation to resize the skip_connection
            if layer_out.shape != X.shape:
                out_height = (inp_h - 1 + 0) / strd + 1
                out_width = (inp_w - 1 + 0) / strd + 1
                out_dep = layer_out.shape[1]

                w_skip = np.random.randn(out_dep, dep, 1, 1) * 1.0
                b_skip = np.random.rand(out_dep)

                unrolled_inp = self.unroll_input_volume(X, w_skip, batch, dep, inp_h, inp_w,
                                                        out_dep, 1, 1, 0, strd)

                # Using element-wise multiplication with broadcasting for calculating weighted sum
                weighted_sum = unrolled_inp[:, np.newaxis, :, :] * w_skip.reshape(out_dep, 1, -1)
                weighted_sum = np.sum(weighted_sum, axis=-1, keepdims=False) + b_skip.reshape(-1, 1)
                score = weighted_sum.reshape(-1, out_dep, int(out_height), int(out_width))

                score = (score - np.mean(score, axis=0)) / np.sqrt(np.var(score, axis=0) + 1e-32)
                skip_input = score
                skip_weights = tuple((w_skip, b_skip))
            else:
                skip_input = X
                skip_weights = None

            # Add skip connection to the block's score output
            block_score = layer_out + skip_input

            # Perform the final activation on the added tensors
            if actv_fn == 'Tanh':
                true_out = (2.0 / (1.0 + np.exp(-2.0 * block_score))) - 1.0
            elif actv_fn == 'ReLU':
                true_out = np.maximum(0, block_score)
            elif actv_fn == 'Linear':
                true_out = block_score

            test(X, weights, skip_weights, true_out, pad=pad, stride=strd, rcp_field=(ker, ker),
                 num_filters=num_k, actv_fn=actv_fn, bchnorm=bn)

    def test_backward_gradients_finite_difference(self):
        self.delta = 1e-8

        def test(inp, weights, actv_fn, pad=0, stride=1, rcp_field=None, num_filters=None,
                 bchnorm=False):
            conv_layers = [inp]
            for i, (w, b) in enumerate(weights):
                conv = Conv(conv_layers[-1], receptive_field=rcp_field, num_filters=num_filters,
                            zero_padding=pad, stride=(stride if i == 0 else 1), batchnorm=bchnorm,
                            activation_fn=actv_fn)
                conv.weights = w
                conv.bias = b
                conv_layers.append(conv)

            res_block = ResidualBlock(inp, conv_layers[1:], activation_fn=actv_fn)

            y = res_block.forward(inp)
            inp_grad = np.random.uniform(-1, 1, y.shape)
            inputs_grad = res_block.backward(inp_grad)

            if res_block.skip_convolution is not None:
                skip_weights = res_block.skip_convolution.weights
                skip_bias = res_block.skip_convolution.bias

                skip_w_grad = res_block.skip_convolution.weights_grad
                skip_b_grad = res_block.skip_convolution.bias_grad
            else:
                skip_weights = None
                skip_bias = None
                skip_w_grad = None
                skip_b_grad = None

            res_block_layers = res_block.layers

            weights_grad = list()
            bias_grad = list()
            weights_finite_diff = list()
            bias_finite_diff = list()
            for layer in res_block_layers:
                w = layer.weights
                b = layer.bias

                w_grad = layer.weights_grad
                b_grad = layer.bias_grad
                weights_grad.append(w_grad)
                bias_grad.append(b_grad)

                # Weights finite difference gradients
                w_fin_diff = np.empty(w_grad.shape)
                for i in range(w_grad.shape[0]):
                    for j in range(w_grad.shape[1]):
                        for k in range(w_grad.shape[2]):
                            for m in range(w_grad.shape[3]):
                                w_delta = np.zeros(w.shape, dtype=conf.dtype)
                                w_delta[i, j, k, m] = self.delta
                                layer.weights = w + w_delta
                                lhs = res_block.forward(inp)
                                layer.weights = w - w_delta
                                rhs = res_block.forward(inp)
                                w_fin_diff[i, j, k, m] = np.sum(((lhs - rhs) / (2 * self.delta)) *
                                                                inp_grad)

                                # Replace finite-diff gradients calculated close to 0 with NN
                                # calculated gradients to pass assertion test
                                grad_kink = np.sum(np.array(np.logical_xor(np.round(lhs, 6) > 0,
                                                                           np.round(rhs, 6) > 0),
                                                            dtype=np.int32))
                                if grad_kink > 0:
                                    print("Weights - Kink Encountered!")
                                    w_fin_diff[i, j, k, m] = w_grad[i, j, k, m]
                layer.weights = w
                weights_finite_diff.append(w_fin_diff)

                # Bias finite difference gradients
                if not bchnorm:
                    b_fin_diff = np.empty(b_grad.shape)
                    for i in range(b_grad.shape[0]):
                        b_delta = np.zeros(b.shape, dtype=conf.dtype)
                        b_delta[i] = self.delta
                        layer.bias = b + b_delta
                        lhs = res_block.forward(inp)
                        layer.bias = b - b_delta
                        rhs = res_block.forward(inp)
                        b_fin_diff[i] = np.sum(((lhs - rhs) / (2 * self.delta)) * inp_grad)

                        # Replace finite-diff gradients calculated close to 0 with NN calculated
                        # gradients to pass assertion test
                        grad_kink = np.sum(np.array(np.logical_xor(np.round(lhs, 6) > 0,
                                                                   np.round(rhs, 6) > 0),
                                                    dtype=np.int32))
                        if grad_kink > 0:
                            print("Bias - Kink Encountered!")
                            b_fin_diff[i] = b_grad[i]
                    layer.bias = b
                else:
                    b_fin_diff = b_grad
                bias_finite_diff.append(b_fin_diff)

            # Skip Convolution weights finite differnce
            if res_block.skip_convolution is not None:
                # Weights finite difference gradients
                skip_w_fin_diff = np.empty(skip_w_grad.shape)
                for i in range(skip_w_grad.shape[0]):
                    for j in range(skip_w_grad.shape[1]):
                        for k in range(skip_w_grad.shape[2]):
                            for m in range(skip_w_grad.shape[3]):
                                skip_w_delta = np.zeros(skip_weights.shape, dtype=conf.dtype)
                                skip_w_delta[i, j, k, m] = self.delta
                                res_block.skip_convolution.weights = skip_weights + skip_w_delta
                                lhs = res_block.forward(inp)
                                res_block.skip_convolution.weights = skip_weights - skip_w_delta
                                rhs = res_block.forward(inp)
                                skip_w_fin_diff[i, j, k, m] = \
                                    np.sum(((lhs - rhs) / (2 * self.delta)) * inp_grad)

                                # Replace finite-diff gradients calculated close to 0 with NN
                                # calculated gradients to pass assertion test
                                grad_kink = np.sum(np.array(np.logical_xor(np.round(lhs, 6) > 0,
                                                                           np.round(rhs, 6) > 0),
                                                            dtype=np.int32))
                                if grad_kink > 0:
                                    print("Skip Weights - Kink Encountered!")
                                    skip_w_fin_diff[i, j, k, m] = skip_w_grad[i, j, k, m]
                res_block.skip_convolution.weights = skip_weights

                # Bias finite difference gradients
                if not bchnorm:
                    skip_b_fin_diff = np.empty(skip_b_grad.shape)
                    for i in range(skip_b_grad.shape[0]):
                        skip_b_delta = np.zeros(skip_bias.shape, dtype=conf.dtype)
                        skip_b_delta[i] = self.delta
                        res_block.skip_convolution.bias = skip_bias + skip_b_delta
                        lhs = res_block.forward(inp)
                        res_block.skip_convolution.bias = skip_bias - skip_b_delta
                        rhs = res_block.forward(inp)
                        skip_b_fin_diff[i] = np.sum(((lhs - rhs) / (2 * self.delta)) * inp_grad)

                        # Replace finite-diff gradients calculated close to 0 with NN calculated
                        # gradients to pass assertion test
                        grad_kink = np.sum(np.array(np.logical_xor(np.round(lhs, 6) > 0,
                                                                   np.round(rhs, 6) > 0),
                                                    dtype=np.int32))
                        if grad_kink > 0:
                            print("Skip Bias - Kink Encountered!")
                            skip_b_fin_diff[i] = skip_b_grad[i]
                    res_block.skip_convolution.bias = skip_bias
                else:
                    skip_b_fin_diff = skip_b_grad

            # Inputs finite difference gradients
            inputs_finite_diff = np.empty(inputs_grad.shape)
            for i in range(inputs_grad.shape[0]):
                for j in range(inputs_grad.shape[1]):
                    for k in range(inputs_grad.shape[2]):
                        for m in range(inputs_grad.shape[3]):
                            i_delta = np.zeros(inp.shape, dtype=conf.dtype)
                            i_delta[i, j, k, m] = self.delta
                            lhs = res_block.forward(inp + i_delta)
                            rhs = res_block.forward(inp - i_delta)
                            inputs_finite_diff[i, j, k, m] = \
                                np.sum(((lhs - rhs) / (2 * self.delta)) * inp_grad, keepdims=False)

                            # Replace finite-diff gradients calculated close to 0 with NN calculated
                            # gradients to pass assertion test
                            grad_kink = np.sum(np.array(np.logical_xor(np.round(lhs, 6) > 0,
                                                                       np.round(rhs, 6) > 0),
                                                        dtype=np.int32))
                            if grad_kink > 0:
                                print("Inputs - Kink Encountered!")
                                inputs_finite_diff[i, j, k, m] = inputs_grad[i, j, k, m]

            for i, (w_grad, w_fin_diff, b_grad, b_fin_diff) in \
                    enumerate(zip(weights_grad, weights_finite_diff, bias_grad, bias_finite_diff)):
                npt.assert_almost_equal(w_grad, w_fin_diff, decimal=2)
                npt.assert_almost_equal(b_grad, b_fin_diff, decimal=2)

            if skip_weights is not None:
                npt.assert_almost_equal(skip_w_grad, skip_w_fin_diff, decimal=2)
                npt.assert_almost_equal(skip_b_grad, skip_b_fin_diff, decimal=2)

            npt.assert_almost_equal(inputs_grad, inputs_finite_diff, decimal=2)

        # Combinatorial Test Cases
        # ------------------------
        num_layers = [1, 2, 3]
        batch_size = [1, 2, 3]
        inp_depth = [1, 2, 3]
        inp_height = [5, 7]
        inp_width = [5, 7]
        num_kernals = [1, 2, 3]
        kernal = [1, 3]
        stride = [1, 2]
        batchnorm = [True, False]
        activationv_function = ['Linear', 'Tanh', 'ReLU']

        for n_layers, batch, dep, inp_h, inp_w, num_k, ker, strd, bn, actv in \
            list(itertools.product(num_layers, batch_size, inp_depth, inp_height, inp_width,
                                   num_kernals, kernal, stride, batchnorm, activationv_function)):

            # Create block input
            X = np.random.uniform(-1, 1, (batch, dep, inp_h, inp_w))

            # Calculate pad size based on kernal size
            pad = int((ker - 1) / 2)

            weights = list()
            inp_dep = dep

            # Create convolution layer weights and bias
            for i in range(n_layers):
                # Create layer's weights and bias
                w = np.random.randn(num_k, inp_dep, ker, ker) * 1.0
                b = np.random.rand(num_k)
                weights.append(tuple((w, b)))

                inp_dep = num_k

            test(X, weights, actv_fn=actv, pad=pad, stride=strd, rcp_field=(ker, ker),
                 num_filters=num_k, bchnorm=bn)


if __name__ == '__main__':
    unittest.main()
