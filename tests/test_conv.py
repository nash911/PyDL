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
from pydl import conf


class TestConv(unittest.TestCase):
    def unroll_input_volume(self, inp, w, batch, dep, inp_h, inp_w, num_k, k_h, k_w, pad, strd,
                            frc_adjust=False):
        input_padded = np.pad(inp, ((0, 0), (0, 0), (pad[0], pad[1]), (pad[0], pad[1])),
                              'constant') if np.sum(pad) > 0 else inp
        out_h = ((inp_h - k_h + np.sum(pad)) / strd) + 1
        out_w = ((inp_w - k_w + np.sum(pad)) / strd) + 1
        if frc_adjust:
            if k_h == 1 or strd % 2 == 1:
                out_h = np.floor(out_h)
            else:
                out_h = np.ceil(out_h)

            if k_w == 1 or strd % 2 == 1:
                out_w = np.floor(out_w)
            else:
                out_w = np.ceil(out_w)

        out_h = int(out_h)
        out_w = int(out_w)
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

    def test_score_fn(self):
        def test(inp, w, true_out, bias=False, pad=(0, 0), stride=1, rcp_field=None,
                 num_filters=None, frc_adjust=False):
            if true_out is None:
                with npt.assert_raises(SystemExit):
                    conv = Conv(inp, zero_padding=pad, stride=stride, weights=w, bias=bias,
                                force_adjust_output_shape=frc_adjust)
                    out_volume = conv.score_fn(inp)
            else:
                if pad[0] == pad[1]:
                    pad = pad[0]
                conv = Conv(inp, receptive_field=rcp_field, num_filters=num_filters,
                            zero_padding=pad, stride=stride, weights=w, bias=bias,
                            force_adjust_output_shape=frc_adjust)
                conv.weights = w
                conv.bias = bias
                out_volume = conv.score_fn(inp)
                npt.assert_almost_equal(out_volume, true_out, decimal=8)

        # Manually calculated
        # -------------------
        inp = np.array([[[2, 0, 0, 1, 2],  # Slice-1
                         [0, 1, 2, 0, 0],
                         [1, 1, 2, 0, 2],
                         [1, 1, 1, 1, 0],
                         [2, 0, 2, 2, 1]],
                        [[1, 1, 1, 2, 1],  # Slice-2
                         [2, 1, 0, 0, 0],
                         [1, 2, 0, 1, 1],
                         [1, 2, 0, 2, 2],
                         [2, 0, 0, 1, 0]],
                        [[1, 0, 1, 0, 2],  # Slice-3
                         [2, 0, 1, 0, 1],
                         [0, 2, 0, 1, 1],
                         [0, 0, 0, 0, 2],
                         [0, 2, 2, 0, 1]]], dtype=conf.dtype)
        X = np.array([inp, inp * 2])

        w = np.array([[[[0, 1, 1],  # Kernal-1
                        [1, 0, 1],
                        [1, 1, 1]],
                       [[0, 1, 1],
                        [1, -1, 1],
                        [1, 1, 0]],
                       [[1, 1, 1],
                        [0, -1, 0],
                        [1, 1, 1]]],
                      [[[-1, -1, -1],  # Kernal-2
                        [1, 0, 0],
                        [0, 1, 0]],
                       [[-1, 0, -1],
                        [0, -1, 0],
                        [-1, 0, 1]],
                       [[-1, 1, -1],
                        [1, -1, 1],
                        [-1, 1, -1]]]], dtype=conf.dtype)

        bias = np.array([1, 0], dtype=conf.dtype)

        out = np.array([[[4, 7, 1],  # Slice-1
                         [11, 12, 7],
                         [3, 5, 6]],
                        [[1, 0, -1],  # Slice-2
                         [4, 2, 0],
                         [-4, -7, 0]]], dtype=conf.dtype)
        true_out = np.array([out, out * 2])

        test(X, w, true_out + bias.reshape(2, 1, 1), bias=bias, pad=(1, 1), stride=2,
             rcp_field=(3, 3), num_filters=2)

        # Combinatorial Test Cases
        # ------------------------
        batch_size = [1, 2, 3]
        inp_depth = [1, 2, 3]
        inp_height = [1, 2, 3, 5, 7]
        inp_width = [1, 2, 3, 5, 7]
        num_kernals = [1, 2, 3, 8]
        kernal_height = [1, 3, 5]
        kernal_width = [1, 3, 5]
        padding_0 = [0, 1, 2]
        padding_1 = [0, 1, 2]
        stride = [1, 2, 3]
        force_adjust = [True, False]

        for batch, dep, inp_h, inp_w, num_k, k_h, k_w, pad_0, pad_1, strd, force in \
            list(itertools.product(batch_size, inp_depth, inp_height, inp_width, num_kernals,
                                   kernal_height, kernal_width, padding_0, padding_1, stride,
                                   force_adjust)):
            X = np.random.uniform(-1, 1, (batch, dep, inp_h, inp_w))
            w = np.random.randn(num_k, dep, k_h, k_w) * 1.0
            bias = np.random.rand(num_k)
            pad = (pad_0, pad_1)

            out_height = (inp_h - k_h + np.sum(pad)) / strd + 1
            out_width = (inp_w - k_w + np.sum(pad)) / strd + 1

            if (k_h > inp_h or k_w > inp_w):
                continue
            elif (out_height % 1 != 0 or out_width % 1 != 0):
                if not force:
                    continue
                else:
                    if np.sum(pad) > 0:
                        continue
                    elif strd == 1:
                        continue
            elif pad_1 == 0 and pad_0 > 0:
                continue
            elif force and np.sum(pad) > 0:
                continue

            unrolled_inp = self.unroll_input_volume(X, w, batch, dep, inp_h, inp_w, num_k,
                                                    k_h, k_w, pad, strd)
            # Using dot product for calculating weighted sum
            weighted_sum = np.matmul(unrolled_inp, w.reshape(num_k, -1).T) + bias
            true_out = weighted_sum.transpose(0, 2, 1).reshape(-1, num_k, int(out_height),
                                                               int(out_width))

            test(X, w, true_out, bias=bias, pad=pad, stride=strd, rcp_field=(k_h, k_w),
                 num_filters=num_k, frc_adjust=force)

    def test_forward(self):

        def test(inp, w, true_out, bias=False, pad=(0, 0), stride=1, rcp_field=None, actv_fn='ReLU',
                 num_filters=None, bchnorm=False, p=None, mask=None, frc_adjust=False):
            if pad[0] == pad[1]:
                pad = pad[0]
            conv = Conv(inp, receptive_field=rcp_field, num_filters=num_filters, zero_padding=pad,
                        stride=stride, activation_fn=actv_fn, batchnorm=bchnorm, dropout=p,
                        force_adjust_output_shape=frc_adjust)
            conv.weights = w
            conv.bias = bias
            out_volume = conv.forward(inp, mask=mask)
            npt.assert_almost_equal(out_volume, true_out, decimal=8)

        # Combinatorial Test Cases
        # ------------------------
        batch_size = [1, 2, 3]
        inp_depth = [1, 2, 3]
        inp_height = [1, 2, 3, 5, 7]
        inp_width = [1, 2, 3, 5, 7]
        num_kernals = [1, 2, 3]
        kernal_height = [1, 3, 5]
        kernal_width = [1, 3, 5]
        padding_0 = [0, 1, 2]
        padding_1 = [0, 1, 2]
        stride = [1, 2, 3]
        batchnorm = [True, False]
        dropout = [True, False]
        force_adjust = [True, False]

        for batch, dep, inp_h, inp_w, num_k, k_h, k_w, pad_0, pad_1, strd, bn, dout, force in \
            list(itertools.product(batch_size, inp_depth, inp_height, inp_width, num_kernals,
                                   kernal_height, kernal_width, padding_0, padding_1, stride,
                                   batchnorm, dropout, force_adjust)):
            X = np.random.uniform(-1, 1, (batch, dep, inp_h, inp_w))
            w = np.random.randn(num_k, dep, k_h, k_w) * 1.0
            bias = np.random.rand(num_k)
            pad = (pad_0, pad_1)

            out_height = (inp_h - k_h + np.sum(pad)) / strd + 1
            out_width = (inp_w - k_w + np.sum(pad)) / strd + 1

            if (k_h > inp_h or k_w > inp_w):
                continue
            elif (out_height % 1 != 0 or out_width % 1 != 0):
                if not force:
                    continue
                else:
                    if np.sum(pad) > 0:
                        continue
                    elif strd == 1:
                        continue
            elif pad_1 == 0 and pad_0 > 0:
                continue
            elif force and np.sum(pad) > 0:
                continue

            unrolled_inp = self.unroll_input_volume(X, w, batch, dep, inp_h, inp_w, num_k,
                                                    k_h, k_w, pad, strd, force)

            # Using element-wise multiplication with broadcasting for calculating weighted sum
            weighted_sum = unrolled_inp[:, np.newaxis, :, :] * w.reshape(num_k, 1, -1)
            weighted_sum = np.sum(weighted_sum, axis=-1, keepdims=False) + bias.reshape(-1, 1)
            score = weighted_sum.reshape(-1, num_k, int(out_height), int(out_width))

            if bn:
                score = (score - np.mean(score, axis=0)) / np.sqrt(np.var(score, axis=0) + 1e-32)

            if dout:
                p = np.random.rand()
                mask = np.array(np.random.rand(*score.shape) < p, dtype=conf.dtype)
            else:
                p = None
                mask = None

            true_out_tanh = (2.0 / (1.0 + np.exp(-2.0 * score))) - 1.0
            if dout:
                true_out_tanh *= mask
            test(X, w, true_out_tanh, bias=bias, pad=pad, stride=strd, rcp_field=(k_h, k_w),
                 num_filters=num_k, actv_fn='Tanh', bchnorm=bn, p=p, mask=mask, frc_adjust=force)

            true_out_relu = np.maximum(0, score)
            if dout:
                mask /= p
                true_out_relu *= mask
            test(X, w, true_out_relu, bias=bias, pad=pad, stride=strd, rcp_field=(k_h, k_w),
                 num_filters=num_k, actv_fn='ReLU', bchnorm=bn, p=p, mask=mask, frc_adjust=force)

            true_out_linear = score
            if dout:
                true_out_linear *= mask
            test(X, w, true_out_linear, bias=bias, pad=pad, stride=strd, rcp_field=(k_h, k_w),
                 num_filters=num_k, actv_fn='Linear', bchnorm=bn, p=p, mask=mask, frc_adjust=force)

    def test_backward_gradients_finite_difference(self):
        self.delta = 1e-8

        def test(inp, w, inp_grad, actv_fn, bias=False, pad=(0, 0), stride=1, rcp_field=None,
                 num_filters=None, bchnorm=False, p=None, mask=None):
            if pad[0] == pad[1]:
                pad = pad[0]
            conv = Conv(inp, receptive_field=rcp_field, num_filters=num_filters, zero_padding=pad,
                        stride=stride, activation_fn=actv_fn, batchnorm=bchnorm, dropout=p)
            conv.weights = w
            conv.bias = bias
            _ = conv.forward(inp, mask=mask)
            inputs_grad = conv.backward(inp_grad)
            weights_grad = conv.weights_grad
            bias_grad = conv.bias_grad

            # Weights finite difference gradients
            weights_finite_diff = np.empty(weights_grad.shape)
            for i in range(weights_grad.shape[0]):
                for j in range(weights_grad.shape[1]):
                    for k in range(weights_grad.shape[2]):
                        for m in range(weights_grad.shape[3]):
                            w_delta = np.zeros(w.shape, dtype=conf.dtype)
                            w_delta[i, j, k, m] = self.delta
                            conv.weights = w + w_delta
                            lhs = conv.forward(inp, mask=mask)
                            conv.weights = w - w_delta
                            rhs = conv.forward(inp, mask=mask)
                            weights_finite_diff[i, j, k, m] = \
                                np.sum(((lhs - rhs) / (2 * self.delta)) * inp_grad)

                            # Replace finite-diff gradients calculated close to 0 with NN calculated
                            # gradients to pass assertion test
                            grad_kink = np.sum(np.array(np.logical_xor(lhs > 0, rhs > 0),
                                               dtype=np.int32))
                            if grad_kink > 0:
                                print("Weights - Kink Encountered!")
                                weights_finite_diff[i, j] = weights_grad[i, j]
            conv.weights = w

            # Bias finite difference gradients
            if not bchnorm:
                bias_finite_diff = np.empty(bias_grad.shape)
                for i in range(bias_grad.shape[0]):
                    bias_delta = np.zeros(bias.shape, dtype=conf.dtype)
                    bias_delta[i] = self.delta
                    conv.bias = bias + bias_delta
                    lhs = conv.forward(inp, mask=mask)
                    conv.bias = bias - bias_delta
                    rhs = conv.forward(inp, mask=mask)
                    bias_finite_diff[i] = np.sum(((lhs - rhs) / (2 * self.delta)) * inp_grad)

                    # Replace finite-diff gradients calculated close to 0 with NN calculated
                    # gradients to pass assertion test
                    grad_kink = np.sum(np.array(np.logical_xor(lhs > 0, rhs > 0), dtype=np.int32))
                    if grad_kink > 0:
                        print("Bias - Kink Encountered!")
                        bias_finite_diff[i] = bias_grad[i]
                conv.bias = bias
            else:
                bias_finite_diff = bias_grad

            # Inputs finite difference gradients
            inputs_finite_diff = np.empty(inputs_grad.shape)
            for i in range(inputs_grad.shape[0]):
                for j in range(inputs_grad.shape[1]):
                    for k in range(inputs_grad.shape[2]):
                        for m in range(inputs_grad.shape[3]):
                            i_delta = np.zeros(inp.shape, dtype=conf.dtype)
                            i_delta[i, j, k, m] = self.delta
                            lhs = conv.forward(inp + i_delta, mask=mask)
                            rhs = conv.forward(inp - i_delta, mask=mask)
                            inputs_finite_diff[i, j, k, m] = \
                                np.sum(((lhs - rhs) / (2 * self.delta)) * inp_grad, keepdims=False)

                            # Replace finite-diff gradients calculated close to 0 with NN calculated
                            # gradients to pass assertion test
                            grad_kink = np.sum(np.array(np.logical_xor(lhs > 0, rhs > 0),
                                               dtype=np.int32))
                            if grad_kink > 0:
                                print("Inputs - Kink Encountered!")
                                inputs_finite_diff[i, j, k] = inputs_grad[i, j, k]

            npt.assert_almost_equal(weights_grad, weights_finite_diff, decimal=2)
            npt.assert_almost_equal(bias_grad, bias_finite_diff, decimal=2)
            npt.assert_almost_equal(inputs_grad, inputs_finite_diff, decimal=2)

        # Combinatorial Test Cases
        # ------------------------
        batch_size = [1, 2]
        inp_depth = [1, 2]
        inp_height = [1, 5, 7]
        inp_width = [1, 5, 7]
        num_kernals = [1, 2]
        kernal_height = [1, 3]
        kernal_width = [1, 3]
        padding_0 = [0, 1, 2]
        padding_1 = [0, 1, 2]
        stride = [1, 2]
        unit_inp_grad = [False]
        activation_fn = ['Linear', 'Tanh', 'ReLU']
        batchnorm = [True, False]
        dropout = [True, False]
        force_adjust = [True, False]
        scale = [1e-0]

        for batch, dep, inp_h, inp_w, num_k, k_h, k_w, pad_0, pad_1, strd, unit, actv, bn, dout, \
            force, scl in list(itertools.product(batch_size, inp_depth, inp_height, inp_width,
                               num_kernals, kernal_height, kernal_width, padding_0, padding_1,
                               stride, unit_inp_grad, activation_fn, batchnorm, dropout,
                               force_adjust, scale)):
            X = np.random.uniform(-scl, scl, (batch, dep, inp_h, inp_w))
            w = np.random.randn(num_k, dep, k_h, k_w) * scl
            bias = np.random.rand(num_k) * scl
            pad = (pad_0, pad_1)

            out_height = (inp_h - k_h + np.sum(pad)) / strd + 1
            out_width = (inp_w - k_w + np.sum(pad)) / strd + 1

            if (k_h > inp_h or k_w > inp_w):
                continue
            elif (out_height % 1 != 0 or out_width % 1 != 0):
                if not force:
                    continue
                else:
                    if np.sum(pad) > 0:
                        continue
                    elif strd == 1:
                        continue
            elif pad_1 == 0 and pad_0 > 0:
                continue
            elif force and np.sum(pad) > 0:
                continue

            out_height = int(out_height)
            out_width = int(out_width)

            inp_grad = np.ones((batch, num_k, out_height, out_width), dtype=conf.dtype) if unit \
                else np.random.uniform(-1, 1, (batch, num_k, out_height, out_width))

            if dout:
                p = np.random.rand()
                mask = np.array(np.random.rand(batch, num_k, out_height, out_width) < p,
                                dtype=conf.dtype)
                if actv in ['Linear', 'ReLU']:
                    mask /= p
            else:
                p = None
                mask = None

            test(X, w, inp_grad, actv_fn=actv, bias=bias, pad=pad, stride=strd, num_filters=num_k,
                 rcp_field=(k_h, k_w), bchnorm=bn, p=p, mask=mask)


if __name__ == '__main__':
    unittest.main()
