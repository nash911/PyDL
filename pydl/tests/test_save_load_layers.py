# ------------------------------------------------------------------------
# MIT License
#
# Copyright (c) [2022] [Avinash Ranganath]
#
# This code is part of the library PyDL <https://github.com/nash911/PyDL>
# This code is licensed under MIT license (see LICENSE.txt for details)
# ------------------------------------------------------------------------

import unittest
import numpy as np
import numpy.testing as npt
import itertools
from random import choice
import json


from pydl.nn.layers import FC
from pydl.nn.conv import Conv
from pydl.nn.pool import Pool
from pydl.nn.nn import NN

np.random.seed(11421111)


class TestSaveLoad(unittest.TestCase):
    def test_fc(self):
        def test(inp, layers, btchnrm=False):
            original_nn = NN(inp, layers)

            if btchnrm and inp.shape[0] > 1:
                for _ in range(1):
                    _ = original_nn.forward(np.random.uniform(-1, 1, inp.shape), inference=False)

            original_nn_out = original_nn.forward(inp, inference=True)
            nn_dict = original_nn.save()

            with open('models/test_model.nn', 'w') as mf:
                json.dump(nn_dict, mf)

            with open('models/test_model.nn') as nnf:
                loaded_nn_dict = json.load(nnf)

            loaded_nn = NN(inp, [])
            loaded_nn.load(loaded_nn_dict)
            loaded_nn_out = loaded_nn.forward(inp, inference=True)

            npt.assert_almost_equal(original_nn_out, loaded_nn_out, decimal=8)

        # Combinatorial Test Cases
        # ------------------------
        num_layers = [2, 3, 6, 10, 15, 30]
        batch_size = [1, 2, 3, 6, 11]
        feature_size = [1, 2, 3, 6, 11]
        num_neurons = [1, 2, 3, 6, 11, 25]
        scale = [1e-3, 1e-1, 1e-0, 2]
        activation_fn = ['Linear', 'Sigmoid', 'Tanh', 'Softmax', 'ReLU']
        batchnorm = [True, False]
        dropout = [True, False]
        num_reps = list(range(10))

        # Single Layer - FC
        for batch, feat, neur, scl, actv_fn, bn, dout in \
            list(itertools.product(batch_size, feature_size, num_neurons, scale, activation_fn,
                                   batchnorm, dropout)):
            X = np.random.uniform(-scl, scl, (batch, feat))
            fc_layer = FC(X, num_neurons=neur, weight_scale=scl, activation_fn=actv_fn,
                          batchnorm=bn, dropout=np.random.rand() if dout else None)
            layers = [fc_layer]
            test(X, layers, btchnrm=bn)

        # Multiple Layer - FC Net
        for batch, feat, scl, l, reps in list(itertools.product(batch_size, feature_size, scale,
                                                                num_layers, num_reps)):
            X = np.random.uniform(-scl, scl, (batch, feat))
            layers = [X]
            for _ in range(l):
                layers.append(FC(layers[-1], num_neurons=choice(num_neurons), weight_scale=scl,
                                 activation_fn=choice(activation_fn), batchnorm=choice(batchnorm),
                                 dropout=np.random.rand() if choice(dropout) else None))
            test(X, layers[1:], btchnrm=True)

    def test_cnn(self):
        def test(inp, layers, btchnrm=False):
            original_nn = NN(inp, layers)

            if btchnrm and inp.shape[0] > 1:
                for _ in range(1):
                    _ = original_nn.forward(np.random.uniform(-1, 1, inp.shape), inference=False)

            original_nn_out = original_nn.forward(inp, inference=True)
            nn_dict = original_nn.save()

            with open('models/test_model.nn', 'w') as mf:
                json.dump(nn_dict, mf)

            with open('models/test_model.nn') as nnf:
                loaded_nn_dict = json.load(nnf)

            loaded_nn = NN(inp, [])
            loaded_nn.load(loaded_nn_dict)
            loaded_nn_out = loaded_nn.forward(inp, inference=True)

            npt.assert_almost_equal(original_nn_out, loaded_nn_out, decimal=8)

            try:
                if loaded_nn.layers[-1].activation.lower() == 'softmax':
                    summed_probs = np.sum(loaded_nn_out, axis=-1)
                    npt.assert_almost_equal(summed_probs, np.ones_like(summed_probs), decimal=8)
            except AttributeError:
                pass

        def build_conv_layer(inp, name):
            inp_h = inp.shape[2]
            inp_w = inp.shape[3]

            while True:
                k_h = choice(kernal_height)
                k_w = choice(kernal_width)
                num_k = choice(num_kernals)
                pad_0 = choice(padding_0)
                pad_1 = choice(padding_1)
                strd = choice(stride)
                force = False

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

                actv_fn = choice(activation_fn)
                bn = choice(batchnorm)
                dout = choice(dropout)
                conv_layer = \
                    Conv(inp, receptive_field=(k_h, k_w), num_filters=num_k, zero_padding=pad,
                         stride=strd, activation_fn=actv_fn, batchnorm=bn, dropout=np.random.rand()
                         if dout else None, force_adjust_output_shape=force, name=name)

                # print("inp_d: %d -- inp_h: %d -- inp_w: %d -- k_h: %d -- k_w: %d -- num_k: %d -- \
                # pad_0: %d -- pad_1: %d -- strd: %d -- force: %d -- out_h: %d -- out_w: %d"
                #       % (inp.shape[1], inp_h, inp_w, k_h, k_w, num_k, pad_0, pad_1, strd, force,
                #          out_height, out_width))

                return conv_layer

        def build_pool_layer(inp, name):
            k_h = choice(kernal_height)
            k_w = choice(kernal_width)
            strd_r = choice(stride_r)
            strd_c = choice(stride_c)
            pool = choice(pool_type)

            pool_layer = Pool(inp, receptive_field=(k_h, k_w), stride=(strd_r, strd_c), pool=pool,
                              name=name)

            # print("inp_d: %d -- inp_h: %d -- inp_w: %d -- k_h: %d -- k_w: %d -- strd_r: %d \
            #         -- strd_c: %d -- pool: %s" % (inp.shape[1], inp.shape[2], inp.shape[3], k_h,
            #                                       k_w, strd_r, strd_c, pool))

            return pool_layer

        def build_fc_layer(inp, name, actv_fn=None):
            actv_fn = choice(activation_fn) if actv_fn is None else actv_fn
            btchnrm = False if actv_fn == 'SoftMax' else choice(batchnorm)
            dout = None if actv_fn == 'SoftMax' else np.random.rand() if choice(dropout) else None
            fc_layer = FC(inp, num_neurons=choice(num_neurons), activation_fn=actv_fn,
                          batchnorm=btchnrm, dropout=dout, name=name)

            return fc_layer

        # Combinatorial Test Cases
        # ------------------------
        batch_size = [1, 2, 3]
        inp_depth = [1, 2, 3]
        inp_height = [1, 2, 3, 5]
        inp_width = [1, 2, 3, 5]
        num_kernals = [1, 2, 3]
        kernal_height = [1, 3, 5]
        kernal_width = [1, 3, 5]
        padding_0 = [0, 1, 2]
        padding_1 = [0, 1, 2]
        stride = [1, 2, 3]  # Conv stride
        activation_fn = ['Linear', 'Sigmoid', 'Tanh', 'ReLU']
        batchnorm = [True, False]
        dropout = [True, False]
        force_adjust = [True, False]
        stride_r = [1, 2, 3, 4]  # Pool stride
        stride_c = [1, 2, 3, 4]  # Pool stride
        pool_type = ['AVG', 'MAX']

        # Single Layer - Conv
        for i, (batch, dep, inp_h, inp_w, num_k, k_h, k_w, pad_0, pad_1, strd, force) in \
            enumerate(list(itertools.product(batch_size, inp_depth, inp_height, inp_width,
                                             num_kernals, kernal_height, kernal_width, padding_0,
                                             padding_1, stride, force_adjust))):
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

            X = np.random.uniform(-1, 1, (batch, dep, inp_h, inp_w))
            bn = choice(batchnorm)
            conv_layer = Conv(X, receptive_field=(k_h, k_w), num_filters=num_k, zero_padding=pad,
                              stride=strd, activation_fn=choice(activation_fn),
                              batchnorm=bn, dropout=np.random.rand() if choice(dropout) else None,
                              force_adjust_output_shape=force)
            layers = [conv_layer]
            test(X, layers, btchnrm=bn)
            # print("Passed Conv Test Case %d" % i)

        # Single Layer - Pool
        for i, (batch, dep, inp_h, inp_w, k_h, k_w, strd_r, strd_c, pool) in \
            enumerate(list(itertools.product(batch_size, inp_depth, inp_height, inp_width,
                                             kernal_height, kernal_width, stride_r, stride_c,
                                             pool_type))):

            X = np.random.uniform(-1, 1, (batch, dep, inp_h, inp_w))

            pool_layer = Pool(X, receptive_field=(k_h, k_w), stride=(strd_r, strd_c), pool=pool)
            layers = [pool_layer]
            test(X, layers)
            # print("Passed Pool Test Case %d" % i)

        # Combinatorial Test Cases
        # ------------------------
        input_dims = [11, 15, 20, 24, 27, 31]
        num_neurons = [1, 2, 3, 6, 11, 25]
        num_reps = list(range(100))

        # Multiple Layer - CNN
        for i in num_reps:
            X = np.random.uniform(-1, 1, (choice(batch_size), choice(inp_depth), choice(input_dims),
                                          choice(input_dims)))
            layers = list()
            layers.append(build_conv_layer(X, name='Conv-1'))
            layers.append(build_conv_layer(layers[-1], name='Conv-2'))
            layers.append(build_pool_layer(layers[-1], name='Pool-3'))
            layers.append(build_conv_layer(layers[-1], name='Conv-4'))
            layers.append(build_pool_layer(layers[-1], name='Pool-5'))
            layers.append(build_conv_layer(layers[-1], name='Conv-6'))
            layers.append(build_pool_layer(layers[-1], name='Pool-7'))
            layers.append(build_fc_layer(layers[-1], name='FC-8'))
            layers.append(build_fc_layer(layers[-1], name='FC-9'))
            layers.append(build_fc_layer(layers[-1], name='FC-10', actv_fn='SoftMax'))

            test(X, layers, btchnrm=True)


if __name__ == '__main__':
    unittest.main()
