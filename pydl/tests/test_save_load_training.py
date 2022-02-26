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
from sklearn import datasets


from pydl.nn.layers import FC
from pydl.nn.conv import Conv
from pydl.nn.pool import Pool
from pydl.nn.nn import NN
from pydl.training.sgd import SGD
from pydl.training.momentum import Momentum
from pydl.training.rmsprop import RMSprop
from pydl.training.adam import Adam
from pydl import conf

np.random.seed(11421111)


class TestSaveLoadBreastCancer_FC(unittest.TestCase):
    def test_brest_cancer(self):
        def test(X, y, nn, norm, optimizer):
            sample_size = X.shape[0]
            order = np.random.permutation(sample_size)
            X = X[order]
            y = y[order]
            train_size = int(sample_size * 0.7)

            train_X = np.array(X[:train_size], dtype=conf.dtype)
            test_X = np.array(X[train_size:], dtype=conf.dtype)

            if optimizer.lower() == 'sgd':
                optimizer = SGD(nn, step_size=1e-3, reg_lambda=1e-2, save=True)
                optimizer.train(X, y, normalize=norm, dims=0.999, epochs=500,
                                shuffle=False, y_onehot=False, log_freq=1000, plot=None,
                                model_file='models/test_model.nn')
            elif optimizer.lower() == 'momentum':
                optimizer = Momentum(nn, step_size=1e-3, mu=0.5, reg_lambda=1e-2, save=True)
                optimizer.train(X, y, normalize=norm, epochs=500, y_onehot=False, log_freq=1000,
                                shuffle=False, plot=None, model_file='models/test_model.nn')
            elif optimizer.lower() == 'rmsprop':
                optimizer = RMSprop(nn, step_size=1e-3, beta=0.9, reg_lambda=1e-2, save=True)
                optimizer.train(X, y, normalize=norm, epochs=500, y_onehot=False, log_freq=1000,
                                shuffle=False, plot=None, model_file='models/test_model.nn')
            elif optimizer.lower() == 'adam':
                optimizer = Adam(nn, step_size=1e-3, beta_1=0.9, beta_2=0.999, reg_lambda=1e-2,
                                 save=True)
                optimizer.train(X, y, normalize=norm, epochs=500, y_onehot=False, log_freq=1000,
                                shuffle=False, plot=None, model_file='models/test_model.nn')

            normalized_train_X = optimizer.train_X
            normalized_test_X = optimizer.test_X

            original_nn_train_out = nn.forward(normalized_train_X, inference=True)
            original_nn_test_out = nn.forward(normalized_test_X, inference=True)

            with open('models/test_model.nn') as nnf:
                loaded_dict = json.load(nnf)

            loaded_nn = NN(X, [])
            loaded_nn.load(loaded_dict['nn'])

            normalizer = loaded_dict['data']['normalizer']
            if normalizer == 'mean':
                _, _, normalized_train_X = \
                    optimizer.mean_normalize(train_X,
                                             mean=np.array(loaded_dict['data']['X_mean']),
                                             std=np.array(loaded_dict['data']['X_std']))
                _, _, normalized_test_X = \
                    optimizer.mean_normalize(test_X,
                                             mean=np.array(loaded_dict['data']['X_mean']),
                                             std=np.array(loaded_dict['data']['X_std']))
            elif normalizer == 'pca':
                _, _, _, _, normalized_train_X = \
                    optimizer.reduce_data_dimensions(train_X, dims=loaded_dict['data']['X_dims'],
                                                     mean=np.array(loaded_dict['data']['X_mean']),
                                                     U=np.array(loaded_dict['data']['X_U']),
                                                     S=np.array(loaded_dict['data']['X_S']),
                                                     N=train_X.shape[0],
                                                     whiten=loaded_dict['data']['X_whiten'])
                _, _, _, _, normalized_test_X = \
                    optimizer.reduce_data_dimensions(test_X, dims=loaded_dict['data']['X_dims'],
                                                     mean=np.array(loaded_dict['data']['X_mean']),
                                                     U=np.array(loaded_dict['data']['X_U']),
                                                     S=np.array(loaded_dict['data']['X_S']),
                                                     N=test_X.shape[0],
                                                     whiten=loaded_dict['data']['X_whiten'])
            else:
                normalized_train_X = train_X
                normalized_test_X = test_X

            loaded_nn_train_out = loaded_nn.forward(normalized_train_X, inference=True)
            loaded_nn_test_out = loaded_nn.forward(normalized_test_X, inference=True)

            npt.assert_almost_equal(original_nn_train_out, loaded_nn_train_out, decimal=8)
            npt.assert_almost_equal(original_nn_test_out, loaded_nn_test_out, decimal=8)

        breast_cancer = datasets.load_breast_cancer()
        X = breast_cancer.data
        y = breast_cancer.target
        y_onehot = np.zeros((y.size, 2))
        y_onehot[range(y.size), y] = 1
        K = np.max(y) + 1

        # Combinatorial Test Cases
        # ------------------------
        num_neurons = [1, 2, 3, 6, 11, 25]
        normalization = [None, 'mean', 'pca']
        xavier = [True, False]
        hidden_activation_fn = ['Linear', 'Sigmoid', 'Tanh', 'ReLU']
        out_activation_fn = ['Sigmoid', 'Softmax']
        batchnorm = [True, False]
        dropout = [True, False]
        optimizer = ['SGD', 'Momentum', 'RMSprop', 'Adam']

        # Single Layer - FC
        for norm, out_actv_fn, bn, dout, optm in \
            list(itertools.product(normalization, out_activation_fn, batchnorm, dropout,
                                   optimizer)):
            l1 = FC(X, num_neurons=choice(num_neurons), weight_scale=0.01, xavier=choice(xavier),
                    activation_fn=choice(hidden_activation_fn), batchnorm=bn,
                    dropout=np.random.rand() if dout else None, name='FC-1')
            l2 = FC(l1, num_neurons=choice(num_neurons), weight_scale=0.01, xavier=choice(xavier),
                    activation_fn=choice(hidden_activation_fn), batchnorm=bn,
                    dropout=np.random.rand() if dout else None, name='FC-2')
            l3 = FC(l2, num_neurons=1 if out_actv_fn == 'Sigmoid' else int(K), weight_scale=0.01,
                    xavier=choice(xavier), activation_fn=out_actv_fn, name='FC-Out')
            layers = [l1, l2, l3]
            nn = NN(X, layers)

            test(X, y_onehot, nn, norm, optimizer=optm)


class TestSaveLoadBoston_FC(unittest.TestCase):
    def test_boston(self):
        def test(X, y, nn, norm, optimizer):
            sample_size = X.shape[0]
            order = np.random.permutation(sample_size)
            X = X[order]
            y = y[order]
            train_size = int(sample_size * 0.7)

            train_X = np.array(X[:train_size], dtype=conf.dtype)
            test_X = np.array(X[train_size:], dtype=conf.dtype)

            if optimizer.lower() == 'sgd':
                optimizer = SGD(nn, step_size=1e-3, reg_lambda=1e-2, regression=True, save=True)
                optimizer.train(X, y, normalize=norm, dims=0.999, epochs=500,
                                shuffle=False, y_onehot=False, log_freq=1000, plot=None,
                                model_file='models/test_model.nn')
            elif optimizer.lower() == 'momentum':
                optimizer = Momentum(nn, step_size=1e-3, mu=0.5, reg_lambda=1e-2, regression=True,
                                     save=True)
                optimizer.train(X, y, normalize=norm, epochs=500, y_onehot=False, log_freq=1000,
                                shuffle=False, plot=None, model_file='models/test_model.nn')
            elif optimizer.lower() == 'rmsprop':
                optimizer = RMSprop(nn, step_size=1e-3, beta=0.9, reg_lambda=1e-2, regression=True,
                                    save=True)
                optimizer.train(X, y, normalize=norm, epochs=500, y_onehot=False, log_freq=1000,
                                shuffle=False, plot=None, model_file='models/test_model.nn')
            elif optimizer.lower() == 'adam':
                optimizer = Adam(nn, step_size=1e-3, beta_1=0.9, beta_2=0.999, reg_lambda=1e-2,
                                 regression=True, save=True)
                optimizer.train(X, y, normalize=norm, epochs=500, y_onehot=False, log_freq=1000,
                                shuffle=False, plot=None, model_file='models/test_model.nn')

            normalized_train_X = optimizer.train_X
            normalized_test_X = optimizer.test_X

            original_nn_train_out = nn.forward(normalized_train_X, inference=True)
            original_nn_test_out = nn.forward(normalized_test_X, inference=True)

            with open('models/test_model.nn') as nnf:
                loaded_dict = json.load(nnf)

            loaded_nn = NN(X, [])
            loaded_nn.load(loaded_dict['nn'])

            normalizer = loaded_dict['data']['normalizer']
            if normalizer == 'mean':
                _, _, normalized_train_X = \
                    optimizer.mean_normalize(train_X,
                                             mean=np.array(loaded_dict['data']['X_mean']),
                                             std=np.array(loaded_dict['data']['X_std']))
                _, _, normalized_test_X = \
                    optimizer.mean_normalize(test_X,
                                             mean=np.array(loaded_dict['data']['X_mean']),
                                             std=np.array(loaded_dict['data']['X_std']))
            elif normalizer == 'pca':
                _, _, _, _, normalized_train_X = \
                    optimizer.reduce_data_dimensions(train_X, dims=loaded_dict['data']['X_dims'],
                                                     mean=np.array(loaded_dict['data']['X_mean']),
                                                     U=np.array(loaded_dict['data']['X_U']),
                                                     S=np.array(loaded_dict['data']['X_S']),
                                                     N=train_X.shape[0],
                                                     whiten=loaded_dict['data']['X_whiten'])
                _, _, _, _, normalized_test_X = \
                    optimizer.reduce_data_dimensions(test_X, dims=loaded_dict['data']['X_dims'],
                                                     mean=np.array(loaded_dict['data']['X_mean']),
                                                     U=np.array(loaded_dict['data']['X_U']),
                                                     S=np.array(loaded_dict['data']['X_S']),
                                                     N=test_X.shape[0],
                                                     whiten=loaded_dict['data']['X_whiten'])
            else:
                normalized_train_X = train_X
                normalized_test_X = test_X

            loaded_nn_train_out = loaded_nn.forward(normalized_train_X, inference=True)
            loaded_nn_test_out = loaded_nn.forward(normalized_test_X, inference=True)

            npt.assert_almost_equal(original_nn_train_out, loaded_nn_train_out, decimal=8)
            npt.assert_almost_equal(original_nn_test_out, loaded_nn_test_out, decimal=8)

        boston = datasets.load_boston()
        X = boston.data
        y = boston.target

        # Combinatorial Test Cases
        # ------------------------
        num_neurons = [1, 2, 3, 6, 11, 25]
        normalization = [None, 'mean', 'pca']
        xavier = [True, False]
        activation_fn = ['Linear', 'Sigmoid', 'Tanh', 'ReLU']
        batchnorm = [True, False]
        dropout = [True, False]
        optimizer = ['SGD', 'Momentum', 'RMSprop', 'Adam']

        # Single Layer - FC
        for norm, bn, dout, optm in list(itertools.product(normalization, batchnorm, dropout,
                                                           optimizer)):
            l1 = FC(X, num_neurons=choice(num_neurons), weight_scale=0.01, xavier=choice(xavier),
                    activation_fn=choice(activation_fn), batchnorm=bn,
                    dropout=np.random.rand() if dout else None, name='FC-1')
            l2 = FC(l1, num_neurons=choice(num_neurons), weight_scale=0.01, xavier=choice(xavier),
                    activation_fn=choice(activation_fn), batchnorm=bn,
                    dropout=np.random.rand() if dout else None, name='FC-2')
            l3 = FC(l2, num_neurons=1, weight_scale=0.01, xavier=choice(xavier),
                    activation_fn=choice(activation_fn), name='FC-Out')
            layers = [l1, l2, l3]
            nn = NN(X, layers)

            test(X, y, nn, norm, optimizer=optm)


class TestSaveLoadMNIST_Conv(unittest.TestCase):
    def test_mnist_conv(self):
        def test(X, y, nn, norm, optimizer):
            sample_size = X.shape[0]
            order = np.random.permutation(sample_size)
            X = X[order]
            y = y[order]
            train_size = int(sample_size * 0.7)

            train_X = np.array(X[:train_size], dtype=conf.dtype)
            test_X = np.array(X[train_size:], dtype=conf.dtype)

            if optimizer.lower() == 'sgd':
                optimizer = SGD(nn, step_size=1e-3, reg_lambda=1e-2, regression=True, save=True)
                optimizer.train(X, y, normalize=norm, dims=0.999, epochs=5, shuffle=False,
                                log_freq=1000, plot=None, model_file='models/test_model.nn')
            elif optimizer.lower() == 'momentum':
                optimizer = Momentum(nn, step_size=1e-3, mu=0.5, reg_lambda=1e-2, regression=True,
                                     save=True)
                optimizer.train(X, y, normalize=norm, epochs=5, log_freq=1000, shuffle=False,
                                plot=None, model_file='models/test_model.nn')
            elif optimizer.lower() == 'rmsprop':
                optimizer = RMSprop(nn, step_size=1e-3, beta=0.9, reg_lambda=1e-2, regression=True,
                                    save=True)
                optimizer.train(X, y, normalize=norm, epochs=5, log_freq=1000, shuffle=False,
                                plot=None, model_file='models/test_model.nn')
            elif optimizer.lower() == 'adam':
                optimizer = Adam(nn, step_size=1e-3, beta_1=0.9, beta_2=0.999, reg_lambda=1e-2,
                                 regression=True, save=True)
                optimizer.train(X, y, normalize=norm, epochs=5, log_freq=1000, shuffle=False,
                                plot=None, model_file='models/test_model.nn')

            normalized_train_X = optimizer.train_X
            normalized_test_X = optimizer.test_X

            original_nn_train_out = nn.forward(normalized_train_X, inference=True)
            original_nn_test_out = nn.forward(normalized_test_X, inference=True)

            with open('models/test_model.nn') as nnf:
                loaded_dict = json.load(nnf)

            loaded_nn = NN(X, [])
            loaded_nn.load(loaded_dict['nn'])

            normalizer = loaded_dict['data']['normalizer']
            if normalizer == 'mean':
                _, _, normalized_train_X = \
                    optimizer.mean_normalize(train_X,
                                             mean=np.array(loaded_dict['data']['X_mean']),
                                             std=np.array(loaded_dict['data']['X_std']))
                _, _, normalized_test_X = \
                    optimizer.mean_normalize(test_X,
                                             mean=np.array(loaded_dict['data']['X_mean']),
                                             std=np.array(loaded_dict['data']['X_std']))
            else:
                normalized_train_X = train_X
                normalized_test_X = test_X

            loaded_nn_train_out = loaded_nn.forward(normalized_train_X, inference=True)
            loaded_nn_test_out = loaded_nn.forward(normalized_test_X, inference=True)

            npt.assert_almost_equal(original_nn_train_out, loaded_nn_train_out, decimal=8)
            npt.assert_almost_equal(original_nn_test_out, loaded_nn_test_out, decimal=8)

        mnist = datasets.fetch_openml('mnist_784')
        X = np.array(mnist.data, dtype=conf.dtype).reshape(-1, 1, 28, 28)[:500]
        y = np.array(mnist.target, dtype=np.int)[:500]
        K = np.max(y) + 1

        # Data Stats
        print("Data Size: ", X.shape[0])
        print("Feature Size: ", X.shape[1])

        # Combinatorial Test Cases
        # ------------------------
        num_neurons = [1, 2, 3, 6, 11, 25]
        normalization = [None, 'mean']
        xavier = [True, False]
        activation_fn = ['Linear', 'Sigmoid', 'Tanh', 'ReLU']
        batchnorm = [True, False]
        dropout = [True, False]
        optimizer = ['SGD', 'Momentum', 'RMSprop', 'Adam']

        # Single Layer - FC
        for norm, bn, dout, optm in list(itertools.product(normalization, batchnorm, dropout,
                                                           optimizer)):
            l1 = Conv(X, receptive_field=(3, 3), num_filters=4, zero_padding=1, stride=1,
                      weight_scale=1.0, xavier=choice(xavier), activation_fn=choice(activation_fn),
                      batchnorm=choice(batchnorm), dropout=np.random.rand() if dout else None,
                      name='Conv-1')
            l2 = Pool(l1, receptive_field=(2, 2), stride=2, pool='MAX', name='MaxPool-2')
            l3 = Conv(l2, receptive_field=(3, 3), num_filters=3, zero_padding=1, stride=1,
                      weight_scale=1.0, xavier=choice(xavier), activation_fn=choice(activation_fn),
                      batchnorm=choice(batchnorm), dropout=np.random.rand() if dout else None,
                      name='Conv-3',)
            l4 = Pool(l3, receptive_field=None, stride=1, pool='AVG', name='MaxPool-4')
            l5 = FC(l4, num_neurons=choice(num_neurons), weight_scale=1.0, xavier=choice(xavier),
                    activation_fn=choice(activation_fn), batchnorm=choice(batchnorm),
                    dropout=np.random.rand() if dout else None, name="FC-5")
            l6 = FC(l5, num_neurons=int(K), weight_scale=1.0, xavier=choice(xavier),
                    activation_fn='SoftMax', name="Output-Layer")

            layers = [l1, l2, l3, l4, l5, l6]
            nn = NN(X, layers)

            test(X, y, nn, norm, optimizer=optm)


if __name__ == '__main__':
    unittest.main()
