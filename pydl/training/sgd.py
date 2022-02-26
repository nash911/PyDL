# ------------------------------------------------------------------------
# MIT License
#
# Copyright (c) [2021] [Avinash Ranganath]
#
# This code is part of the library PyDL <https://github.com/nash911/PyDL>
# This code is licensed under MIT license (see LICENSE.txt for details)
# ------------------------------------------------------------------------

from collections import OrderedDict

from pydl.training.plain_training import PlainTraining
from pydl.training.recurrent_training import RecurrentTraining


class SGD(PlainTraining, RecurrentTraining):
    def __init__(self, nn=None, step_size=1e-2, reg_lambda=0, train_size=70, test_size=30,
                 activatin_type=None, regression=False, save=False, name=None):
        super().__init__(nn=nn, step_size=step_size, reg_lambda=reg_lambda, train_size=train_size,
                         test_size=test_size, activatin_type=activatin_type, regression=regression,
                         save=save, name=name)

        if save:
            training_dict = OrderedDict()
            training_dict['step_size'] = step_size
            training_dict['reg_lambda'] = reg_lambda
            training_dict['train_size'] = train_size
            training_dict['test_size'] = test_size
            training_dict['regression'] = regression
            self._save_dict['training'] = training_dict

    def train(self, X, y, normalize=None, dims=None, shuffle=True, batch_size=256, epochs=1000,
              y_onehot=False, plot=None, log_freq=100, model_file=None):
        self.prepare_data(X=X, y=y, normalize=normalize, dims=dims, shuffle=shuffle,
                          batch_size=batch_size, y_onehot=y_onehot)

        training_logs_dict = super().train(batch_size=batch_size, epochs=epochs, plot=plot,
                                           log_freq=log_freq, model_file=model_file)
        return training_logs_dict

    def train_recurrent(self, X, y=None, batch_size=256, epochs=1000, sample_length=100,
                        normalize=None, data_diff=False, temperature=1.0, plot=None,
                        fit_test_data=False, log_freq=100):
        self.prepare_sequence_data(X, y, normalize, data_diff)

        training_logs_dict = \
            super().train_recurrent(batch_size=batch_size, epochs=epochs, plot=plot,
                                    sample_length=sample_length, normalize=normalize,
                                    data_diff=data_diff, temperature=temperature,
                                    fit_test_data=fit_test_data, log_freq=log_freq)
        return training_logs_dict

    def update_network(self, t=None):
        for layer in self._nn.layers:
            if layer.type in ['FC_Layer', 'Convolution_Layer', 'RNN_Layer', 'LSTM_Layer',
                              'GRU_Layer']:
                # Update Weights
                layer.weights += -self._step_size * layer.weights_grad

                # Update Bias
                if layer.bias is not None:
                    layer.bias += -self._step_size * layer.bias_grad

                # Update Cell State
                if layer.cell_state_grad is not None:
                    layer.init_cell_state += -self._step_size * layer.cell_state_grad

                # Update Hidden State
                if layer.hidden_state_grad is not None:
                    layer.init_hidden_state += -self._step_size * layer.hidden_state_grad
            layer.reset()
