import torch

from ._base_recurrent_model import _TF_RNN_mixin


class TFRNNDecoder(_TF_RNN_mixin):

    type_name = "rnn"

    @staticmethod
    def _create_intermediate(k):
        return torch.nn.RNN(
            k,
            k,
            1,
            bias=False,
            nonlinearity='relu',
            batch_first=True
        )


class TFLSTMDecoder(_TF_RNN_mixin):

    type_name = "lstm"

    @staticmethod
    def _create_intermediate(k):
        return torch.nn.LSTM(
            k,
            k,
            1,
            bias=False,
            batch_first=True
        )


class TFGRUDecoder(_TF_RNN_mixin):

    type_name = "gru"

    @staticmethod
    def _create_intermediate(k):
        return torch.nn.GRU(
            k,
            k,
            1,
            bias=False,
            batch_first=True
        )
