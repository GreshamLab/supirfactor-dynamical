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
