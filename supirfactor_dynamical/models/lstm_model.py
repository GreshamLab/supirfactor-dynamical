import torch

from ._base_recurrent_model import _TF_RNN_mixin


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
