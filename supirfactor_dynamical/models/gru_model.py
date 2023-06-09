import torch
from ._base_recurrent_model import _TF_RNN_mixin


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
