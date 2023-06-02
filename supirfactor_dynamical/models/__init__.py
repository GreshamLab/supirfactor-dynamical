from .ae_model import (
    TFAutoencoder,
    TFMetaAutoencoder
)

from .rnn_model import (
    TFRNNDecoder
)

from .lstm_model import (
    TFLSTMDecoder
)

from .gru_model import (
    TFGRUDecoder
)

_CLASS_DICT = {
    TFAutoencoder.type_name: TFAutoencoder,
    TFMetaAutoencoder.type_name: TFMetaAutoencoder,
    TFRNNDecoder.type_name: TFRNNDecoder,
    TFGRUDecoder.type_name: TFGRUDecoder,
    TFLSTMDecoder.type_name: TFLSTMDecoder
}
