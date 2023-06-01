from .ae_model import (
    TFAutoencoder
)

from .rnn_model import (
    TFRecurrentAutoencoder,
    TFRNNDecoder
)

from .lstm_model import (
    TFLSTMAutoencoder,
    TFLSTMDecoder
)

from .gru_model import (
    TFGRUAutoencoder,
    TFGRUDecoder
)

_CLASS_DICT = {
    TFAutoencoder.type_name: TFAutoencoder,
    TFRecurrentAutoencoder.type_name: TFRecurrentAutoencoder,
    TFLSTMAutoencoder.type_name: TFLSTMAutoencoder,
    TFGRUAutoencoder.type_name: TFGRUAutoencoder,
    TFRNNDecoder.type_name: TFRNNDecoder,
    TFGRUDecoder.type_name: TFGRUDecoder,
    TFLSTMDecoder.type_name: TFLSTMDecoder
}
