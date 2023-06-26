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

from ._base_velocity_model import (
    _VelocityMixin,
    _DecayMixin
)

from ._base_model import _TFMixin

_CLASS_DICT = {
    TFAutoencoder.type_name: TFAutoencoder,
    TFMetaAutoencoder.type_name: TFMetaAutoencoder,
    TFRNNDecoder.type_name: TFRNNDecoder,
    TFGRUDecoder.type_name: TFGRUDecoder,
    TFLSTMDecoder.type_name: TFLSTMDecoder
}


def get_model(
    model,
    velocity=False,
    decay=False
) -> _TFMixin:

    try:
        model = _CLASS_DICT[model]
    except KeyError:
        pass

    if velocity and decay:
        class TFVelocityDecay(_DecayMixin, _VelocityMixin, model):
            pass

        return TFVelocityDecay

    elif velocity:
        class TFVelocity(_VelocityMixin, model):
            pass

        return TFVelocity

    elif decay:
        class TFDecay(_DecayMixin, model):
            pass

        return TFDecay

    else:
        return model
