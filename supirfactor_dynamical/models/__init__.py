from .ae_model import (
    TFAutoencoder,
    TFMetaAutoencoder
)

from .recurrent_models import (
    TFRNNDecoder,
    TFLSTMDecoder,
    TFGRUDecoder
)

from .biophysical_model import SupirFactorBiophysical
from .decay_model import DecayModule
from .chromatin_model import (
    ChromatinModule,
    ChromatinAwareModel
)

# Standard mixins
from ._base_velocity_model import (
    _VelocityMixin
)
from ._base_model import _TFMixin
from ._base_trainer import _TrainingMixin


_CLASS_DICT = {
    TFAutoencoder.type_name: TFAutoencoder,
    TFMetaAutoencoder.type_name: TFMetaAutoencoder,
    TFRNNDecoder.type_name: TFRNNDecoder,
    TFGRUDecoder.type_name: TFGRUDecoder,
    TFLSTMDecoder.type_name: TFLSTMDecoder,
    SupirFactorBiophysical.type_name: SupirFactorBiophysical,
    DecayModule.type_name: DecayModule,
    ChromatinAwareModel.type_name: ChromatinAwareModel,
    ChromatinModule.type_name: ChromatinModule
}

_not_velocity = [
    SupirFactorBiophysical,
    DecayModule,
    ChromatinModule,
    ChromatinAwareModel
]


def get_model(
    model,
    velocity=False
) -> _TFMixin:

    try:
        model = _CLASS_DICT[model]
    except KeyError:
        pass

    if velocity and (model not in _not_velocity):
        class TFVelocity(_VelocityMixin, model):
            pass

        return TFVelocity

    else:
        return model
