from .ae_model import (
    TFAutoencoder,
    TFMetaAutoencoder,
    TFMultilayerAutoencoder
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
from ._model_mixins import (
    _TrainingMixin,
    _VelocityMixin,
    _MultiSubmoduleMixin
)
from ._base_model import _TFMixin


_CLASS_DICT = {
    TFAutoencoder.type_name: TFAutoencoder,
    TFMetaAutoencoder.type_name: TFMetaAutoencoder,
    TFMultilayerAutoencoder.type_name: TFMultilayerAutoencoder,
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
    velocity=False,
    multisubmodel=False
) -> _TFMixin:

    try:
        model = _CLASS_DICT[model]
    except KeyError:
        pass

    if velocity and (model not in _not_velocity):
        class TFVelocity(_VelocityMixin, model):
            pass

        model = TFVelocity

    if multisubmodel:
        class TFMultimodule(_MultiSubmoduleMixin, model):
            pass

        model = TFMultimodule

    return model
