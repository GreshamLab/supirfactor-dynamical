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

# Toy models
from .simple_models import (
    LogisticRegressionTorch
)

# Standard mixins
from ._model_mixins import (
    _TrainingMixin,
    _VelocityMixin,
    _MultiSubmoduleMixin,
    _MultiModalDataMixin
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
    ChromatinModule.type_name: ChromatinModule,
    LogisticRegressionTorch.type_name: LogisticRegressionTorch
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
    multisubmodel=False,
    multimodal_data=False
) -> _TFMixin:

    try:
        model = _CLASS_DICT[model]
    except KeyError:
        model = model

    if velocity and (model not in _not_velocity):
        model = [_VelocityMixin] + [model]
    else:
        model = [model]

    if multisubmodel:
        model = [_MultiSubmoduleMixin] + model

    if multimodal_data:
        model = [_MultiModalDataMixin] + model

    class SupirFactorModel(*model):
        pass

    return SupirFactorModel
