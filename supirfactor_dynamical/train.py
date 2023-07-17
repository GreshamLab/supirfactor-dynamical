from .models import (
    _CLASS_DICT,
    TFMetaAutoencoder,
    TFRNNDecoder
)

from .models._base_model import (
    _TFMixin
)

from . import evaluate_results

DROPOUT_KWARGS = [
    ('input_dropout_rate', 0.5),
    ('hidden_dropout_rate', 0.0)
]


def _is_model(model):

    try:
        return issubclass(model, _TFMixin)
    except TypeError:
        return False


def pretrain_and_tune_dynamic_model(
    pretraining_training_dataloader,
    prediction_tuning_training_dataloader,
    prior_network,
    epochs,
    optimizer_params=None,
    pretraining_validation_dataloader=None,
    prediction_tuning_validation_dataloader=None,
    prediction_length=None,
    prediction_loss_offset=None,
    model_type=TFRNNDecoder,
    return_erv=False,
    **kwargs
):

    pretrain_dropout, tune_dropout = _dropout_kwargs(
        kwargs
    )

    model, pretrain_results = model_training(
        pretraining_training_dataloader,
        prior_network,
        epochs,
        validation_dataloader=pretraining_validation_dataloader,
        optimizer_params=optimizer_params,
        prediction_length=False,
        model_type=model_type,
        **kwargs,
        **pretrain_dropout
    )

    model.train()

    model.set_dropouts(**tune_dropout)

    model, tuned_results, _final_erv = model_training(
        prediction_tuning_training_dataloader,
        prior_network,
        epochs,
        validation_dataloader=prediction_tuning_validation_dataloader,
        optimizer_params=optimizer_params,
        prediction_length=prediction_length,
        prediction_loss_offset=prediction_loss_offset,
        model_type=model,
        return_erv=True,
        **kwargs
    )

    if return_erv:
        return model, pretrain_results, tuned_results, _final_erv
    else:
        return model, pretrain_results, tuned_results


def model_training(
    training_dataloader,
    prior_network,
    epochs,
    validation_dataloader=None,
    optimizer_params=None,
    gold_standard=None,
    model_type=None,
    prediction_length=None,
    prediction_loss_offset=None,
    return_erv=False,
    **kwargs
):

    # extract kwargs
    scaling_params = dict(
        count_scaling=kwargs.pop('count_scaling', None),
        velocity_scaling=kwargs.pop('velocity_scaling', None)
    )

    # If model_type is an instance of a model
    # use it as-is
    if isinstance(model_type, _TFMixin):
        model_obj = model_type

    # Otherwise get a model and instantiate it
    else:

        if model_type is None:
            model = TFMetaAutoencoder

        elif _is_model(model_type):
            model = model_type

        else:
            model = _CLASS_DICT[model_type]

        model_obj = model(
            prior_network,
            **kwargs
        )

    if hasattr(model_obj, 'set_scaling'):
        model_obj.set_scaling(
            **scaling_params
        )

    model_obj.set_time_parameters(
        output_t_plus_one=prediction_length is not None,
        n_additional_predictions=prediction_length,
        loss_offset=prediction_loss_offset
    )

    model_obj.train_model(
        training_dataloader,
        epochs,
        validation_dataloader=validation_dataloader,
        optimizer=optimizer_params
    )

    model_obj.eval()

    _weights = model_obj.output_weights(
        as_dataframe=True
    )

    _erv = model_obj.erv(
        training_dataloader,
        as_data_frame=True
    )

    result = evaluate_results(
        _weights,
        _erv,
        prior_network,
        gold_standard if gold_standard is not None else prior_network
    )

    if return_erv:
        return model_obj, result, _erv
    else:
        return model_obj, result


def joint_model_training(
    static_training_dataloader,
    dynamic_training_dataloader,
    prior_network,
    epochs,
    static_validation_dataloader=None,
    dynamic_validation_dataloader=None,
    optimizer_params=None,
    gold_standard=None,
    prediction_length=None,
    prediction_loss_offset=None,
    static_model_type=TFMetaAutoencoder,
    dynamic_model_type=TFRNNDecoder,
    **kwargs
):

    ae_static, ae_results = model_training(
        static_training_dataloader,
        prior_network,
        epochs,
        validation_dataloader=static_validation_dataloader,
        optimizer_params=optimizer_params,
        gold_standard=gold_standard,
        prediction_length=prediction_length,
        prediction_loss_offset=prediction_loss_offset,
        model_type=static_model_type,
        **kwargs
    )

    ae_dynamic, dyn_results = model_training(
        dynamic_training_dataloader,
        prior_network,
        epochs,
        validation_dataloader=dynamic_validation_dataloader,
        optimizer_params=optimizer_params,
        gold_standard=gold_standard,
        prediction_length=prediction_length,
        prediction_loss_offset=prediction_loss_offset,
        model_type=dynamic_model_type,
        **kwargs
    )

    return ae_static, ae_results, ae_dynamic, dyn_results


def _dropout_kwargs(kwargs):

    dropout_kwargs = [
        {
            k: v if not isinstance(v, tuple) else v[i]
            for k, v in {
                k: kwargs.pop(k, v)
                for k, v in DROPOUT_KWARGS
            }.items()
        }
        for i in range(2)
    ]

    return dropout_kwargs[0], dropout_kwargs[1]
