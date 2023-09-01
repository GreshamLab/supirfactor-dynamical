import torch

from .predict import predict_perturbation
from .._utils.misc import _cat


def perturbation_tfa_gradient(
    model,
    input_data,
    observed_data,
    observed_data_type='count',
    perturbation=None,
    observed_data_delta_t=0,
    loss_function=torch.nn.MSELoss()
):

    _grads = []
    _handle = model._transcription_model.encoder[1].register_backward_hook(
        lambda m, gi, go: _grads.append(torch.clone(go[0]))
    )

    predicts = predict_perturbation(
        model,
        input_data,
        perturbation,
        observed_data_delta_t
    )

    if observed_data_type == 'count':
        _predict = predicts[1]
    elif observed_data_type == 'velocity':
        _predict = predicts[0]
    elif observed_data_type == 'decay_rate':
        _predict = predicts[2]

    if observed_data_delta_t is not None:
        _predict = _predict[:, [-1], :]

    loss = loss_function(
        _predict,
        observed_data
    )

    loss.backward()

    _handle.remove()

    return loss, _cat(_grads, 1), predicts[3]
