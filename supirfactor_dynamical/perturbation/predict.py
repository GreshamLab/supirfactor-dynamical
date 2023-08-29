import torch

from supirfactor_dynamical import SupirFactorBiophysical
from supirfactor_dynamical._utils.misc import (
    _add,
    _cat
)


def predict_perturbation(
    model: SupirFactorBiophysical,
    data: torch.Tensor,
    perturbation,
    n_time_steps,
    return_submodels=False,
    unmodified_counts=False
):
    """
    Predict transcriptional perturbation by
    inferring decay rates from unperturbed model,
    and then re-predicting transcription from
    perturbed model, using the unperturbed decay

    :param model: Fit model
    :type model: SupirFactorBiophysical
    :param data: Input data for prediction
    :type data: torch.Tensor
    :param perturbation: TF(s) to perturb
    :type perturbation: str, list(str)
    :param n_time_steps: Number of time steps to predict
    :type n_time_steps: int
    :param return_submodels: Return positive and negative velocities
        separately, defaults to False
    :type return_submodels: bool, optional
    :param unmodified_counts: Return input data counts without modification,
        defaults to False
    :type unmodified_counts: bool, optional
    :return: Predicted expression data
    :rtype: torch.Tensor
    """

    model.eval().set_drop_tfs(None)

    with torch.no_grad():
        # Unperturbed decay
        decay_rates = model(
            data,
            n_time_steps=n_time_steps,
            return_decays=True,
            unmodified_counts=unmodified_counts
        )

    model.set_drop_tfs(perturbation)

    return _perturbed_model_forward(
        model,
        data,
        decay_rates,
        perturbation,
        n_time_steps=n_time_steps,
        unmodified_counts=unmodified_counts,
        return_submodels=return_submodels
    )


def _perturbed_model_forward(
    model,
    data,
    decay_rates,
    perturbation,
    n_time_steps=0,
    unmodified_counts=False,
    return_submodels=False
):

    _L = data.shape[1]

    model.set_drop_tfs(perturbation)

    # Perturbed transcription
    _v = _get_perturbed_velocities(
        model,
        data,
        decay_rates[:, 0:_L, :]
    )

    if unmodified_counts:
        counts = data
    else:
        counts = model.next_count_from_velocity(
            data,
            _v
        )

    # Do forward predictions
    _output_velo = [_v]
    _output_count = [counts]
    _x = counts[:, [-1], :]

    # Iterate through the number of steps for prediction
    for i in range(n_time_steps):

        _offset = _L + i

        _v = _get_perturbed_velocities(
            model,
            _x,
            decay_rates[:, _offset:_offset + 1, :],
            hidden_state=True
        )

        _x = model.next_count_from_velocity(
            _x,
            _v
        )

        _output_velo.append(_v)
        _output_count.append(_x)

    model.set_drop_tfs(None)

    if n_time_steps > 0:
        _output_velo = _cat(_output_velo, 1)
        _output_count = _cat(_output_count, 1)
    else:
        _output_velo = _output_velo[0]
        _output_count = _output_count[0]

    if not return_submodels:
        _output_velo = _add(
            _output_velo[0],
            _output_velo[1]
        )

    return (
        _output_velo,
        _output_count,
        decay_rates
    )


def _get_perturbed_velocities(
    model,
    data,
    decay_rates,
    return_submodels=True,
    **kwargs
):

    # Perturbed transcription
    x_fwd, _ = model(
        data,
        return_submodels=True,
        **kwargs
    )

    x_rev = model.rescale_velocity(
        torch.multiply(
            data,
            decay_rates
        )
    )

    if return_submodels:
        return x_fwd, x_rev
    else:
        return _add(x_fwd, x_rev)
