import torch

from supirfactor_dynamical.models.biophysical_model import (
    SupirFactorBiophysical,
    _cat
)
from supirfactor_dynamical._utils.misc import _add


def predict_perturbation(
    model: SupirFactorBiophysical,
    data: torch.Tensor,
    perturbation,
    n_time_steps,
    return_submodels=False,
    unmodified_counts=False
):

    _L = data.shape[1]

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

    with torch.no_grad():
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
