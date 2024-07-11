import numpy as np


def check_loss_for_early_stop(
    loss,
    min_delta=1e-4,
    patience=10,
    smear_recent_loss=5
):

    _look = patience + smear_recent_loss

    if len(loss) < _look:
        return False

    mean_old_loss = np.mean(loss[-1*_look:-1*smear_recent_loss])
    mean_new_loss = np.mean(loss[-1*smear_recent_loss:])

    if (mean_old_loss - mean_new_loss) < min_delta:
        return True
    else:
        return False
