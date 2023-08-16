import torch


def _cat(_data, dim):

    if isinstance(_data[0], tuple):
        return (
            _cat([d[0] for d in _data], dim),
            _cat([d[1] for d in _data], dim)
        )

    else:
        return torch.cat(
            _data,
            dim=dim
        )


def _add(x, y):

    if x is None and y is None:
        return None
    elif x is None:
        return y
    elif y is None:
        return x
    else:
        return torch.add(x, y)
