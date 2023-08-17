import torch


def _cat(_data, dim):

    if isinstance(_data[0], (tuple, list)):
        return tuple(
            _cat([d[i] for d in _data], dim)
            for i in range(len(_data[0]))
        )

    elif _data is None or all(x is None for x in _data):
        return None

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
