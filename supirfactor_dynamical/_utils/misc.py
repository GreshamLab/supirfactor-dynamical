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


def _unsqueeze(x, dim):
    if isinstance(x, (tuple, list)):
        return tuple(
            _unsqueeze([d[i] for d in x], dim)
            for i in range(len(x[0]))
        )

    elif x is None or all(d is None for d in x):
        return None

    else:
        return torch.unsqueeze(x, dim)
