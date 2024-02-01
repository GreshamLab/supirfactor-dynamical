import torch


def to(obj, device):

    if obj is None:
        return None

    if isinstance(obj, (tuple, list)):
        return tuple(to(thing, device) for thing in obj)

    else:
        return obj.to(device)


def _cat(_data, dim):
    """
    Concatenate list of tensors or list of tensor tuples
    If a list of tuples, returns a tuple of concatenated tensors

    :param _data: Data
    :type _data: torch.Tensor, list, tuple
    :param dim: Dimension to concatenate
    :type dim: int
    :return: Concatenated data
    :rtype: torch.Tensor, tuple
    """

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
