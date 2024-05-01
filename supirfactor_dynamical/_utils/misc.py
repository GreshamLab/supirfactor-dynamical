import torch


def to(obj, device):

    if obj is None:
        return None

    if isinstance(obj, (tuple, list)):
        return tuple(to(thing, device) for thing in obj)

    else:
        return obj.to(device)


def argmax_last_dim(x):
    """
    Get the argmax for the last dimension of a tensor

    :param x: Tensor
    :type x: torch.tensor
    :return: Argmax of the last dimension
    :rtype: torch.tensor
    """

    return torch.argmax(
        x.view(-1, x.shape[-1]),
        axis=1
    ).reshape(x.shape[:-1])


def _to_tensor(data):

    if isinstance(data, tuple):
        return tuple(
            _to_tensor(x)
            for x in data
        )

    if not torch.is_tensor(data):
        data = torch.Tensor(data)

    return data


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


def _nobs(x):

    if isinstance(x, (tuple, list)):
        return _nobs(x[0])

    if x.ndim == 1:
        return 1
    else:
        return x.shape[0]


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
