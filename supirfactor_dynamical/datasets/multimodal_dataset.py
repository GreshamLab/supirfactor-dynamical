import torch
from scipy.sparse import issparse


class MultimodalDataset(torch.utils.data.Dataset):

    data = None
    n = None

    def __init__(
        self,
        *data
    ):
        self.n = self._n_samples(*data)
        self.data = self._to_tensor(*data)

    def __getitem__(self, i):

        return tuple(
            self._get_data(d, i) for d in self.data
        )

    def __len__(self):
        return self.n

    @staticmethod
    def _n_samples(*data):

        _ns = list(d.shape[0] for d in data)

        if len(set(_ns)) > 1:
            raise ValueError(
                f"Number of observations is not equal in each data set: "
                f"{', '.join(map(str, _ns))}"
            )

        return _ns[0]

    @staticmethod
    def _to_tensor(*data):

        return tuple(
            torch.Tensor(d)
            if (not issparse(d) and not torch.is_tensor(d))
            else d
            for d in data
        )

    @staticmethod
    def _get_data(data, idx):

        _data = data[idx, :]

        if issparse(_data):
            _data = torch.Tensor(_data.A.reshape(-1))

        return _data


class MultimodalDataLoader(torch.utils.data.DataLoader):

    def __init__(self, args, **kwargs):

        if 'collate_fn' not in kwargs.keys():
            kwargs['collate_fn'] = multimodal_collate_fn

        super().__init__(args, **kwargs)


def multimodal_collate_fn(data):

    if torch.is_tensor(data):
        return data

    return tuple(
        torch.stack([d[i] for d in data])
        for i in range(len(data[0]))
    )
