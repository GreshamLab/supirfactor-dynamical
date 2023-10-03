import torch
from scipy.sparse import isspmatrix


class ChromatinDataset(torch.utils.data.Dataset):

    expression = None
    chromatin = None

    n = None

    @property
    def expression_sparse(self):
        if self.expression is None:
            return None
        else:
            return isspmatrix(self.expression)

    @property
    def chromatin_sparse(self):
        if self.chromatin is None:
            return None
        else:
            return isspmatrix(self.chromatin)

    def __init__(
        self,
        gene_expression_data,
        chromatin_state_data
    ):
        self.n = self._n_samples(
            gene_expression_data,
            chromatin_state_data
        )

        self.chromatin = chromatin_state_data
        self.expression = gene_expression_data

        if not torch.is_tensor(self.expression) and not self.expression_sparse:
            self.expression = torch.Tensor(self.expression)

        if not torch.is_tensor(self.chromatin) and not self.chromatin_sparse:
            self.chromatin = torch.Tensor(self.chromatin)

    def __getitem__(self, i):

        e = self.expression[i, :]
        c = self.chromatin[i, :]

        if isspmatrix(e):
            e = torch.Tensor(e.A.ravel())

        if isspmatrix(c):
            c = torch.Tensor(c.A.ravel())

        return e, c

    def __len__(self):
        return self.n

    @staticmethod
    def _n_samples(expr, peaks):

        _n_expr = expr.shape[0]
        _n_peaks = peaks.shape[0]

        if _n_expr != _n_peaks:
            raise ValueError(
                f"Expression data {expr.shape} and peak data {peaks.shape} "
                "do not have the same number of observations"
            )

        return _n_expr


class ChromatinDataLoader(torch.utils.data.DataLoader):

    def __init__(self, args, **kwargs):
        kwargs['collate_fn'] = chromatin_collate_fn
        super().__init__(args, **kwargs)


def chromatin_collate_fn(data):

    return (
        torch.stack([d[0] for d in data]),
        torch.stack([d[1] for d in data])
    )
