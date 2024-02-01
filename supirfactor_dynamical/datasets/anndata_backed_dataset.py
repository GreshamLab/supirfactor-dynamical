import torch
import h5py
import numpy as np
from scipy.sparse import csr_array


class H5ADDataset(torch.utils.data.Dataset):

    _filehandle = None

    _data_reference = None
    _data_shape = None
    _data_sparse_format = False
    _data_sparse_indptr = None

    file_name = None
    layer = None

    def __init__(
        self,
        file_name,
        layer='X'
    ):
        self.file_name = file_name
        self.layer = layer

        self._filehandle = h5py.File(file_name)

        if layer == 'X':
            self._data_reference = self._filehandle['X']
        else:
            self._data_reference = self._filehandle['layers'][layer]

        self._data_shape = self._get_shape(self._data_reference)
        self._data_sparse_format = self._get_issparse(self._data_reference)

        if self._data_sparse_format:
            self._data_sparse_indptr = self._data_reference['indptr'][:]

    def __del__(
        self
    ):
        if self._filehandle is not None:
            self._filehandle.close()

        super()

    def __getitem__(self, i):

        if self._data_sparse_format:
            return self._get_data_sparse(i)
        else:
            return self._get_data_dense(i)

    def __len__(self):
        return self._data_shape[0]

    @staticmethod
    def _get_shape(ref):
        if 'shape' in ref.attrs:
            return tuple(ref.attrs['shape'])
        else:
            return ref.shape

    @staticmethod
    def _get_issparse(ref):
        _encoding = ref.attrs['encoding-type']

        if _encoding == 'array':
            return False
        elif _encoding == 'csr_matrix':
            return True
        elif _encoding == 'csc_matrix':
            raise RuntimeError(
                "Sparse data must be CSR because sampling "
                "is row-wise"
            )
        else:
            raise ValueError(f"{_encoding} unknown")

    def _get_data_dense(self, idx):
        return torch.Tensor(self._data_reference[idx, :])

    def _get_data_sparse(self, idx):
        _left = self._data_sparse_indptr[idx]
        _right = self._data_sparse_indptr[idx + 1]

        return torch.Tensor(
            csr_array(
                (
                    self._data_reference['data'][_left:_right],
                    self._data_reference['indices'][_left:_right],
                    np.array([0, _right - _left])
                ),
                shape=(1, self._data_shape[1])
            ).todense().reshape(-1)
        )
