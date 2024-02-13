import math
import torch
import torch.utils.data
import h5py
import numpy as np
import pandas as pd


def _batched_len(things, batch_length):

    n = len(things)

    _start = 0
    for _ in range(math.ceil(n / batch_length)):
        _end = min(_start + batch_length, n)
        yield things[_start:_end]
        _start = _end


def _batched_n(things, n_batches):

    n = len(things)
    batch_length = math.floor(n / n_batches)
    extra = n - batch_length * n_batches

    _start = 0
    for i in range(n_batches):
        if i < extra:
            _end = min(_start + batch_length + 1, n)
        else:
            _end = min(_start + batch_length, n)
        yield things[_start:_end]
        _start = _end


class _H5ADLoader:

    _filehandle = None

    _data_reference = None
    _data_shape = None
    _data_sparse_format = False
    _data_sparse_indptr = None
    _data_row_index = None

    file_name = None
    layer = None

    def __init__(
        self,
        file_name,
        layer='X',
        obs_include_mask=None
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

        self._data_row_index = np.arange(self._data_shape[0])

        if obs_include_mask is not None:
            self._data_row_index = np.sort(
                self._data_row_index[obs_include_mask]
            )

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

    def _load_sparse(self, start, end):
        indptr = self._data_sparse_indptr[start:end+1]
        _left = indptr[0]
        _right = indptr[-1]

        return torch.sparse_csr_tensor(
            torch.tensor(
                indptr - _left,
                dtype=torch.int64
            ),
            torch.tensor(
                self._data_reference['indices'][_left:_right],
                dtype=torch.int64
            ),
            torch.Tensor(
                self._data_reference['data'][_left:_right]
            ),
            size=(end - start, self._data_shape[1])
        ).to_dense()

    def _load_dense(self, start, end):
        return torch.Tensor(
            self._data_reference[start:end, :]
        )

    def _load_obs_cat(self, obs_col):

        _obs_keys = list(self._filehandle['obs'].keys())
        if obs_col in _obs_keys:

            _cats = self._filehandle['obs'][obs_col]['categories'][:]

            try:
                _cats = list(
                    map(
                        lambda x: x.decode(),
                        _cats
                    )
                )
            except AttributeError:
                pass

            series = pd.Series(
                pd.Categorical.from_codes(
                    self._filehandle['obs'][obs_col]['codes'][:],
                    categories=_cats,
                    ordered=self._filehandle['obs'][obs_col].attrs['ordered']
                )
            )
            series.index = series.index.astype(str)
            series.name = obs_col
            return series
        else:
            raise ValueError(
                f"Key {obs_col} not present in "
                f"obs: {_obs_keys}"
            )

    def close(self):
        self._data_reference = None
        self._filehandle.close()


class H5ADDataset(
    _H5ADLoader,
    torch.utils.data.Dataset
):

    def __getitem__(self, i):

        row_idx = self._data_row_index[i]

        if self._data_sparse_format:
            return self._get_data_sparse(row_idx)
        else:
            return self._get_data_dense(row_idx)

    def __len__(self):
        return self._data_row_index.shape[0]

    def _get_data_dense(self, idx):
        return self._load_dense(
            idx,
            idx + 1
        ).reshape(-1)

    def _get_data_sparse(self, idx):

        return self._load_sparse(
            idx,
            idx + 1
        ).reshape(-1)


class H5ADDatasetIterable(
    _H5ADLoader,
    torch.utils.data.IterableDataset
):

    rng = None

    file_chunks = None
    _data_loaded_chunk = None
    _chunk_index_order = None

    def __init__(
        self,
        file_name,
        file_chunk_size=1000,
        random_seed=500,
        layer='X',
        obs_include_mask=None
    ):
        super().__init__(
            file_name,
            layer=layer,
            obs_include_mask=obs_include_mask
        )

        self.rng = np.random.default_rng(random_seed)

        self.file_chunks = list(_batched_len(
            self._data_row_index,
            file_chunk_size
        ))

    def load_chunk(self, chunk):

        if (
            chunk == 0 and
            len(self.file_chunks) == 1 and
            self._data_loaded_chunk is not None
        ):
            return

        if self._data_sparse_format:
            self._data_loaded_chunk = self._load_sparse(
                self.file_chunks[chunk][0],
                self.file_chunks[chunk][-1] + 1
            )
        else:
            self._data_loaded_chunk = self._load_dense(
                self.file_chunks[chunk][0],
                self.file_chunks[chunk][-1] + 1
            )

        self._data_loaded_chunk = self._data_loaded_chunk[
            self.file_chunks[chunk] - self.file_chunks[chunk][0],
            :
        ]

    def get_chunk_order(self):
        self._chunk_index_order = np.arange(self._data_loaded_chunk.shape[0])
        self.rng.shuffle(self._chunk_index_order)

    def clear_chunks(self):

        if len(self.file_chunks) > 1:
            self._data_loaded_chunk = None

        self._chunk_index_order = None

    def __iter__(self):

        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:
            return iter(self.generator())

        else:
            return iter(
                self.generator(
                    list(_batched_n(
                        np.arange(len(self.file_chunks)),
                        worker_info.num_workers
                    ))[worker_info.id]
                    )
                )

    def generator(self, worker_chunks=None):

        if worker_chunks is None:
            worker_chunks = self.file_chunks

        for c in range(len(worker_chunks)):

            self.load_chunk(c)
            self.get_chunk_order()

            for d in self._chunk_index_order:
                yield self._data_loaded_chunk[d, :]

        self.clear_chunks()


class H5ADDatasetStratified(
    H5ADDatasetIterable
):

    stratification_grouping = None
    _data_loaded_stratification = None
    _min_strat_size = None

    _data_loaded_strat = None

    def __init__(
        self,
        file_name,
        stratification_grouping_obs_column,
        file_chunk_size=1000,
        random_seed=500,
        layer='X',
        obs_include_mask=None
    ):
        super().__init__(
            file_name,
            layer=layer,
            random_seed=random_seed,
            obs_include_mask=obs_include_mask,
            file_chunk_size=file_chunk_size
        )

        self.stratification_grouping = self._load_obs_cat(
            stratification_grouping_obs_column
        )

    def load_chunk(self, chunk):

        super().load_chunk(chunk)

        _chunk_groups = self.stratification_grouping.iloc[
            self.file_chunks[chunk]
        ]
        n_groups = len(_chunk_groups.cat.categories)

        self._data_loaded_stratification = [
            np.nonzero(_chunk_groups.cat.codes.values == x)[0]
            for x in range(n_groups)
        ]

        self._min_strat_size = min(
            len(x)
            for x in self._data_loaded_stratification
            if len(x) > 0
        )

        for i in self._data_loaded_stratification:
            self.rng.shuffle(i)

    def clear_chunks(self):
        super().clear_chunks()
        self._data_loaded_stratification = None

    def get_chunk_order(self):

        self._chunk_index_order = np.ascontiguousarray(
            np.stack(tuple(
                self._data_loaded_stratification[i][:self._min_strat_size]
                for i in range(len(self._data_loaded_stratification))
                if len(self._data_loaded_stratification[i]) > 0
            ), axis=-1)
        ).reshape(-1)
