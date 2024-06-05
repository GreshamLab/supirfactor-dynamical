import h5py
import warnings
import torch
import numpy as np
import gc

from anndata._io.h5ad import read_dataframe


class _H5ADFileLoder:

    def __init__(self, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def load_file(file_name, layer):

        with h5py.File(file_name) as file_handle:

            if layer == 'X':
                _data_reference = file_handle['X']
            elif layer == 'obs':
                return read_dataframe(file_handle['obs'])
            elif layer in file_handle['layers'].keys():
                _data_reference = file_handle['layers'][layer]
            elif layer in file_handle['obsm'].keys():
                _data_reference = file_handle['obsm'][layer]
            else:
                raise ValueError(
                    f"Cannot find {layer} in `layers` or `obsm`"
                )

            if _H5ADFileLoder._issparse(_data_reference):
                return [
                    _H5ADFileLoder._load_sparse(_data_reference),
                    read_dataframe(file_handle['obs'])
                ]
            else:
                return [
                    _H5ADFileLoder._load_dense(_data_reference),
                    read_dataframe(file_handle['obs'])
                ]

    @staticmethod
    def _get_shape(ref):
        if 'shape' in ref.attrs:
            return tuple(ref.attrs['shape'])
        else:
            return ref.shape

    @staticmethod
    def _issparse(ref):

        try:
            _encoding = ref.attrs['encoding-type']
        except KeyError:
            return False

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

    @staticmethod
    def _load_sparse(ref):

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            return torch.sparse_csr_tensor(
                torch.tensor(
                    ref['indptr'][:],
                    dtype=torch.int64
                ),
                torch.tensor(
                    ref['indices'][:],
                    dtype=torch.int64
                ),
                torch.Tensor(
                    ref['data'][:]
                ),
                size=_H5ADFileLoder._get_shape(ref)
            ).to_dense()

    def _load_dense(ref):
        return torch.Tensor(ref[:])


class StratifiedFilesDataset(
    _H5ADFileLoder,
    torch.utils.data.IterableDataset
):

    file_metadata = None
    rng = None

    file_name_column = None
    file_data_layer = 'X'
    stratification_column = None

    yield_obs_cats = None
    one_hot_obs_cats = True
    obs_categories = None

    num_strat_cats = 0
    max_files = None
    epoch_n = None

    loaded_data = None
    loaded_data_order = None
    loaded_n = 0

    iter_position = 0

    def __init__(
        self,
        file_data,
        file_name_column='file',
        stratification_column='group',
        random_state=None,
        file_data_layer='X',
        yield_obs_cats=None,
        obs_categories=None,
        one_hot_obs_cats=True
    ):

        self.file_metadata = file_data.copy()
        self.file_name_column = file_name_column
        self.file_data_layer = file_data_layer

        if not isinstance(yield_obs_cats, (tuple, list)):
            yield_obs_cats = [yield_obs_cats]

        self.yield_obs_cats = yield_obs_cats
        self.one_hot_obs_cats = one_hot_obs_cats
        self.obs_categories = obs_categories

        self.stratification_column = stratification_column
        self.rng = np.random.default_rng(random_state)

        _counts = self.file_metadata[
            self.stratification_column
        ].value_counts()

        self.max_files = _counts.iloc[0]
        self.epoch_n = _counts.iloc[-1]
        self.num_strat_cats = len(_counts)

        self._get_stratifications()

    def _get_stratifications(self):

        def _extract_and_expand(cat, n):
            _values = cat.tolist()
            self.rng.shuffle(_values)

            return np.array(
                _values * int(np.ceil(n / len(_values)))
            )[0:n]

        self.strat_files = [
            _extract_and_expand(
                df[self.file_name_column],
                self.max_files
            )
            for _, df in self.file_metadata.groupby(
                self.stratification_column
            )
        ]

    def _load_stratification_files(self, i):

        if self.loaded_data is not None:
            self._delete_loaded_data()

        self.loaded_data = [
            self.load_file(x[i], layer=self.file_data_layer)
            for x in self.strat_files
        ]

        self.loaded_data_order = [
            np.arange(x[0].shape[0])
            for x in self.loaded_data
        ]

        for arr in self.loaded_data_order:
            self.rng.shuffle(arr)

        self.loaded_n = min(
            [x.shape[0] for x in self.loaded_data_order]
        )

        for i in range(len(self.loaded_data)):
            self.loaded_data[i][1] = self._get_obs_cats(
                self.loaded_data[i][1]
            )

    def _get_obs_cats(self, obs_df):

        if self.yield_obs_cats is None:
            return None

        obs_codes = []

        for col in self.yield_obs_cats:

            # Make sure the categories are unified from a master
            # dataframe
            if (
                self.obs_categories is not None and
                col in self.obs_categories.keys()
            ):
                obs_df[col] = obs_df[col].cat.set_categories(
                    self.obs_categories[col]
                )

            obs_codes.append(
                torch.LongTensor(
                    obs_df[col].cat.codes.values.copy()
                )
            )

        # Cast to one-hot
        if self.one_hot_obs_cats:
            for i, col in enumerate(self.yield_obs_cats):
                obs_codes[i] = torch.nn.functional.one_hot(
                    obs_codes[i],
                    num_classes=len(obs_df[col].cat.categories)
                ).type(
                    torch.Tensor
                )

        return obs_codes

    def _delete_loaded_data(self):
        del self.loaded_data
        del self.loaded_data_order
        self.loaded_n = 0

        gc.collect()

        self.loaded_data = None
        self.loaded_data_order = None

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        if (self.iter_position + self.epoch_n) > self.max_files:
            self._get_stratifications()
            self.iter_position = 0

        # Loop through files
        for i in range(self.epoch_n):
            self.iter_position = self.iter_position + i

            if i % num_workers != worker_id:
                continue

            self._load_stratification_files(self.iter_position)

            # Loop through loaded_n (min number of observations)
            for j in range(self.loaded_n):
                for k in range(self.num_strat_cats):
                    yield self.get_data(k, j)

            self._delete_loaded_data()

    def get_data(self, strat_id, loc):

        _data = self.loaded_data[strat_id][0][
            self.loaded_data_order[strat_id][loc]
        ]

        if self.yield_obs_cats is None:
            return _data

        if self.loaded_data[strat_id][1] is None:
            return _data

        _sample = self.loaded_data_order[strat_id][loc]
        _cat_data = tuple(
            self.loaded_data[strat_id][1][k][
                _sample
            ]
            for k in range(len(self.loaded_data[strat_id][1]))
        )

        return (_data, ) + _cat_data
