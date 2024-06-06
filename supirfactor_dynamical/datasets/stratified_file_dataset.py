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
    def load_file(
        file_name,
        layer,
        extra_layers=None,
        append_obs=True
    ):

        with h5py.File(file_name) as file_handle:

            if layer == 'obs':
                return _H5ADFileLoder.load_layer(file_handle, layer)

            _data = [_H5ADFileLoder.load_layer(file_handle, layer)]

            if extra_layers is not None:
                _data = _data + [
                    _H5ADFileLoder.load_layer(file_handle, _elayer)
                    for _elayer in extra_layers
                ]

            if append_obs:
                _data.append(
                    _H5ADFileLoder.load_layer(file_handle, 'obs')
                )

            return _data

    @staticmethod
    def load_layer(file_handle, layer):
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
            return _H5ADFileLoder._load_sparse(_data_reference)
        else:
            return _H5ADFileLoder._load_dense(_data_reference)

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

    @staticmethod
    def _load_dense(ref):
        return torch.Tensor(ref[:])

    @staticmethod
    def _get_obs_cats(
        obs_df,
        cols,
        col_level_dict,
        one_hot=False
    ):

        if cols is None:
            return []

        obs_codes = []

        for col in cols:

            # Make sure the categories are unified from a master
            # dataframe
            if (
                col_level_dict is not None and
                col in col_level_dict.keys()
            ):
                obs_df[col] = obs_df[col].cat.set_categories(
                    col_level_dict[col]
                )

            obs_codes.append(
                torch.LongTensor(
                    obs_df[col].cat.codes.values.copy()
                )
            )

        # Cast to one-hot
        if one_hot:
            for i, col in enumerate(cols):
                obs_codes[i] = torch.nn.functional.one_hot(
                    obs_codes[i],
                    num_classes=len(obs_df[col].cat.categories)
                ).type(
                    torch.Tensor
                )

        return obs_codes


class StratifiedFilesDataset(
    _H5ADFileLoder,
    torch.utils.data.IterDataPipe
):

    file_metadata = None
    rng = None
    prefetch = False

    file_name_column = None
    file_data_layer = 'X'
    yield_extra_layers = None
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
    loaded_data_index = None
    job = None

    iter_position = 0
    worker_id = None
    num_workers = None

    def __init__(
        self,
        file_data,
        file_name_column='file',
        stratification_column='group',
        random_state=None,
        file_data_layer='X',
        yield_extra_layers=None,
        yield_obs_cats=None,
        obs_categories=None,
        one_hot_obs_cats=True,
        epoch_len=None
    ):

        self.file_metadata = file_data.copy()
        self.file_name_column = file_name_column
        self.file_data_layer = file_data_layer
        self.yield_extra_layers = yield_extra_layers

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
        self.num_strat_cats = len(_counts)

        if epoch_len is None:
            self.epoch_n = _counts.iloc[-1]
        else:
            self.epoch_n = epoch_len

        self._get_stratifications()

    def set_random_state(self, random_state):
        self.rng = np.random.default_rng(random_state)

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

        if self.loaded_data_index == i:
            return

        if self.loaded_data is not None:
            self._delete_loaded_data()

        # Load each file into a list [C, ]
        self.loaded_data = [
            self.load_file(
                x[i],
                layer=self.file_data_layer,
                extra_layers=self.yield_extra_layers
            )
            for x in self.strat_files
        ]

        # Create a random access order index in list [C, ]
        # for each category
        self.loaded_data_order = [
            np.arange(x[0].shape[0])
            for x in self.loaded_data
        ]

        for arr in self.loaded_data_order:
            self.rng.shuffle(arr)

        # Use the smallest number of observations in any
        # category as the number of observations to yield
        # to DataLoader
        self.loaded_n = min(
            [x.shape[0] for x in self.loaded_data_order]
        )

        # Break the obs dataframe in the final position
        # into tensors and put them into the list
        # if yield_obs_cats is not None
        for i in range(len(self.loaded_data)):
            self.loaded_data[i].extend(
                self._get_obs_cats(
                    self.loaded_data[i].pop(-1),
                    self.yield_obs_cats,
                    self.obs_categories,
                    one_hot=self.one_hot_obs_cats
                )
            )

        self.loaded_data_index = i

    def _delete_loaded_data(self):
        del self.loaded_data
        del self.loaded_data_order
        self.loaded_n = 0
        self.loaded_data_index = None

        gc.collect()

        self.loaded_data = None
        self.loaded_data_order = None

    def worker_info(self):

        if self.worker_id is not None:
            return self.worker_id, self.num_workers

        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            self.worker_id = worker_info.id
            self.num_workers = worker_info.num_workers
            worker_id = self.worker_id
            num_workers = self.num_workers

        return worker_id, num_workers

    def worker_jobs(self):

        if self.job is None:

            # Get jobs that this worker should do
            worker_id, num_workers = self.worker_info()

            # Assign stratification batch numbers to this worker
            jobs = np.arange(0, self.max_files, num_workers)
            jobs += worker_id
            jobs = jobs[jobs < self.max_files]

            def _gen():
                while True:

                    # Run through the jobs
                    for j in jobs:
                        self._load_stratification_files(j)
                        yield j

                    # Call the stratification reshuffler
                    self._delete_loaded_data()
                    self._get_stratifications()

            self.job = _gen()

    def __iter__(self):

        self.worker_jobs()
        next(self.job)

        def _get_loaded_data():
            for j in range(self.loaded_n):
                for k in range(self.num_strat_cats):
                    yield self.get_data(k, j)

        return _get_loaded_data()

    def get_data(self, strat_id, loc):

        return tuple(
            self.loaded_data[strat_id][i][
                self.loaded_data_order[strat_id][loc]
            ]
            for i in range(len(self.loaded_data[strat_id]))
        )
