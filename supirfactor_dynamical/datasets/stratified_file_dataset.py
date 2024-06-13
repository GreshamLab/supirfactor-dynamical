import h5py
import warnings
import torch
import numpy as np
import scipy.sparse as sps
import gc

from anndata._io.h5ad import read_dataframe


class _H5ADFileLoader:

    file_data_layer = 'X'

    yield_extra_layers = None
    yield_obs_cats = None

    one_hot_obs_cats = True
    obs_categories = None

    def __init__(
        self,
        file_data_layer='X',
        yield_extra_layers=None,
        yield_obs_cats=None,
        obs_categories=None,
        one_hot_obs_cats=True
    ):
        self.file_data_layer = file_data_layer
        self.yield_extra_layers = yield_extra_layers

        if not isinstance(yield_obs_cats, (tuple, list)):
            yield_obs_cats = [yield_obs_cats]

        self.yield_obs_cats = yield_obs_cats
        self.one_hot_obs_cats = one_hot_obs_cats
        self.obs_categories = obs_categories

    @staticmethod
    def load_file(
        file_name,
        layer,
        obs_include_mask=None,
        extra_layers=None,
        append_obs=True
    ):

        with h5py.File(file_name) as file_handle:

            if layer == 'obs':
                return _H5ADFileLoader.load_layer(
                    file_handle,
                    layer,
                    obs_include_mask
                )

            _data = [_H5ADFileLoader.load_layer(
                file_handle,
                layer,
                obs_include_mask
            )]

            if extra_layers is not None:
                _data = _data + [
                    _H5ADFileLoader.load_layer(
                        file_handle,
                        _elayer,
                        obs_include_mask
                    )
                    for _elayer in extra_layers
                ]

            if append_obs:
                _data.append(
                    _H5ADFileLoader.load_layer(
                        file_handle,
                        'obs',
                        obs_include_mask
                    )
                )

            return _data

    @staticmethod
    def load_layer(file_handle, layer, obs_include_mask=None):

        # Get OBS dataframe
        if layer == 'obs':
            df = read_dataframe(file_handle['obs'])

            if obs_include_mask is not None:
                df = df.iloc[
                    np.arange(df.shape[0])[obs_include_mask],
                    :
                ]

            return df

        # Check for X layer
        elif layer == 'X':
            _data_reference = file_handle['X']

        # Check for layer key in .layers
        elif layer in file_handle['layers'].keys():
            _data_reference = file_handle['layers'][layer]

        # Check for layer key in .obsm
        elif layer in file_handle['obsm'].keys():
            _data_reference = file_handle['obsm'][layer]

        # Couldn't find anything
        else:
            raise ValueError(
                f"Cannot find {layer} in `layers` or `obsm`"
            )

        if sparse_type := _H5ADFileLoader._issparse(_data_reference):
            return _H5ADFileLoader._load_sparse(
                _data_reference,
                sparse_type,
                obs_include_mask,
            )

        else:
            return _H5ADFileLoader._load_dense(
                _data_reference,
                obs_include_mask
            )

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
            return 'csr'
        elif _encoding == 'csc_matrix':
            return 'csc'
        else:
            raise ValueError(f"{_encoding} unknown")

    @staticmethod
    def _load_sparse(ref, sparse_type='csr', obs_mask=None):

        if sparse_type == 'csr':
            _sp_func = sps.csr_array
            _torch_func = torch.sparse_csr_tensor
        elif sparse_type == 'csc':
            _sp_func = sps.csc_array
            _torch_func = torch.sparse_csc_tensor
        else:
            raise ValueError(
                f"Sparse type {sparse_type} unknown"
            )

        if obs_mask is not None:

            # Create and slice a sparse python object
            # because torch doesn't support slicing
            arr = _sp_func(
                (
                    ref['data'][:],
                    ref['indices'][:],
                    ref['indptr'][:]
                ),
                shape=_H5ADFileLoader._get_shape(ref)
            )[obs_mask, :]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                arr = _torch_func(
                    torch.tensor(
                        arr.indptr,
                        dtype=torch.int64
                    ),
                    torch.tensor(
                        arr.indices,
                        dtype=torch.int64
                    ),
                    torch.Tensor(
                        arr.data
                    ),
                    size=arr.shape
                )

        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                arr = _torch_func(
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
                    size=_H5ADFileLoader._get_shape(ref)
                )

        gc.collect()
        return arr.to_dense()

    @staticmethod
    def _load_dense(ref, obs_mask=None):
        if obs_mask is None:
            return torch.Tensor(ref[:])
        else:
            return torch.Tensor(ref[obs_mask, ...])

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

    @staticmethod
    def get_stratification_indices(
        obs,
        stratification_columns,
        discard_categories=None
    ):

        obs = obs[stratification_columns].copy()
        obs['row_idx_loc'] = np.arange(obs.shape[0])

        indices = []

        for vals, idxes in obs.groupby(
            stratification_columns,
            observed=False
        )['row_idx_loc']:

            if discard_categories is None:
                pass
            elif (
                isinstance(vals, tuple) and
                any(v in discard_categories for v in vals)
            ):
                continue
            elif vals in discard_categories:
                continue

            if len(idxes) > 0:
                indices.append(
                    idxes.values
                )

        return indices


class StratifySingleFileDataset(
    _H5ADFileLoader,
    torch.utils.data.IterDataPipe
):

    loaded_data = None
    yields_tuple = None

    stratification_group_indexes = None
    min_strat_size = None
    n_strat_groups = None

    def __init__(
        self,
        file_name,
        stratification_columns,
        obs_include_mask=None,
        discard_categories=None,
        random_state=None,
        file_data_layer='X',
        yield_extra_layers=None,
        yield_obs_cats=None,
        obs_categories=None,
        one_hot_obs_cats=True
    ):

        self.loaded_data = self.load_file(
            file_name,
            layer=file_data_layer,
            extra_layers=yield_extra_layers,
            append_obs=True,
            obs_include_mask=obs_include_mask
        )
        self.yields_tuple = len(self.loaded_data) > 1

        obs = self.loaded_data.pop(-1)

        if yield_obs_cats is not None:
            self.loaded_data.extend(
                self._get_obs_cats(
                    obs,
                    yield_obs_cats,
                    obs_categories,
                    one_hot=one_hot_obs_cats
                )
            )

        self.stratification_group_indexes = self.get_stratification_indices(
            obs,
            stratification_columns,
            discard_categories=discard_categories
        )
        self.min_strat_size = min(
            len(x)
            for x in self.stratification_group_indexes
            if len(x) > 0
        )
        self.n_strat_groups = len(self.stratification_group_indexes)

        self.rng = np.random.default_rng(random_state)
        self.shuffle()

    def __iter__(self):

        worker_info = torch.utils.data.get_worker_info()

        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
        else:
            worker_id = 0
            num_workers = 1

        def _get_loaded_data():
            for j in np.arange(self.min_strat_size)[worker_id::num_workers]:
                for k in range(self.n_strat_groups):
                    yield self.get_data(
                        self.stratification_group_indexes[k][j]
                    )

            self.shuffle()

        return _get_loaded_data()

    def get_data(self, loc):

        if self.yields_tuple:
            return tuple(
                self.loaded_data[i][loc, ...]
                for i in range(len(self.loaded_data))
            )
        else:
            return self.loaded_data[0][loc, ...]

    def shuffle(self):

        for i in self.stratification_group_indexes:
            self.rng.shuffle(i)


class StratifiedFilesDataset(
    _H5ADFileLoader,
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

        super().__init__(
            file_data_layer=file_data_layer,
            yield_extra_layers=yield_extra_layers,
            yield_obs_cats=yield_obs_cats,
            obs_categories=obs_categories,
            one_hot_obs_cats=one_hot_obs_cats
        )

        self.file_metadata = file_data.copy()
        self.file_name_column = file_name_column

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
