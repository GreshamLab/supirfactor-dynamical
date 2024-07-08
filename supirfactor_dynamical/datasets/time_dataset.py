import torch
import numpy as np
import warnings
import collections
import math

from scipy.sparse import issparse


class _TimeDataMixin:

    n = 0
    n_steps = None
    with_replacement = True
    return_times = False

    rng = None
    data = None

    # Time stratification properties
    time_vector = None
    shuffle_time_limits = None
    _base_time_vector = None

    t_min = None
    t_max = None
    t_step = None
    wrap_times = False

    strat_idxes = None
    shuffle_idxes = None

    # Sequence length properties
    _sequence_length = None
    _sequence_length_options = None

    @property
    def sequence_length(self):
        return self._sequence_length

    @sequence_length.setter
    def sequence_length(self, x):

        if x is None:
            self._sequence_length = None

        elif isinstance(
            x,
            (tuple, list, np.ndarray)
        ):
            self._sequence_length_options = x
            self.shuffle_sequence_length()

        else:

            self._sequence_length = x

            if x is not None:
                self.n = min(map(len, self.strat_idxes))
                self.n *= int(max(np.floor(self.n_steps / x), 1))

    def __init__(
        self,
        data_reference,
        time_vector,
        t_min,
        t_max,
        t_step=None,
        sequence_length=None,
        random_seed=500,
        with_replacement=True,
        shuffle_time_vector=None,
        wrap_times=False,
        return_times=False
    ):

        self.time_vector = time_vector
        self.t_min = t_min
        self.t_max = t_max
        self.t_step = t_step

        self.data = self._to_tensor(data_reference)
        self._intial_time_processing()

        self.rng = np.random.default_rng(random_seed)
        self.return_times = return_times
        self.shuffle_time_limits = shuffle_time_vector
        self.with_replacement = with_replacement
        self.wrap_times = wrap_times

        self._initial_index_processing(sequence_length)

    def _intial_time_processing(
        self
    ):

        _data_keep_idx = np.logical_and(
            self.t_min <= self.time_vector,
            self.time_vector < self.t_max
        )

        if not np.all(_data_keep_idx):
            self.data = self._get_data(
                self.data,
                _data_keep_idx,
                keep_sparse=True
            )
            self.time_vector = self.time_vector[_data_keep_idx]

        # Make sure it's not a pandas series
        try:
            self.time_vector = self.time_vector.values
        except AttributeError:
            pass

    def _initial_index_processing(self, sequence_length):

        if self.t_step is not None:

            # Create a list of arrays, where each element
            # is an array of indices to observations for that
            # time window
            self.strat_idxes = self._generate_stratified_indices()

            self.n = min(map(len, self.strat_idxes))
            self.n_steps = len(self.strat_idxes)
            self.sequence_length = sequence_length
            self.shuffle()

        else:

            self.n = self._n_samples(self.data)
            self.strat_idxes = None

        if self.n == 0:
            warnings.warn(
                f"Data with {self.data.shape[0]} observations will "
                f"yield zero data sequences",
                UserWarning
            )

    def _generate_stratified_indices(self):

        if self.t_min is None or self.t_max is None or self.t_step is None:
            raise ValueError(
                f"Cannot create indices from t_min {self.t_min} "
                f"t_max {self.t_max} and t_step {self.t_step}"
            )

        return [
            np.where(
                np.logical_and(
                    i <= self.time_vector,
                    self.time_vector < (i + self.t_step)
                )
            )[0]
            for i in range(self.t_min, self.t_max, self.t_step)
        ]

    def shuffle(self):
        """
        Shuffle data for another epoch
        """

        if self.strat_idxes is not None:

            self.shuffle_sequence_length()
            self.shuffle_time_vector()

            self.shuffle_idxes = self._get_shuffle_indexes(
                with_replacement=self.with_replacement,
                n=self.n
            )

        else:
            self.shuffle_idxes = None

    def shuffle_sequence_length(self):
        """
        If multple options for sequence length exist
        select one at random
        """

        if self._sequence_length_options is not None:

            self.sequence_length = self.rng.choice(
                self._sequence_length_options,
                size=1
            )[0]

        else:
            pass

    def shuffle_time_vector(self):
        """
        Shuffle the time labels on values between a start and
        a stop time
        """

        if self.shuffle_time_limits is None:
            return

        if self._base_time_vector is None:
            self._base_time_vector = self.time_vector.copy()

        self.time_vector = self._base_time_vector.copy()

        shuffle_idx = self._base_time_vector >= self.shuffle_time_limits[0]
        shuffle_idx &= self._base_time_vector < self.shuffle_time_limits[1]

        shuffle_data = self.time_vector[shuffle_idx]
        self.rng.shuffle(shuffle_data)
        self.time_vector[shuffle_idx] = shuffle_data

    def _get_shuffle_indexes(
        self,
        with_replacement=True,
        n=None,
        seq_length=None
    ):

        # Sequence length L
        if seq_length is None and self.sequence_length is not None:
            seq_length = self.sequence_length
        elif seq_length is None:
            seq_length = len(self.strat_idxes)

        # Number of individual data sequences N
        if n is None and not with_replacement:
            n = min(map(len, self.strat_idxes))
        elif n is None:
            n = int(np.median(list(map(len, self.strat_idxes))))

        # For each time interval, reshuffle by random selection
        # to length of n
        _idxes = np.ascontiguousarray(
            np.array([
                self.rng.choice(
                    x,
                    size=n,
                    replace=with_replacement
                )
                for x in self.strat_idxes
            ]).T
        )

        if self.wrap_times:
            return np.array(
                [
                    self._get_wrap_indices(
                        seq_length,
                        _idxes[i],
                        n_wraps=math.ceil(seq_length / len(self.strat_idxes))
                    )
                    for i in range(n)
                ]
            )

        else:
            return np.array(
                [
                    self._get_no_wrap_indices(
                        seq_length,
                        _idxes[i]
                    )
                    for i in range(n)
                ]
            )

    def _get_no_wrap_indices(
        self,
        seq_length,
        indexes
    ):

        # If L is shorter than the total number of time intervals,
        # randomly select a starting time on data sequence
        # to get L observations from the shuffled indices
        if seq_length < len(indexes):

            start_position = self.rng.integers(
                0,
                len(self.strat_idxes) - seq_length + 1,
            )

            return indexes[start_position:start_position + seq_length]

        elif seq_length == len(indexes):
            return indexes

        else:
            raise ValueError(
                f"Cannot make sequence of length {seq_length} "
                f"from {len(self.strat_idxes)} bins"
            )

    def _get_wrap_indices(
        self,
        seq_length,
        indexes,
        n_wraps=1
    ):

        indexes = collections.deque(indexes)
        indexes.rotate(self.rng.integers(
            0,
            len(self.strat_idxes),
        ))
        indexes = np.array(indexes * n_wraps)

        return indexes[:seq_length]

    def get_data_time(
        self,
        t_start=None,
        t_stop=None
    ):

        if t_start is None:
            t_start = np.min(self.time_vector)

        if t_stop is None:
            t_stop = np.max(self.time_vector) + 1

        _data_idx = np.logical_and(
            t_start <= self.time_vector,
            self.time_vector < t_stop
        )

        return self._get_data(self.data, _data_idx)

    def get_times_in_order(
        self
    ):

        all_time_indexes = self._get_shuffle_indexes(
            seq_length=len(self.strat_idxes),
            with_replacement=True
        )

        if self._sequence_length_options is not None:
            seq_length = min(self._sequence_length_options)
        elif self.sequence_length is not None:
            seq_length = self.sequence_length
        else:
            seq_length = len(self.strat_idxes)

        n_steps = len(self.strat_idxes) - seq_length + 1

        def _timeiterator():
            for i in range(n_steps):
                yield torch.stack([
                    self._get_data(self.data, a[i:i + seq_length])
                    for a in all_time_indexes
                ])

        return _timeiterator()

    def get_aggregated_times(
        self
    ):

        _aggregate_data = torch.stack([
            self._get_data(self.data, self.strat_idxes[i]).mean(axis=0)
            for i in range(self.n_steps)
        ])

        if self.sequence_length is not None:
            return [
                _aggregate_data[i:i + self.sequence_length, :]
                for i in range(self.n_steps - self.sequence_length + 1)
            ]
        else:
            return _aggregate_data

    @staticmethod
    def _get_data(data, idx, keep_sparse=False):

        _data = data[idx, :]

        if issparse(_data) and not keep_sparse:

            _data = _data.toarray()
            _data = torch.Tensor(
                _data.reshape(-1) if _data.shape[0] == 1 else _data
            )

        return _data

    @staticmethod
    def _to_tensor(data):

        if torch.is_tensor(data):
            return data

        elif not issparse(data):
            return torch.Tensor(data)

        else:
            return data

    @staticmethod
    def _n_samples(data):

        return data.shape[0]


class TimeDataset(
    _TimeDataMixin,
    torch.utils.data.Dataset
):

    def __getitem__(self, i):

        if self.strat_idxes is not None:
            _data_index = self.shuffle_idxes[i]

        else:
            _data_index = i

        if self.return_times:
            return (
                self._get_data(self.data, _data_index),
                torch.Tensor(self.time_vector[_data_index])
            )
        else:
            return self._get_data(self.data, _data_index)

    def __len__(self):
        return self.n


class TimeDatasetIter(
    _TimeDataMixin,
    torch.utils.data.IterableDataset
):

    """
    TimeDataset for multiple workers
    """

    def __iter__(self):

        worker_info = torch.utils.data.get_worker_info()

        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
        else:
            worker_id = 0
            num_workers = 1

        def _time_generator():

            for i in np.arange(self.n)[worker_id::num_workers]:
                if self.strat_idxes is not None:
                    _data_index = self.shuffle_idxes[i]

                else:
                    _data_index = i

                if self.return_times:
                    yield (
                        self._get_data(self.data, _data_index),
                        torch.Tensor(self.time_vector[_data_index])
                    )
                else:
                    yield self._get_data(self.data, _data_index)

            self.shuffle()

        return _time_generator()
