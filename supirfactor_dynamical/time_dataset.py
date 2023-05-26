import torch
import numpy as np

from scipy.sparse import isspmatrix


class TimeDataset(torch.utils.data.Dataset):

    n = 0
    n_steps = None
    with_replacement = True

    rng = None
    data = None

    idxes = None
    shuffle_idxes = None

    def __init__(
        self,
        data_reference,
        time_vector,
        t_min,
        t_max,
        t_step=None,
        sequence_length=None,
        random_seed=500,
        with_replacement=True
    ):

        # Only keep data that's usable
        # erase the rest
        _data_keep_idx = np.logical_and(
            t_min <= time_vector,
            time_vector < t_max
        )

        if not np.all(_data_keep_idx):
            self.data = data_reference[_data_keep_idx, :]
            self.time_vector = time_vector[_data_keep_idx]
        else:
            self.data = data_reference
            self.time_vector = time_vector

        if not torch.is_tensor(self.data) and not isspmatrix(self.data):
            self.data = torch.tensor(
                self.data,
                dtype=torch.float32
            )

        self.with_replacement = with_replacement
        self.rng = np.random.default_rng(random_seed)

        if t_step is not None:

            # Create a list of arrays, where each element
            # is an array of indices to observations for that
            # time window
            self.strat_idxes = [
                np.where(
                    np.logical_and(
                        i <= self.time_vector,
                        self.time_vector < (i + t_step)
                    )
                )[0]
                for i in range(t_min, t_max, t_step)
            ]

            self.n = min(map(len, self.strat_idxes))
            self.n_steps = len(self.strat_idxes)
            self.sequence_length = sequence_length

            if sequence_length is not None:
                self.n = self.n * int(np.floor(self.n_steps / sequence_length))

            self.shuffle()

        else:

            self.n = self.data.shape[0]
            self.strat_idxes = None

    def shuffle(self):

        if self.strat_idxes is not None:

            self.shuffle_idxes = self._get_shuffle_indexes(
                with_replacement=self.with_replacement,
                n=self.n
            )

        else:
            self.shuffle_idxes = None

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
        _idxes = [
            self.rng.choice(
                x,
                size=n,
                replace=with_replacement
            )
            for x in self.strat_idxes
        ]

        # If L is shorter than the total number of time intervals,
        # randomly select a starting time on data sequence
        # to get L observations from the shuffled indices
        if seq_length < len(self.strat_idxes):

            start_position = np.arange(
                len(self.strat_idxes) - seq_length + 1,
            )

            def _get_sequence():
                start = self.rng.choice(start_position)
                return slice(start, start + seq_length)

        else:
            def _get_sequence():
                return slice(None)

        return [
            np.array([
                x[i] for x in _idxes[_get_sequence()]
            ]) for i in range(n)
        ]

    def __getitem__(self, i):

        if self.strat_idxes is not None:
            _data_index = self.shuffle_idxes[i]

        else:
            _data_index = i

        return self._get_data(_data_index)

    def __len__(self):
        return self.n

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

        return self._get_data(_data_idx)

    def get_times_in_order(
        self
    ):

        all_time_indexes = self._get_shuffle_indexes(
            seq_length=len(self.strat_idxes),
            with_replacement=True
        )

        if self.sequence_length is not None:
            seq_length = self.sequence_length
        else:
            seq_length = len(self.strat_idxes)

        n_steps = len(self.strat_idxes) - seq_length + 1

        def _timeiterator():
            for i in range(n_steps):
                yield torch.stack([
                    self._get_data(a[i:i + seq_length])
                    for a in all_time_indexes
                ])

        return _timeiterator()

    def get_aggregated_times(
        self
    ):

        _aggregate_data = torch.stack([
            self._get_data(self.strat_idxes[i]).mean(axis=0)
            for i in range(self.n_steps)
        ])

        if self.sequence_length is not None:
            return [
                _aggregate_data[i:i + self.sequence_length, :]
                for i in range(self.n_steps - self.sequence_length + 1)
            ]
        else:
            return _aggregate_data

    def _get_data(
        self,
        idx
    ):

        data = self.data[idx, :]

        if isspmatrix(data):
            data = data.A,

        if not torch.is_tensor(data):
            data = torch.tensor(
                data,
                dtype=torch.float32
            )

        return data
