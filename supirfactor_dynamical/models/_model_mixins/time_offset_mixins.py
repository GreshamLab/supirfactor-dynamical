from supirfactor_dynamical._utils import _check_data_offsets


class _TimeOffsetMixinRecurrent:

    def input_data(self, x):

        if self._offset_data:
            input_offset, _ = _check_data_offsets(
                x.shape[1],
                output_t_plus_one=self.output_t_plus_one,
                loss_offset=self.loss_offset,
                n_additional_predictions=self.n_additional_predictions
            )
            return x[:, 0:input_offset, :]

        else:
            return x


class _TimeOffsetMixinStatic:

    def input_data(self, x):

        if self._offset_data:
            return x[:, [0], ...]
        else:
            return x

    def output_data(
        self,
        x,
        output_t_plus_one=None,
        **kwargs
    ):

        if not self._offset_data:
            return x

        L = x.shape[1]
        max_L = 1

        if output_t_plus_one is None:
            output_t_plus_one = self.output_t_plus_one

        if output_t_plus_one:
            max_L += 1

        if self.n_additional_predictions is not None:
            max_L += self.n_additional_predictions

        if max_L == L:
            return super().output_data(
                x,
                output_t_plus_one=output_t_plus_one,
                **kwargs
            )
        elif max_L > L:
            raise ValueError(
                f"Cannot train on {L} sequence length with "
                f"{self.n_additional_predictions} additional predictions and "
                f"{self.loss_offset} values excluded from loss"
            )
        else:
            return super().output_data(
                x[:, 0:max_L, ...]
            )
