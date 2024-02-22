class _MultiModalDataMixin:

    _multimodal_data_model = True

    input_data_idx = 0
    output_data_idx = 1

    def input_data(self, x):
        return super().input_data(x[self.input_data_idx])

    def output_data(self, x):
        return super().output_data(x[self.output_data_idx])

    def set_input_output_idx(
        self,
        input_data_idx=None,
        output_data_idx=None
    ):
        if input_data_idx is not None:
            self.input_data_idx = input_data_idx
        if output_data_idx is not None:
            self.output_data_idx = output_data_idx
