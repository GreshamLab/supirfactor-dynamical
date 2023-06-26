import torch


class _VelocityMixin:

    _velocity_model = True

    def input_data(self, x, **kwargs):

        return super().input_data(x[..., 0], **kwargs)

    def output_data(self, x, keep_all_dims=False, **kwargs):

        if keep_all_dims:
            return super().output_data(x, **kwargs)
        else:
            return super().output_data(x[..., 1], **kwargs)


class _DecayMixin:

    _decay_model = True

    decay_encoder = None
    decay_decoder = None
    decay_hidden = None

    decay_invert = None

    def initalize_decay_module(
        self,
        k
    ):

        self.decay_encoder = torch.nn.RNN(
            self.g,
            k
        )

        self.decay_decoder = torch.nn.RNN(
            k,
            self.g,
            nonlinearity='ReLU'
        )

        self.decay_hidden = [None, None]

        self.decay_invert = torch.diag(
            torch.full(
                self.g,
                -1.0
            )
        )

    def forward_decay_model(
        self,
        x,
        x_count,
        decay_hidden_states=None
    ):

        if decay_hidden_states is None:
            decay_hidden_states = [None, None]

        x, self.decay_hidden[0] = self.decay_encoder(
            x, decay_hidden_states[0]
        )

        x = self.hidden_dropout(x)
        x, self.decay_hidden[1] = self.decay_decoder(
            x, decay_hidden_states[1]
        )

        x = torch.mul(x, x_count)
        x = torch.matmul(x, self.decay_invert)

        return x

    def forward_model(
        self,
        x,
        hidden_state=None
    ):

        if hidden_state is None:
            hidden_state = [None, None, None]

        x_tf = self.drop_encoder(x)
        x_tf = self.hidden_dropout(x_tf)
        x_tf = self.decoder(x_tf, hidden_state[0], intermediate_only=True)

        x_output_positive = self._decoder(x_tf)

        x_output_negative = self.forward_decay_model(
            x_tf,
            x,
            [hidden_state[1], hidden_state[2]]
        )

        x_output = torch.add(
            x_output_positive,
            x_output_negative
        )

        self.hidden_final = [
            self.hidden_final,
            self.decay_hidden[0],
            self.decay_hidden[1]
        ]

        return x_output
