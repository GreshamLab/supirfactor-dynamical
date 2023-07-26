import unittest
import tempfile
import os

import numpy.testing as npt

import torch
from scipy.linalg import pinv

from supirfactor_dynamical import (
    read,
    get_model
)

from ._stubs import (
    X_tensor,
    XTV_tensor,
    A
)


class _ModelStub:

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def load_state_dict(self, x):
        self.state_dict = x

    def set_time_parameters(self, **kwargs):
        self.time_kwargs = kwargs

    def set_scaling(self, **kwargs):
        self.scaler_kwargs = kwargs


class _SetupMixin:

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix='pytest')
        self.temp_name = self.temp.name
        self.temp_file_name = os.path.join(self.temp_name, "static.h5")
        return super().setUp()

    def tearDown(self) -> None:
        self.temp.cleanup()
        return super().tearDown()

    def _compare_module(self, module1, module2):

        module1_state = module1.state_dict()
        module2_state = module2.state_dict()

        with torch.no_grad():
            for k, v in module1_state.items():
                npt.assert_almost_equal(
                    v.numpy(),
                    module2_state[k].numpy()
                )


class TestSerializer(_SetupMixin, unittest.TestCase):

    velocity = False

    def test_h5_static(self):

        ae = get_model(
            'static',
            velocity=self.velocity
        )(A, use_prior_weights=True)

        ae.decoder[0].weight = torch.nn.parameter.Parameter(
            torch.tensor(pinv(A).T, dtype=torch.float32)
        )

        ae.save(self.temp_file_name)

        stub = read(
            self.temp_file_name,
            model_class=_ModelStub
        )

        npt.assert_almost_equal(
            A,
            stub.args[0].values
        )

        self.assertDictEqual(
            stub.kwargs,
            {
                'input_dropout_rate': 0.5,
                'hidden_dropout_rate': 0.0,
                'output_relu': True
            }
        )

        self.assertDictEqual(
            stub.time_kwargs,
            {
                'output_t_plus_one': False,
                'n_additional_predictions': 0,
                'loss_offset': 0
            }
        )

        self.assertDictEqual(
            stub.scaler_kwargs,
            {
                'count_scaling': None,
                'velocity_scaling': None
            }
        )

    def test_h5_dynamic(self):

        ae = get_model(
            'rnn',
            velocity=self.velocity
        )(A, use_prior_weights=True)
        ae._decoder[0].weight = torch.nn.parameter.Parameter(
            torch.tensor(pinv(A).T, dtype=torch.float32)
        )

        ae.save(self.temp_file_name)

        stub = read(
            self.temp_file_name,
            model_class=_ModelStub
        )

        npt.assert_almost_equal(
            A,
            stub.args[0].values
        )

        self.assertDictEqual(
            stub.kwargs,
            {
                'input_dropout_rate': 0.5,
                'hidden_dropout_rate': 0.0,
                'output_relu': True
            }
        )

        self.assertDictEqual(
            stub.time_kwargs,
            {
                'output_t_plus_one': False,
                'n_additional_predictions': 0,
                'loss_offset': 0
            }
        )

    def test_h5_dynamic_notnone(self):

        ae = get_model(
            'rnn',
            velocity=self.velocity
        )(
            A,
            use_prior_weights=False,
            output_relu=False
        )

        ae.set_time_parameters(
            output_t_plus_one=True,
            n_additional_predictions=1,
            loss_offset=0
        )

        ae._decoder[0].weight = torch.nn.parameter.Parameter(
            torch.tensor(pinv(A).T, dtype=torch.float32)
        )

        ae.save(self.temp_file_name)

        stub = read(
            self.temp_file_name,
            model_class=_ModelStub
        )

        npt.assert_almost_equal(
            A,
            stub.args[0].values
        )

        self.assertDictEqual(
            stub.kwargs,
            {
                'input_dropout_rate': 0.5,
                'hidden_dropout_rate': 0.0,
                'output_relu': False
            }
        )

        self.assertDictEqual(
            stub.time_kwargs,
            {
                'output_t_plus_one': True,
                'n_additional_predictions': 1,
                'loss_offset': 0
            }
        )

    def test_serialize_static(self):

        ae = get_model(
            'static',
            velocity=self.velocity
        )(A, use_prior_weights=True)
        ae.decoder[0].weight = torch.nn.parameter.Parameter(
            torch.tensor(pinv(A).T, dtype=torch.float32)
        )

        ae.save(self.temp_file_name)
        ae.eval()

        loaded_ae = read(self.temp_file_name)
        loaded_ae.eval()

        self._compare_module(
            ae.encoder,
            loaded_ae.encoder
        )

        self._compare_module(
            ae.decoder,
            loaded_ae.decoder
        )

        with torch.no_grad():
            npt.assert_almost_equal(
                ae(X_tensor).numpy(),
                loaded_ae(X_tensor).numpy()
            )

    def test_serialize_scaling(self):

        ae = get_model(
            'rnn',
            velocity=self.velocity
        )(A, use_prior_weights=True)

        ae.set_scaling(
            torch.ones(4),
            torch.ones(4)
        )

        ae.save(self.temp_file_name)
        ae.eval()

        loaded_ae = read(self.temp_file_name)
        loaded_ae.eval()

        torch.testing.assert_close(
            ae.count_scaler,
            loaded_ae.count_scaler
        )

        torch.testing.assert_close(
            torch.eye(4),
            loaded_ae.count_scaler
        )

        torch.testing.assert_close(
            ae.velocity_scaler,
            loaded_ae.velocity_scaler
        )

        torch.testing.assert_close(
            torch.eye(4),
            loaded_ae.velocity_scaler
        )

        torch.testing.assert_close(
            ae.count_to_velocity_scaler,
            loaded_ae.count_to_velocity_scaler
        )

        torch.testing.assert_close(
            torch.eye(4),
            loaded_ae.count_to_velocity_scaler
        )

        torch.testing.assert_close(
            ae.velocity_to_count_scaler,
            loaded_ae.velocity_to_count_scaler
        )

        torch.testing.assert_close(
            torch.eye(4),
            loaded_ae.velocity_to_count_scaler
        )

    def test_serialize_dynamic(self):

        ae = get_model(
            'rnn',
            velocity=self.velocity
        )(A, use_prior_weights=True)

        ae._decoder[0].weight = torch.nn.parameter.Parameter(
            torch.tensor(pinv(A).T, dtype=torch.float32)
        )

        ae.save(self.temp_file_name)
        ae.eval()

        loaded_ae = read(self.temp_file_name)
        loaded_ae.eval()

        self._compare_module(
            ae.encoder,
            loaded_ae.encoder
        )

        self._compare_module(
            ae._decoder,
            loaded_ae._decoder
        )

        self._compare_module(
            ae._intermediate,
            loaded_ae._intermediate
        )

        with torch.no_grad():
            npt.assert_almost_equal(
                ae(X_tensor).numpy(),
                loaded_ae(X_tensor).numpy()
            )


class TestBiophysical(_SetupMixin, unittest.TestCase):

    def test_serialize_biophysical(self):

        ae = get_model('static_meta')(A, use_prior_weights=True)

        ae._decoder[0].weight = torch.nn.parameter.Parameter(
            torch.tensor(pinv(A).T, dtype=torch.float32)
        )

        biophysical = get_model('biophysical')(
            A,
            trained_count_model=ae
        )

        biophysical.save(self.temp_file_name)
        biophysical.eval()

        loaded_biophysical = read(self.temp_file_name)
        loaded_biophysical.eval()

        self._compare_module(
            biophysical._count_model,
            loaded_biophysical._count_model
        )

        self._compare_module(
            biophysical._transcription_model,
            loaded_biophysical._transcription_model
        )

        self._compare_module(
            biophysical._decay_model,
            loaded_biophysical._decay_model
        )

        with torch.no_grad():
            npt.assert_almost_equal(
                biophysical(
                    biophysical.input_data(XTV_tensor)
                ).numpy(),
                loaded_biophysical(
                    loaded_biophysical.input_data(XTV_tensor)
                ).numpy()
            )

    def test_serialize_biophysical_nocount(self):

        biophysical = get_model('biophysical')(
            A,
        )

        biophysical.save(self.temp_file_name)
        biophysical.eval()

        loaded_biophysical = read(self.temp_file_name)
        loaded_biophysical.eval()

        self.assertIsNone(biophysical._count_model)
        self.assertIsNone(loaded_biophysical._count_model)

        self._compare_module(
            biophysical._transcription_model,
            loaded_biophysical._transcription_model
        )

        self._compare_module(
            biophysical._decay_model,
            loaded_biophysical._decay_model
        )

        with torch.no_grad():
            npt.assert_almost_equal(
                biophysical(
                    biophysical.input_data(XTV_tensor)
                ).numpy(),
                loaded_biophysical(
                    loaded_biophysical.input_data(XTV_tensor)
                ).numpy()
            )

    def test_serialize_biophysical_nodecay(self):

        biophysical = get_model('biophysical')(
            A,
            decay_model=False
        )

        biophysical.save(self.temp_file_name)
        biophysical.eval()

        loaded_biophysical = read(self.temp_file_name)
        loaded_biophysical.eval()

        self.assertIsNone(biophysical._count_model)
        self.assertIsNone(loaded_biophysical._count_model)

        self._compare_module(
            biophysical._transcription_model,
            loaded_biophysical._transcription_model
        )

        self.assertIsNone(biophysical._decay_model)
        self.assertIsNone(loaded_biophysical._decay_model)

        with torch.no_grad():
            npt.assert_almost_equal(
                biophysical(
                    biophysical.input_data(XTV_tensor)
                ).numpy(),
                loaded_biophysical(
                    loaded_biophysical.input_data(XTV_tensor)
                ).numpy()
            )

    def test_serialize_decay_module(self):

        decay = get_model('decay')(
            3
        )

        decay.save(self.temp_file_name)
        decay.eval()

        loaded_decay = read(self.temp_file_name)
        loaded_decay.eval()

        self._compare_module(
            decay,
            loaded_decay
        )


class TestSerializerVelocity(TestSerializer):

    velocity = True
