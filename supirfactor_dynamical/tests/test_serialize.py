import unittest
import tempfile
import os

import numpy as np
import numpy.testing as npt

import torch
from scipy.linalg import pinv

from supirfactor_dynamical import (
    TFAutoencoder,
    TFRNNDecoder,
    read
)

from ._stubs import (
    X_tensor,
    A
)


class _ModelStub:

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def load_state_dict(self, x):
        self.state_dict = x


class TestSerializer(unittest.TestCase):

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix='pytest')
        self.temp_name = self.temp.name
        self.temp_file_name = os.path.join(self.temp_name, "static.h5")
        return super().setUp()

    def tearDown(self) -> None:
        self.temp.cleanup()
        return super().tearDown()

    def test_h5_static(self):

        ae = TFAutoencoder(A, use_prior_weights=True)
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
                'layer_dropout_rate': 0.0,
                'output_relu': True,
                'prediction_offset': None
            }
        )

    def test_h5_dynamic(self):

        ae = TFRNNDecoder(A, use_prior_weights=True)
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
                'layer_dropout_rate': 0.0,
                'output_relu': True,
                'prediction_offset': None,
            }
        )

    def test_h5_dynamic_notnone(self):

        ae = TFRNNDecoder(
            A,
            use_prior_weights=False,
            prediction_offset=1,
            output_relu=False
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
                'layer_dropout_rate': 0.0,
                'output_relu': False,
                'prediction_offset': 1
            }
        )

    def test_serialize_static(self):

        ae = TFAutoencoder(A, use_prior_weights=True)
        ae.decoder[0].weight = torch.nn.parameter.Parameter(
            torch.tensor(pinv(A).T, dtype=torch.float32)
        )

        ae.save(self.temp_file_name)
        ae.eval()

        loaded_ae = read(self.temp_file_name)
        loaded_ae.eval()

        with torch.no_grad():
            npt.assert_almost_equal(
                ae.encoder[0].weight_orig.numpy(),
                loaded_ae.encoder[0].weight_orig.numpy()
            )

            npt.assert_almost_equal(
                ae.encoder[0].weight_mask.numpy(),
                loaded_ae.encoder[0].weight_mask.numpy()
            )

            npt.assert_almost_equal(
                ae.decoder[0].weight.numpy(),
                loaded_ae.decoder[0].weight.numpy()
            )

            npt.assert_almost_equal(
                ae(X_tensor).numpy(),
                loaded_ae(X_tensor).numpy()
            )

    def test_serialize_dynamic(self):

        ae = TFAutoencoder(A, use_prior_weights=True)
        ae.decoder[0].weight = torch.nn.parameter.Parameter(
            torch.tensor(pinv(A).T, dtype=torch.float32)
        )

        ae.save(self.temp_file_name)
        ae.eval()

        loaded_ae = read(self.temp_file_name)
        loaded_ae.eval()

        with torch.no_grad():
            npt.assert_almost_equal(
                ae.encoder[0].weight_orig.numpy(),
                loaded_ae.encoder[0].weight_orig.numpy()
            )

            npt.assert_almost_equal(
                ae.encoder[0].weight_mask.numpy(),
                loaded_ae.encoder[0].weight_mask.numpy()
            )

            npt.assert_almost_equal(
                ae.decoder[0].weight.numpy(),
                loaded_ae.decoder[0].weight.numpy()
            )

            npt.assert_almost_equal(
                ae(X_tensor).numpy(),
                loaded_ae(X_tensor).numpy()
            )
