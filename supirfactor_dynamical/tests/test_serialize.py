import unittest
import tempfile
import os
import h5py

import numpy as np
import numpy.testing as npt
import pandas.testing as pdt
import anndata as ad
import pandas as pd

import torch
from scipy.linalg import pinv
from scipy.sparse import csr_matrix

from supirfactor_dynamical import (
    read,
    get_model
)

from supirfactor_dynamical._utils import to

from supirfactor_dynamical._io._network import (
    _read_index,
    _write_index,
    _read_df,
    _write_df,
    _read_ad,
    _write_ad
)

from ._stubs import (
    X_tensor,
    XTV_tensor,
    A,
    G_TO_PEAK_PRIOR,
    PEAK_TO_TF_PRIOR
)


class _ModelStub:

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

        self.prior_network = kwargs.pop('prior_network', None)

    def load_state_dict(self, x):
        self.state_dict = x

    def set_time_parameters(self, **kwargs):
        self.time_kwargs = kwargs

    def set_scaling(self, **kwargs):
        self.scaler_kwargs = kwargs


class _SetupMixin:

    device = 'cpu'

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

        for k, v in module1_state.items():
            torch.testing.assert_close(
                v,
                module2_state[k]
            )


class TestSerializeHelpers(_SetupMixin, unittest.TestCase):

    def test_index_strings(self):

        idx = pd.Index([1, 2, 3, 4, 5]).astype(str)

        with h5py.File(self.temp_file_name, "w") as f:

            _write_index(
                f,
                idx,
                "index"
            )

        with h5py.File(self.temp_file_name, "r") as f:

            idx2 = _read_index(
                f,
                "index"
            )

        pdt.assert_index_equal(idx, idx2)

    def test_index_ints(self):

        idx = pd.Index([1, 2, 3, 4, 5])

        with h5py.File(self.temp_file_name, "w") as f:

            _write_index(
                f,
                idx,
                "index"
            )

        with h5py.File(self.temp_file_name, "r") as f:

            idx2 = _read_index(
                f,
                "index"
            )

        pdt.assert_index_equal(idx.astype(str), idx2)

    def test_df_strings(self):

        df = pd.DataFrame([["A", 1], ["B", 2]])

        _write_df(self.temp_file_name, df, "df")

        df2 = _read_df(self.temp_file_name, "df")

        pdt.assert_frame_equal(df, df2)

    def test_df_strings_indexes(self):

        df = pd.DataFrame([["A", 1], ["B", 2]])
        df.index = df.index.astype(str)
        df.columns = df.columns.astype(str)

        _write_df(self.temp_file_name, df, "df")

        df2 = _read_df(self.temp_file_name, "df")

        pdt.assert_frame_equal(df, df2)

    def test_h5ad_dense(self):

        adata = ad.AnnData(np.random.rand(5, 2))

        _write_ad(self.temp_file_name, adata, "adata")

        adata2 = _read_ad(self.temp_file_name, "adata")

        npt.assert_almost_equal(adata.X, adata2.X)
        pdt.assert_index_equal(adata.obs_names, adata2.obs_names)
        pdt.assert_index_equal(adata.var_names, adata2.var_names)

    def test_h5ad_sparse(self):

        adata = ad.AnnData(csr_matrix(np.random.rand(5, 2)))

        _write_ad(self.temp_file_name, adata, "adata")

        adata2 = _read_ad(self.temp_file_name, "adata")

        npt.assert_almost_equal(adata.X.A, adata2.X.A)
        pdt.assert_index_equal(adata.obs_names, adata2.obs_names)
        pdt.assert_index_equal(adata.var_names, adata2.var_names)


class TestSerializer(_SetupMixin, unittest.TestCase):

    velocity = False
    prior = A
    inv_prior = pinv(A).T

    def test_h5_static(self):

        ae = get_model(
            'static',
            velocity=self.velocity
        )(self.prior, use_prior_weights=True)

        ae._decoder[0].weight = torch.nn.parameter.Parameter(
            torch.tensor(self.inv_prior, dtype=torch.float32)
        )

        to(ae, self.device)
        ae.save(self.temp_file_name)
        to(ae, 'cpu')

        stub = read(
            self.temp_file_name,
            model_class=_ModelStub
        )

        npt.assert_almost_equal(
            A,
            stub.prior_network.values
        )

        self.assertDictEqual(
            stub.kwargs,
            {
                'input_dropout_rate': 0.5,
                'hidden_dropout_rate': 0.0,
                'output_activation': 'relu',
                'activation': 'relu'
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

    def test_h5_no_prior(self):

        ae = get_model(
            'static',
            velocity=self.velocity
        )((4, 3))

        to(ae, self.device)
        ae.save(self.temp_file_name)
        to(ae, 'cpu')

        stub = read(
            self.temp_file_name,
            model_class=_ModelStub
        )

        self.assertEqual(stub.prior_network, (4, 3))

    def test_h5_train_checkpoint(self):

        ae = get_model(
            'static',
            velocity=self.velocity
        )(self.prior, use_prior_weights=True)

        ae.train_model([X_tensor], 1)
        self.assertEqual(ae.current_epoch, 0)

        to(ae, self.device)
        ae.save(self.temp_file_name)
        to(ae, 'cpu')

        loaded = read(self.temp_file_name)
        self.assertEqual(loaded.current_epoch, 0)
        loaded.train_model([X_tensor], 2)
        self.assertEqual(loaded.current_epoch, 1)

    def test_h5_dynamic(self):

        ae = get_model(
            'rnn',
            velocity=self.velocity
        )(self.prior, use_prior_weights=True)
        ae._decoder[0].weight = torch.nn.parameter.Parameter(
            torch.tensor(self.inv_prior, dtype=torch.float32)
        )

        to(ae, self.device)
        ae.save(self.temp_file_name)
        to(ae, 'cpu')

        stub = read(
            self.temp_file_name,
            model_class=_ModelStub
        )

        npt.assert_almost_equal(
            A,
            stub.prior_network.values
        )

        self.assertDictEqual(
            stub.kwargs,
            {
                'input_dropout_rate': 0.5,
                'hidden_dropout_rate': 0.0,
                'output_activation': 'relu',
                'activation': 'relu'
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
            self.prior,
            use_prior_weights=False,
            activation='softplus',
            output_activation='softplus'
        )

        ae.set_time_parameters(
            output_t_plus_one=True,
            n_additional_predictions=1,
            loss_offset=0
        )

        ae._decoder[0].weight = torch.nn.parameter.Parameter(
            torch.tensor(self.inv_prior, dtype=torch.float32)
        )

        to(ae, self.device)
        ae.save(self.temp_file_name)
        to(ae, 'cpu')

        stub = read(
            self.temp_file_name,
            model_class=_ModelStub
        )

        npt.assert_almost_equal(
            A,
            stub.prior_network.values
        )

        self.assertDictEqual(
            stub.kwargs,
            {
                'input_dropout_rate': 0.5,
                'hidden_dropout_rate': 0.0,
                'output_activation': 'softplus',
                'activation': 'softplus'
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
        )(self.prior, use_prior_weights=True)
        ae._decoder[0].weight = torch.nn.parameter.Parameter(
            torch.tensor(self.inv_prior, dtype=torch.float32)
        )

        ae._training_loss = np.array([1., 1., 1.])
        ae._validation_loss = np.array([2., 2., 2.])

        to(ae, self.device)
        ae.save(self.temp_file_name)
        to(ae, 'cpu')
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

        npt.assert_almost_equal(
            loaded_ae._training_loss,
            ae._training_loss
        )

        npt.assert_almost_equal(
            loaded_ae._validation_loss,
            ae._validation_loss
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
        )(self.prior, use_prior_weights=True)

        ae.set_scaling(
            torch.ones(4),
            torch.ones(4)
        )

        to(ae, self.device)
        ae.save(self.temp_file_name)
        to(ae, 'cpu')
        ae.eval()

        loaded_ae = read(self.temp_file_name)
        loaded_ae.eval()

        torch.testing.assert_close(
            ae.count_scaler,
            loaded_ae.count_scaler
        )

        torch.testing.assert_close(
            torch.ones(4),
            loaded_ae.count_scaler
        )

        torch.testing.assert_close(
            ae.velocity_scaler,
            loaded_ae.velocity_scaler
        )

        torch.testing.assert_close(
            torch.ones(4),
            loaded_ae.velocity_scaler
        )

        torch.testing.assert_close(
            ae.count_to_velocity_scaler,
            loaded_ae.count_to_velocity_scaler
        )

        torch.testing.assert_close(
            torch.ones(4),
            loaded_ae.count_to_velocity_scaler
        )

        torch.testing.assert_close(
            ae.velocity_to_count_scaler,
            loaded_ae.velocity_to_count_scaler
        )

        torch.testing.assert_close(
            torch.ones(4),
            loaded_ae.velocity_to_count_scaler
        )

    def test_serialize_dynamic(self):

        ae = get_model(
            'rnn',
            velocity=self.velocity
        )(self.prior, use_prior_weights=True)

        ae._decoder[0].weight = torch.nn.parameter.Parameter(
            torch.tensor(self.inv_prior, dtype=torch.float32)
        )

        to(ae, self.device)
        ae.save(self.temp_file_name)
        to(ae, 'cpu')

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


class TestSerializeDF(TestSerializer):

    prior = pd.DataFrame(A)
    inv_prior = pd.DataFrame(pinv(A)).T.values


class TestSerializeAD(TestSerializer):

    prior = ad.AnnData(A)
    inv_prior = pd.DataFrame(pinv(A)).T.values


class TestSerializeADSparseCSR(TestSerializer):

    prior = ad.AnnData(csr_matrix(A))
    inv_prior = pd.DataFrame(pinv(A)).T.values


class TestBiophysical(_SetupMixin, unittest.TestCase):

    prior = A

    def test_serialize_biophysical(self):

        biophysical = get_model('biophysical')(
            self.prior,
            activation='tanh',
            output_activation='softplus'
        )

        biophysical._training_loss = np.array([(1., 1., 1.), (1., 1., 1.)])
        biophysical._validation_loss = np.array([(2., 2., 2.), (2., 2., 2.)])
        biophysical.current_epoch = 3

        to(biophysical, self.device)
        biophysical.save(self.temp_file_name)
        to(biophysical, 'cpu')
        biophysical.eval()

        loaded_biophysical = read(self.temp_file_name)
        loaded_biophysical.eval()

        self._compare_module(
            biophysical._transcription_model,
            loaded_biophysical._transcription_model
        )

        self._compare_module(
            biophysical._decay_model,
            loaded_biophysical._decay_model
        )

        npt.assert_almost_equal(
            loaded_biophysical._training_loss,
            biophysical._training_loss
        )

        npt.assert_almost_equal(
            loaded_biophysical._validation_loss,
            biophysical._validation_loss
        )

        npt.assert_almost_equal(
            loaded_biophysical.current_epoch,
            biophysical.current_epoch
        )

        self.assertEqual(
            loaded_biophysical.activation,
            'tanh'
        )

        self.assertEqual(
            loaded_biophysical.output_activation,
            'softplus'
        )

        with torch.no_grad():
            torch.testing.assert_close(
                biophysical(
                    biophysical.input_data(XTV_tensor)
                ),
                loaded_biophysical(
                    loaded_biophysical.input_data(XTV_tensor)
                )
            )

    def test_serialize_biophysical_nodecay(self):

        biophysical = get_model('biophysical')(
            self.prior,
            decay_model=False,
            activation='tanh',
            output_activation='softplus'
        )

        to(biophysical, self.device)
        biophysical.save(self.temp_file_name)
        to(biophysical, 'cpu')
        biophysical.eval()

        loaded_biophysical = read(self.temp_file_name)
        loaded_biophysical.eval()

        self._compare_module(
            biophysical._transcription_model,
            loaded_biophysical._transcription_model
        )

        self.assertIsNone(biophysical._decay_model)
        self.assertIsNone(loaded_biophysical._decay_model)

        self.assertEqual(
            loaded_biophysical.activation,
            'tanh'
        )

        self.assertEqual(
            loaded_biophysical.output_activation,
            'softplus'
        )

        with torch.no_grad():
            torch.testing.assert_close(
                biophysical(
                    biophysical.input_data(XTV_tensor)
                ),
                loaded_biophysical(
                    loaded_biophysical.input_data(XTV_tensor)
                )
            )

    def test_serialize_biophysical_diffk(self):

        biophysical = get_model('biophysical')(
            self.prior,
            decay_k=50
        )

        biophysical.set_scaling(
            count_scaling=np.arange(4),
            velocity_scaling=np.arange(4, 8)
        )

        biophysical._decay_model.set_scaling(
            count_scaling=np.arange(4),
            velocity_scaling=np.arange(4, 8)
        )

        to(biophysical, self.device)
        biophysical.save(self.temp_file_name)
        to(biophysical, 'cpu')
        biophysical.eval()

        loaded_biophysical = read(self.temp_file_name)
        loaded_biophysical.eval()

        torch.testing.assert_close(
            biophysical._velocity_inverse_scaler,
            loaded_biophysical._velocity_inverse_scaler
        )

        torch.testing.assert_close(
            biophysical._count_inverse_scaler,
            loaded_biophysical._count_inverse_scaler
        )

        torch.testing.assert_close(
            biophysical._decay_model._velocity_inverse_scaler,
            loaded_biophysical._decay_model._velocity_inverse_scaler
        )

        torch.testing.assert_close(
            biophysical._decay_model._count_inverse_scaler,
            loaded_biophysical._decay_model._count_inverse_scaler
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
            torch.testing.assert_close(
                biophysical(
                    biophysical.input_data(XTV_tensor)
                ),
                loaded_biophysical(
                    loaded_biophysical.input_data(XTV_tensor)
                )
            )

    def test_serialize_decay_module(self):

        decay = get_model('decay')(
            3
        )

        decay.set_scaling(
            count_scaling=np.arange(3),
            velocity_scaling=np.arange(3, 6)
        )

        to(decay, self.device)
        decay.save(self.temp_file_name)
        to(decay, 'cpu')
        decay.eval()

        loaded_decay = read(self.temp_file_name)
        loaded_decay.eval()

        torch.testing.assert_close(
            decay._velocity_inverse_scaler,
            loaded_decay._velocity_inverse_scaler
        )

        torch.testing.assert_close(
            decay._count_inverse_scaler,
            loaded_decay._count_inverse_scaler
        )

        self._compare_module(
            decay,
            loaded_decay
        )

    def test_serialize_decay_module_diffk(self):

        decay = get_model('decay')(
            n_genes=3,
            hidden_layer_width=10
        )

        to(decay, self.device)
        decay.save(self.temp_file_name)
        to(decay, 'cpu')
        decay.eval()

        loaded_decay = read(self.temp_file_name)
        loaded_decay.eval()

        self._compare_module(
            decay,
            loaded_decay
        )


class TestChromatin(_SetupMixin, unittest.TestCase):

    def test_h5_chromatin(self):

        model = get_model(
            'chromatin',
        )(
            4,
            25
        )

        to(model, self.device)
        model.save(self.temp_file_name)
        to(model, 'cpu')

        loaded_model = read(self.temp_file_name)
        loaded_model.eval()

        self._compare_module(
            model,
            loaded_model
        )

    def test_h5_chromatin_aware(self):

        model = get_model(
            'chromatin_aware',
        )(
            G_TO_PEAK_PRIOR,
            PEAK_TO_TF_PRIOR
        )

        to(model, self.device)
        model.save(self.temp_file_name)
        to(model, 'cpu')

        loaded_model = read(self.temp_file_name)
        loaded_model.eval()

        self._compare_module(
            model,
            loaded_model
        )

    def test_h5_chromatin_aware_ad(self):

        model = get_model(
            'chromatin_aware',
        )(
            ad.AnnData(
                csr_matrix(G_TO_PEAK_PRIOR, shape=G_TO_PEAK_PRIOR.shape)
            ),
            ad.AnnData(
                csr_matrix(PEAK_TO_TF_PRIOR, shape=PEAK_TO_TF_PRIOR.shape)
            )
        )

        to(model, self.device)
        model.save(self.temp_file_name)
        to(model, 'cpu')

        loaded_model = read(self.temp_file_name)
        loaded_model.eval()

        self._compare_module(
            model,
            loaded_model
        )


class TestSerializerVelocity(TestSerializer):

    velocity = True

    @unittest.skip
    def test_h5_train_checkpoint(self):
        pass


class TestSerializeMultimodel(_SetupMixin, unittest.TestCase):

    prior = A

    def test_multimodels(self):

        model = get_model('static_multilayer', multisubmodel=True)(
            prior_network=self.prior,
            intermediate_sizes=(3, 3),
            decoder_sizes=(3, 3),
            tfa_activation='softplus',
            activation='tanh',
            output_activation='relu',
            input_dropout_rate=0.2,
            hidden_dropout_rate=0.5,
            intermediate_dropout_rate=0.5
        )

        model.add_submodel(
            'testy',
            model.create_submodule(
                (3, 4, 3),
                activation='softplus'
            )
        )

        model.select_submodel(
            'testy',
            'intermediate'
        )

        to(model, self.device)
        model.save(self.temp_file_name)
        to(model, 'cpu')

        loaded_model = read(
            self.temp_file_name,
            submodule_templates=[
                (
                    'testy',
                    model.create_submodule(
                        (3, 4, 3),
                        activation='softplus'
                    )
                )
            ]
        )

        loaded_model.select_submodel(
            'testy',
            'intermediate'
        )

        self._compare_module(
            model,
            loaded_model
        )

        model.select_submodel(
            'default_intermediate',
            'intermediate'
        )

        loaded_model.select_submodel(
            'default_intermediate',
            'intermediate'
        )

        self._compare_module(
            model.module_bag['testy'],
            loaded_model.module_bag['testy']
        )

        self._compare_module(
            model,
            loaded_model
        )


class TestSerializerCUDA(TestSerializer):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'


class TestBiophysicalCUDA(TestBiophysical):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
