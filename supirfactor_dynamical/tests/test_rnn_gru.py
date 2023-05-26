import unittest

from supirfactor_dynamical import (
    TFGRUAutoencoder,
    TFGRUDecoder,

)

from .test_rnn_dynamic import (
    TestTFRecurrentAutoencoder,
    TestTFRecurrentDecoder
)


class TestTFGRUAutoencoder(TestTFRecurrentAutoencoder):

    weight_stack = 3
    class_holder = TFGRUAutoencoder

    @unittest.SkipTest
    def test_train_loop_offset_predict(self):
        pass

    @unittest.SkipTest
    def test_erv(self):
        pass

    @unittest.SkipTest
    def test_latent_layer(self):
        pass

    @unittest.SkipTest
    def test_r2_model(self):
        pass

    @unittest.SkipTest
    def test_r2_over_timemodel(self):
        pass

    @unittest.SkipTest
    def test_r2_over_timemodel_len2(self):
        pass


class TestTFGRUDecoder(TestTFRecurrentDecoder):

    class_holder = TFGRUDecoder
