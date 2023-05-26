import unittest

from supirfactor_dynamical import (
    TFLSTMAutoencoder,
    TFLSTMDecoder
)

from .test_rnn_dynamic import (
    TestTFRecurrentAutoencoder,
    TestTFRecurrentDecoder
)


class TestTFLSTMAutoencoder(TestTFRecurrentAutoencoder):

    weight_stack = 4
    class_holder = TFLSTMAutoencoder

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


class TestTFLSTMDecoder(TestTFRecurrentDecoder):

    class_holder = TFLSTMDecoder
