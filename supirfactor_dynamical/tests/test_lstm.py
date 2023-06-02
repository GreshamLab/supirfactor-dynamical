import unittest

from supirfactor_dynamical import (
    TFLSTMDecoder
)

from .test_rnn import (
    TestTFRecurrentDecoder
)


class TestTFLSTMDecoder(TestTFRecurrentDecoder):

    weight_stack = 4
    class_holder = TFLSTMDecoder

    @unittest.SkipTest
    def test_train_loop_offset_predict():
        pass

    @unittest.SkipTest
    def test_r2_over_timemodel():
        pass

    @unittest.SkipTest
    def test_r2_over_timemodel_len2():
        pass

    @unittest.SkipTest
    def test_r2_model():
        pass

    @unittest.SkipTest
    def test_erv():
        pass
