import unittest


from supirfactor_dynamical.models._base_trainer import _TrainingMixin


class TestInputOutputOffsets(unittest.TestCase):

    L = 10

    def test_no_shift(self):

        a, b = _TrainingMixin._get_data_offsets(
            self.L,
            False,
            0,
            0
        )

        self.assertEqual(
            a, self.L
        )

        self.assertEqual(
            b, 0
        )

    def test_loss_shift(self):

        a, b = _TrainingMixin._get_data_offsets(
            self.L,
            False,
            0,
            5
        )

        self.assertEqual(
            a, self.L
        )

        self.assertEqual(
            b, 5
        )

    def test_predict_shift(self):

        a, b = _TrainingMixin._get_data_offsets(
            self.L,
            False,
            5,
            0
        )

        self.assertEqual(
            a, self.L - 5
        )

        self.assertEqual(
            b, 0
        )

    def test_both(self):

        a, b = _TrainingMixin._get_data_offsets(
            self.L,
            False,
            5,
            5
        )

        self.assertEqual(
            a, self.L - 5
        )

        self.assertEqual(
            b, 5
        )


class TestInputOutputOffsetsWithShift(unittest.TestCase):

    L = 10

    def test_no_shift(self):

        a, b = _TrainingMixin._get_data_offsets(
            self.L,
            True,
            0,
            0
        )

        self.assertEqual(
            a, self.L - 1
        )

        self.assertEqual(
            b, 1
        )

    def test_loss_shift(self):

        a, b = _TrainingMixin._get_data_offsets(
            self.L,
            True,
            0,
            5
        )

        self.assertEqual(
            a, self.L - 1
        )

        self.assertEqual(
            b, 6
        )

    def test_predict_shift(self):

        a, b = _TrainingMixin._get_data_offsets(
            self.L,
            True,
            5,
            0
        )

        self.assertEqual(
            a, self.L - 6
        )

        self.assertEqual(
            b, 1
        )

    def test_both(self):

        a, b = _TrainingMixin._get_data_offsets(
            self.L,
            True,
            5,
            5
        )

        self.assertEqual(
            a, self.L - 6
        )

        self.assertEqual(
            b, 6
        )
