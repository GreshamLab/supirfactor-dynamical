import unittest
import torch
import torcheval
import torcheval.metrics
import numpy as np

from supirfactor_dynamical.postprocessing.eval import (
    f1_score,
    r2_score
)

from supirfactor_dynamical._utils.misc import (
    argmax_last_dim
)

from supirfactor_dynamical._utils._math import (
    _true_positive,
    _true_negative,
    _false_negative,
    _false_positive
)


class _ModelStub():

    _model_device = 'cpu'

    def eval(self):
        pass

    def output_data(self, x):
        return x

    def _slice_data_and_forward(self, x):
        return x


class TestEvalR2Raw(unittest.TestCase):

    multioutput = 'raw_values'
    seed = 150

    def test_r2(self):

        rng = np.random.default_rng(self.seed)

        target = torch.Tensor(
            rng.normal(10, 3, size=(20, 10, 5))
        )
        predicts = target + torch.Tensor(
            rng.normal(size=(20, 10, 5))
        )

        r2s = r2_score(
            [[[target, predicts]]],
            _ModelStub(),
            target_data_idx=0,
            input_data_idx=1,
            multioutput=self.multioutput,
            exclude_low_variance=0.001
        )

        te_r2s = torcheval.metrics.functional.r2_score(
            predicts.reshape(-1, 5),
            target.reshape(-1, 5),
            multioutput=self.multioutput
        )

        torch.testing.assert_close(
            r2s,
            te_r2s
        )


class TestEvalR2Avg(TestEvalR2Raw):

    multioutput = "uniform_average"


class TestEvalR2VarAvg(TestEvalR2Raw):

    multioutput = "variance_weighted"


class TestEvalF1Raw(unittest.TestCase):

    multioutput = None
    seed = 150

    @classmethod
    def setUpClass(cls) -> None:
        cls.rng = np.random.default_rng(cls.seed)

        cls.target = torch.LongTensor(
            cls.rng.choice([0, 1, 2], size=(10, 2))
        )

        cls.target_encoded = torch.nn.functional.one_hot(
            cls.target
        )

        cls.inputs = cls.target_encoded.type(torch.Tensor)

        cls.inputs[0, 0, :] = torch.Tensor([1., .10, .10])
        cls.inputs[1, 0, :] = torch.Tensor([.10, .10, 1.])
        cls.inputs[2, 0, :] = torch.Tensor([.10, 1., .10])

        return super().setUpClass()

    def test_maths(self):
        torch.testing.assert_close(
            torch.sum(
                torch.logical_and(self.inputs == 1., self.target_encoded == 1),
                axis=(0, 1)
            ),
            _true_positive(self.inputs == 1., self.target_encoded)
        )
        torch.testing.assert_close(
            torch.sum(
                torch.logical_and(self.inputs == 1., self.target_encoded == 0),
                axis=(0, 1)
            ),
            _false_positive(self.inputs == 1., self.target_encoded)
        )
        torch.testing.assert_close(
            torch.sum(
                torch.logical_and(self.inputs != 1., self.target_encoded == 1),
                axis=(0, 1)
            ),
            _false_negative(self.inputs == 1., self.target_encoded)
        )
        torch.testing.assert_close(
            torch.sum(
                torch.logical_and(self.inputs != 1., self.target_encoded == 0),
                axis=(0, 1)
            ),
            _true_negative(self.inputs == 1., self.target_encoded)
        )

    def test_f1(self):

        f1s = f1_score(
            [[self.target_encoded, self.inputs]],
            _ModelStub(),
            target_data_idx=0,
            input_data_idx=1,
            multioutput=self.multioutput
        )

        te_f1s = torcheval.metrics.functional.multiclass_f1_score(
            self.inputs.reshape(-1, 3),
            argmax_last_dim(self.target_encoded).reshape(-1),
            average=self.multioutput,
            num_classes=3
        )

        torch.testing.assert_close(
            f1s,
            te_f1s
        )


class TestEvalF1Micro(TestEvalF1Raw):

    multioutput = 'micro'


class TestEvalF1Macro(TestEvalF1Raw):

    multioutput = 'macro'


class TestEvalF1Weighted(TestEvalF1Raw):

    multioutput = 'weighted'
