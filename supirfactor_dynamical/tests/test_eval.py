import unittest
import torch
import torcheval
import numpy as np

from supirfactor_dynamical.postprocessing.eval import (
    f1_score,
    r2_score
)

from supirfactor_dynamical._utils.misc import (
    argmax_last_dim
)


class _ModelStub():

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
            [[target, predicts]],
            _ModelStub(),
            target_data_idx=0,
            input_data_idx=1,
            multioutput=self.multioutput
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

    def test_r2(self):

        rng = np.random.default_rng(self.seed)

        target = torch.LongTensor(
            rng.choice([0, 1, 2], size=(10, 2))
        )

        target_encoded = torch.nn.functional.one_hot(
            target
        )

        inputs = torch.clone(target_encoded)

        inputs[0, 0, :] = torch.LongTensor([1, 0, 0])
        inputs[1, 0, :] = torch.LongTensor([0, 0, 1])
        inputs[2, 0, :] = torch.LongTensor([0, 1, 0])

        f1s = f1_score(
            [[target_encoded, inputs]],
            _ModelStub(),
            target_data_idx=0,
            input_data_idx=1,
            multioutput=self.multioutput
        )

        te_f1s = torcheval.metrics.functional.multiclass_f1_score(
            inputs.reshape(-1, 3),
            argmax_last_dim(target_encoded).reshape(-1),
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
