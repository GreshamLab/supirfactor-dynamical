from inferelator import (
    CrossValidationManager,
    inferelator_workflow,
    inferelator_verbose_level,
    MPControl,
    PreprocessData
)

from inferelator.tfa.pinv_tfa import ActivityOnlyPinvTFA

import argparse
import gc
import os

ap = argparse.ArgumentParser()

ap.add_argument(
    "--expression",
    dest="expression",
    action='store_const',
    const=True,
    default=False
)

ap.add_argument(
    "--denoised",
    dest="denoised",
    action='store_const',
    const=True,
    default=False
)

ap.add_argument(
    "--velocity",
    dest="velocity",
    action='store_const',
    const=True,
    default=False
)

ap.add_argument(
    "--decay_constant",
    dest="decay_constant",
    action='store_const',
    const=True,
    default=False
)

ap.add_argument(
    "--decay_variable",
    dest="decay_variable",
    action='store_const',
    const=True,
    default=False
)

args = ap.parse_args()

YEASTRACT_PRIOR = "JOINT_PRIOR_20230701.tsv.gz"
EXPRESSION_FILE = "2021_INFERELATOR_DATA.h5ad"

INPUT_DIR = '/mnt/ceph/users/cjackson/inferelator/data/RAPA/'
OUTPUT_PATH = '/mnt/ceph/users/cjackson/inferelator_rapa/'

REGRESSION = "stars"

os.makedirs(OUTPUT_PATH, exist_ok=True)


def set_up_workflow(wkf, full=False):
    wkf.set_file_paths(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_PATH,
        priors_file=YEASTRACT_PRIOR,
        gold_standard_file='gold_standard.tsv.gz'
    )

    wkf.set_tfa(tfa_driver=ActivityOnlyPinvTFA)

    if not full:
        wkf.set_crossvalidation_parameters(
            split_gold_standard_for_crossvalidation=True,
            cv_split_ratio=0.2
        )

        wkf.set_run_parameters(
            num_bootstraps=5
        )

    else:
        wkf.set_run_parameters(
            num_bootstraps=50
        )

    if REGRESSION == "bbsr":
        wkf.set_regression_parameters(clr_only=True)
    else:
        wkf.set_regression_parameters(max_iter=5000)

    return wkf


def set_up_cv(wkf):
    cv = CrossValidationManager(wkf)
    cv.add_gridsearch_parameter(
        'random_seed',
        list(range(42, 52))
    )

    return cv


if __name__ == "__main__":

    inferelator_verbose_level(1)

    PreprocessData.set_preprocessing_method(
        method_tfa=None,
        method_predictors='robustscaler',
        method_response='robustscaler'
    )

    MPControl.set_multiprocess_engine("dask-cluster")
    MPControl.client.use_default_configuration("rusty_rome", n_jobs=1)
    MPControl.client.add_worker_conda(
        "source ~/.local/anaconda3/bin/activate inferelator"
    )

    worker = set_up_workflow(
        inferelator_workflow(regression=REGRESSION, workflow="single-cell")
    )
    worker.set_expression_file(h5ad=EXPRESSION_FILE)
    worker.set_count_minimum(0.05)

    worker.append_to_path('output_dir', "expression")

    cv = set_up_cv(worker)
    cv.run()

    del worker
    del cv

    gc.collect()

    worker = set_up_workflow(
        inferelator_workflow(regression=REGRESSION, workflow="single-cell")
    )
    worker.set_expression_file(h5ad=EXPRESSION_FILE)
    worker.set_count_minimum(0.05)
    worker.set_shuffle_parameters(shuffle_prior_axis=-1)
    worker.append_to_path('output_dir', "expression_shuffled")

    cv = set_up_cv(worker)
    cv.run()

    del worker
    del cv

    gc.collect()

    worker = set_up_workflow(
        inferelator_workflow(regression=REGRESSION, workflow="single-cell"),
        full=True
    )
    worker.set_expression_file(h5ad=EXPRESSION_FILE)
    worker.set_count_minimum(0.05)

    worker.append_to_path('output_dir', "expression_full")
    worker.run()
