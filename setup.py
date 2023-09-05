import os
from setuptools import setup, find_packages

install_requires = [
    "numpy",
    "pandas>=1.0",
    "anndata>=0.8",
    "tqdm",
    "torch",
    "scipy",
    "inferelator"
]

tests_require = [
    "coverage",
    "pytest"
]

version = "1.0.0"

# Description from README.md
long_description = "\n\n".join(
    [open(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "README.md"
        ),
        "r"
    ).read()]
)

setup(
    name="supirfactor_dynamical",
    version=version,
    description="Dynamical Model Extension of the Supirfactor Model",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/GreshamLab/supirfactor-dynamical",
    author="Chris Jackson",
    author_email="cj59@nyu.edu",
    maintainer="Chris Jackson",
    maintainer_email="cj59@nyu.edu",
    packages=find_packages(include=[
        "supirfactor_dynamical",
        "supirfactor_dynamical.*"
    ]),
    zip_safe=False,
    install_requires=install_requires,
    tests_require=tests_require,
    test_suite="pytest",
)
