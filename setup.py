from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os

from setuptools import setup, find_packages

install_requires = [
    "torch>=1.10.0",
    "numpy>=1.17.2",
    "scipy>=1.6.0",
    "pandas>=1.0.5",
    "tqdm>=4.48.2",
    "colorlog==4.7.2",
    "colorama==0.4.4",
    "scikit_learn>=0.23.2",
    "pyyaml>=5.1.0",
    "tensorboard>=2.5.0",
    "thop>=0.1.1.post2207130030",
    "tabulate>=0.8.10",
    "plotly>=4.0.0",
]

setup_requires = []

extras_require = {"hyperopt": ["hyperopt==0.2.5"], "ray": ["ray>=1.13.0"]}

classifiers = ["License :: OSI Approved :: MIT License"]

long_description = (
    "RecBole is developed based on Python and PyTorch for "
    "reproducing and developing recommendation algorithms in "
    "a unified, comprehensive and efficient framework for "
    "research purpose. In the first version, our library "
    "includes 53 recommendation algorithms, covering four "
    "major categories: General Recommendation, Sequential "
    "Recommendation, Context-aware Recommendation and "
    "Knowledge-based Recommendation. View RecBole homepage "
    "for more information: https://recbole.io"
)

# Readthedocs requires Sphinx extensions to be specified as part of
# install_requires in order to build properly.
on_rtd = os.environ.get("READTHEDOCS", None) == "True"
if on_rtd:
    install_requires.extend(setup_requires)

setup(
    name="recbole",
    version="1.1.1",  # please remember to edit recbole/__init__.py in response, once updating the version
    description="A unified, comprehensive and efficient recommendation library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/RUCAIBox/RecBole",
    author="RecBoleTeam",
    author_email="recbole@outlook.com",
    packages=[package for package in find_packages() if package.startswith("recbole")],
    include_package_data=True,
    install_requires=install_requires,
    setup_requires=setup_requires,
    extras_require=extras_require,
    zip_safe=False,
    classifiers=classifiers,
)
