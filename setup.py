from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os

from setuptools import setup, find_packages

install_requires = ['numpy>=1.17.2', 'torch>=1.6.0', 'scipy>=1.3.1', 'pandas>=1.0.5', 'tqdm>=4.48.2',
                    'scikit_learn>=0.23.2', 'pyyaml>=5.1.0', 'matplotlib>=3.1.3', 'hyperopt>=0.2.4']

setup_requires = []

extras_require = {}

# Readthedocs requires Sphinx extensions to be specified as part of
# install_requires in order to build properly.
on_rtd = os.environ.get('READTHEDOCS', None) == 'True'
if on_rtd:
    install_requires.extend(setup_requires)

setup(
    name='recbole',
    version=
    '0.1.0',  # please remember to edit recbole/__init__.py in response, once updating the version
    description='A package for building recommender systems',
    url='https://github.com/RUCAIBox/RecBole',
    author='RecBoleTeam',
    author_email='recbole@outlook.com',
    packages=[
        package for package in find_packages()
        if package.startswith('recbole')
    ],
    include_package_data=True,
    install_requires=install_requires,
    setup_requires=setup_requires,
    extras_require=extras_require,
    zip_safe=False)
