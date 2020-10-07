from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os

from setuptools import setup, find_packages

install_requires = ['numpy>=1.17.2', 'torch>=1.2.0', 'scipy>=1.1.0', 'pandas>=1.1.2', 'tqdm>=4.48.2', 'scikit_learn>=0.20.3']

setup_requires = []

extras_require = {
    'matplotlib': ['matplotlib>=3.1.3'],
    'hyperopt': ['hyperopt>=0.2.4'],
    'yapf': ['yapf==0.30.0'],
    'flake8': ['flake8==3.7.7', 'flake8-quotes==2.1.0']
}

# Readthedocs requires Sphinx extensions to be specified as part of
# install_requires in order to build properly.
on_rtd = os.environ.get('READTHEDOCS', None) == 'True'
if on_rtd:
    install_requires.extend(setup_requires)

setup(
    name='recbox',
    version=
    '0.0.1',  # please remember to edit recbox/__init__.py in response, once updating the version
    description='A package for building recommender systems',
    url='https://github.com/RUCAIBox/RecBox',
    author='RecBoxTeam',
    author_email='ContactRecBoxTeam',
    packages=[
        package for package in find_packages()
        if package.startswith('recbox')
    ],
    install_requires=install_requires,
    setup_requires=setup_requires,
    extras_require=extras_require,
    zip_safe=False)
