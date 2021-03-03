Install RecBole
======================
RecBole can be installed from ``conda``, ``pip`` and source files.


System requirements
------------------------
RecBole is compatible with the following operating systems:

* Linux
* Windows 10
* macOS X

Python 3.6 (or later), torch 1.6.0 (or later) are required to install our library. If you want to use RecBole with GPU,
please ensure that CUDA or CUDAToolkit version is 9.2 or later.
This requires NVIDIA driver version >= 396.26 (for Linux) or >= 397.44 (for Windows10).


Install from conda
--------------------------
``Conda`` can be installed from `miniconda <https://conda.io/miniconda.html>`_ or
the full `anaconda <https://www.anaconda.com/download/>`_.
If you are in China, `Tsinghua Mirrors <https://mirror.tuna.tsinghua.edu.cn/help/anaconda/>`_ is recommended.

After installing ``conda``,
run `conda create -n recbole python=3.6` to create the Python 3.6 conda environment.
Then the environment can be activated by `conda activate recbole`.
At last, run the following command to install RecBole:

.. code:: bash

    conda install -c aibox recbole


Install from pip
-------------------------
To install RecBole from pip, only the following command is needed:

.. code:: bash

    pip install recbole


Install from source
-------------------------
Download the source files from GitHub.

.. code:: bash

    git clone https://github.com/RUCAIBox/RecBole.git && cd RecBole

Run the following command to install:

.. code:: bash

    pip install -e . --verbose
