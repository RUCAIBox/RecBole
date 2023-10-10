Random
===========

Introduction
---------------------

When discussing recommendation systems, accuracy is often regarded as the most crucial metric. 
However, besides accuracy, several other key metrics can evaluate the effectiveness of a recommendation system, such as diversity, coverage, and efficiency. 
In this context, the random recommendation algorithm is a valuable baseline. 
In terms of implementation, for a given user and item, the random recommendation algorithm provides a random rating.

Running with RecBole



**A Running Example:**

Write the following code to a python file, such as `run.py`

.. code:: python

   from recbole.quick_start import run_recbole

   run_recbole(model='Random', dataset='ml-100k')

And then:

.. code:: bash

   python run.py


If you want to change parameters, dataset or evaluation settings, take a look at

- :doc:`../../../user_guide/config_settings`
- :doc:`../../../user_guide/data_intro`
- :doc:`../../../user_guide/train_eval_intro`
- :doc:`../../../user_guide/usage`