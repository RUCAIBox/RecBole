Significance Test
=============

Tests for statistical significance indicate whether observed differences between assessment results occur because of sampling error or chance. 
Such "insignificant" results should be ignored because they do not reflect real differences. In RecBole, we implemented ``significance_test.py``
for significance test. In this section, we will present a typical usage of this script.

Create Config files
-------------

Usually, we execute the significance test for our model and baseline, thus we create two config files for ours and baseline, respectively. For example, we 
create ``ours.yaml`` and ``baseline.yaml`` for ours and baseline model.
    
Run Command 
-------------------------------------------------

.. code:: none

    python significant_test.py --model_ours ours --model_baseline baseline --run_times 30 --config_files "ours.yaml baseline.yaml"

View Results
-------------------------------------------------
Finally, we can view the results of significant test from the terminal and ``significant_test.txt``.