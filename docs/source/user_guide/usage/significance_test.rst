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
Finally, we can view the results of significant test from the terminal and ``significant_test.txt``. For example, we can see the results of significance test as follows:

.. code:: none

    recall@10 = Ttest_relResult(statistic=-21.593406593406602, pvalue=0.014730543885695793)
    mrr@10 = Ttest_relResult(statistic=-20.321888412017177, pvalue=0.01565077667413463)
    ndcg@10 = Ttest_relResult(statistic=-26.763440860215045, pvalue=0.011887928803273138)
    hit@10 = Ttest_relResult(statistic=-26.316239316239344, pvalue=0.012089752150296406)
    precision@10 = Ttest_relResult(statistic=-186.79999999999984, pvalue=0.0017039981023832399)

The meanings of statistic and pvalue can be referred to `link`__ .

.. __: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_rel.html