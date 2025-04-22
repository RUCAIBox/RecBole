SASRecCPR
===========

Introduction
---------------------

`[paper] <https://dl.acm.org/doi/10.1145/3616855.3635755>`_

**Title:** To Copy, or not to Copy; That is a Critical Issue of the Output Softmax Layer in Neural Sequential Recommenders

**Authors:** Haw-Shiuan Chang, Nikhil Agarwal, Andrew McCallum

**Abstract:**  Recent studies suggest that the existing neural models have difficulty handling repeated items in sequential recommendation tasks. However, our understanding of this difficulty is still limited. In this study, we substantially advance this field by identifying a major source of the problem: the single hidden state embedding and static item embeddings in the output softmax layer. Specifically, the similarity structure of the global item embedding in the softmax layer sometimes forces the single hidden state embedding to be close to new items when copying is a better choice, while sometimes forcing the hidden state to be close to the items from the input inappropriately. To alleviate the problem, we adapt the recently-proposed softmax alternatives such as softmax-CPR to sequential recommendation tasks and demonstrate that the new softmax architectures unleash the capability of the neural encoder on learning when to copy and when to exclude the items from the input sequence. By only making some simple modifications on the output softmax layer for SASRec and GRU4Rec, softmax-CPR achieves consistent improvement in 12 datasets. With almost the same model size, our best method not only improves the average NDCG\@10 of GRU4Rec in 5 datasets with duplicated items by 10% (4%-17% individually) but also improves 7 datasets without duplicated items by 24% (8%-39%)!

Running with RecBole
-------------------------

**Model Hyper-Parameters:**

- ``hidden_size (int)`` : The number of features in the hidden state. It is also the initial embedding size of item. Defaults to ``64``.
- ``inner_size (int)`` : The inner hidden size in feed-forward layer. Defaults to ``256``.
- ``n_layers (int)`` : The number of transformer layers in transformer encoder. Defaults to ``2``.
- ``n_heads (int)`` : The number of attention heads for multi-head attention layer. Defaults to ``2``.
- ``hidden_dropout_prob (float)`` : The probability of an element to be zeroed. Defaults to ``0.5``.
- ``attn_dropout_prob (float)`` : The probability of an attention score to be zeroed. Defaults to ``0.5``.
- ``hidden_act (str)`` : The activation function in feed-forward layer. Defaults to ``'gelu'``. Range in ``['gelu', 'relu', 'swish', 'tanh', 'sigmoid']``.
- ``layer_norm_eps (float)`` : A value added to the denominator for numerical stability. Defaults to ``1e-12``.
- ``initializer_range (float)`` : The standard deviation for normal initialization. Defaults to 0.02``.
- ``loss_type (str)`` : The type of loss function. If it is set to ``'CE'``, the training task is regarded as a multi-classification task and the target item is the ground truth. In this way, negative sampling is not needed. If it is set to ``'BPR'``, the training task will be optimized in the pair-wise way, which maximizes the difference between the positive item and the negative one. In this way, negative sampling is necessary, such as setting ``--train_neg_sample_args="{'distribution': 'uniform', 'sample_num': 1}"``. Defaults to ``'CE'``. Range in ``['BPR', 'CE']``.
- ``use_out_emb (bool)``: If False, we share the output item embedding and input item embedding. Defaults to ``False``
- ``n_facet_all (int)``: Number of linear layers for context partition, reranker partition, pointer network, and most items in the vocabulary. Notice that n_facet_all = n_facet + n_facet_context + n_facet_reranker*len(reranker_CAN_NUM_arr) + n_facet_emb. Default to ``5``.
- ``n_facet (int)``: Number of the output hidden states for most items in the vocabulary. If n_facet \> 1, we will use mixture of softmax (MoS). Default to ``1``.
- ``n_facet_context (int)``: Number of the output hidden states for the context partition. This number should be either 0, 1 or n_facet (If you use MoS). Default to ``1``.
- ``n_facet_reranker (int)``: Number of the output hidden states for a single reranker partition. This number should be either 0, 1 or n_facet (If you use MoS). Default to ``1``.
- ``reranker_CAN_NUM (str)``: The size of reranker partitions. If you want to use 3 reranker partitions with size 500, 100, and 20, set "500,100,20". Notice that the number should have a descent order (e.g., setting it to 20,100,500 is incorrect). Default to ``100``.
- ``n_facet_emb (int)``: Number of the output hidden states for pointer network. This number should be either 0 or 2. Default to ``2``.
- ``n_facet_hidden (int)``: min(n_facet_hidden, num_layers) = H hyperparameter in multiple input hidden states (Mi) [3]_. If not using Mi, set this number to 1. Default to ``1``.
- ``n_facet_window (int)``: -n_facet_window + 1 is the W hyperparameter in multiple input hidden states [3]_. If not using Mi, set this number to 0. Default to ``-2``.
- ``n_facet_MLP (int)``: The dimension of q_ct in [3]_ is (-n_facet_MLP + 1)*embedding_size. If not using Mi, set this number to 0. Default to ``-1``.
- ``weight_mode (str)``: The method of merging probability distribution in MoS. The value could be "dynamic" [4]_, "static", and "max_logits" [1]_. Default to ``''``.
- ``context_norm (int)``: If setting 0, we remove the denominator in Equation (5) of [1]_. Default to ``1``.
- ``partition_merging_mode (str)``: If "replace", the logit from context partition and pointer network would overwrite the logit from reranker partition and original softmax. Otherwise, the logit would be added. Default to ``'replace'``.
- ``reranker_merging_mode (str)``: If "add", the logit from reranker partition would be added with the original softmax. Otherwise, the softmax logit would be replaced by the logit from reranker partition. Default to ``'replace'``.
- ``use_proj_bias (bool)``: In linear layers for all output hidden states, if we want to use the bias term. Default to ``1``.
- ``post_remove_context (int)``: Setting the probability of all the items in the history to be 0 [2]_. Default to ``0``.

.. [1] Haw-Shiuan Chang, Nikhil Agarwal, and Andrew McCallum. "To Copy, or not to Copy; That is a Critical Issue of the Output Softmax Layer in Neural Sequential Recommenders." In Proceedings of The 17th ACM Inernational Conference on Web Search and Data Mining (WSDM 24)
.. [2] Ming Li, Ali Vardasbi, Andrew Yates, and Maarten de Rijke. 2023. Repetition and Exploration in Sequential Recommendation. In SIGIR 2023: 46th international ACM SIGIR Conference on Research and Development in Information Retrieval. ACM, 2532–2541.
.. [3] Haw-Shiuan Chang and Andrew McCallum. 2022. Softmax bottleneck makes language models unable to represent multi-mode word distributions. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 8048–8073
.. [4] Zhilin Yang, Zihang Dai, Ruslan Salakhutdinov, and William W. Cohen. "Breaking the Softmax Bottleneck: A High-Rank RNN Language Model." In International Conference on Learning Representations. 2018.

**A Running Example:**

Write the following code to a python file, such as `run.py`

.. code:: python

   from recbole.quick_start import run_recbole

   parameter_dict = {
      'train_neg_sample_args': None,
   }
   run_recbole(model='SASRecCPR', dataset='ml-100k', config_dict=parameter_dict)

And then:

.. code:: bash

   python run.py

Tuning Hyper Parameters
-------------------------

If you want to use ``HyperTuning`` to tune hyper parameters of this model, you can copy the following settings and name it as ``hyper.test``.

.. code:: bash

   learning_rate choice [0.01,0.005,0.001,0.0005,0.0001]
   attn_dropout_prob choice [0.2,0.5]
   hidden_dropout_prob choice [0.2,0.5]
   n_heads choice [1,2]
   n_layers choice [1,2,3]

Note that we just provide these hyper parameter ranges for reference only, and we can not guarantee that they are the optimal range of this model.

Then, with the source code of RecBole (you can download it from GitHub), you can run the ``run_hyper.py`` to tuning:

.. code:: bash

	python run_hyper.py --model=[model_name] --dataset=[dataset_name] --config_files=[config_files_path] --params_file=hyper.test

For more details about Parameter Tuning, refer to :doc:`../../../user_guide/usage/parameter_tuning`.


If you want to change parameters, dataset or evaluation settings, take a look at

- :doc:`../../../user_guide/config_settings`
- :doc:`../../../user_guide/data_intro`
- :doc:`../../../user_guide/train_eval_intro`
- :doc:`../../../user_guide/usage`