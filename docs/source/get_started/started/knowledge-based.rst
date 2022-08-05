Quick Start: Knowledge-based Recommendation
=============================================
For Knowledge-based Recommendation, we choose **CKE** model to show you how
to train and test it on the **ml-100k** dataset from both **API** and
**source code**.

.. _header-n152:

Quick-start From API
---------------------

.. _header-n153:

1. Prepare your data:
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

Before running a model, firstly you need to prepare and load data. To
help users quickly get start, RecBole has a build-in dataset **ml-100k**
and you can directly use it. However, if you want to use other datasets,
you can read `Running New
Dataset <https://recbole.io/docs/user_guide/usage/running_new_dataset.html>`__
for more information.

Then, you need to set data config for data loading. You can create a
yaml file called test.yaml and write the following settings:

.. code:: yaml

   # dataset config : Knowledge-based Recommendation
   USER_ID_FIELD: user_id
   ITEM_ID_FIELD: item_id
   HEAD_ENTITY_ID_FIELD: head_id
   TAIL_ENTITY_ID_FIELD: tail_id
   RELATION_ID_FIELD: relation_id
   ENTITY_ID_FIELD: entity_id
   load_col:
       inter: [user_id, item_id]
       kg: [head_id, relation_id, tail_id]
       link: [item_id, entity_id]

Knowledge-based recommendation models utilize KG information to make
recommendations, so it needs to specify and load the kg information of
the dataset. 

You can get the kg triplets by calling ``dataset.kg_feat``, it looks like:

.. code:: 

            head_id                          relation_id    tail_id
   0      m.04ctbw8                   film.producer.film    m.0bln8
   1       m.0c3wmn                      film.film.actor  m.02vxxgs
   2        m.04t36  film.film_genre.films_in_this_genre   m.05sbv3
   3       m.08jl3y                      film.film.actor  m.0v187kf
   4      m.0513fcb                      film.film.actor  m.0glmggf
   ...          ...                                  ...        ...
   91626  m.09v46zg                    film.film.prequel  m.02862zk
   91627    m.0jyx6           film.film.award_nomination    m.0gr51
   91628  m.043qq5y             film.film.cinematography  m.0bnth9_
   91629  m.0b_zqd8                      film.film.actor   m.07xv9s
   91630   m.0cr7n8                      film.film.actor  m.0g99qg5

RecBole also offer ``entity2id`` and ``id2entity``, which map ``item_id`` and ``entity``


.. _header-n159:

2. Choose a model:
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

You can choose a model from our `Model
Introduction <https://recbole.io/docs/user_guide/model_intro.html>`__.
Here we choose CKE model to demonstrate how to train and test the
knowledge-based Recommendation,model.

Then, you need to set the parameter for CKE model. You can check the
`CKE <https://recbole.io/docs/user_guide/model/knowledge/cke.html>`__
and add the model settings into the test.yaml, like:

.. code:: yaml

   # model config
   embedding_size: 64
   kg_embedding_size: 64
   reg_weights: [1e-02,1e-02]

If you want to run different models, you can read `Running Different
Models <https://recbole.io/docs/user_guide/usage/running_different_models.html>`__
for more information.

.. _header-n164:

3. Set training and evaluation config:
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

In RecBole, we support multiple training and evaluation methods. You can
choose how to train and test model by simply setting the config.

Here we want to train and test the CKE model in training-validation-test
method (optimize model parameters on the training set, do parameter
selection according to the results on the validation set, and finally
report the results on the test set) and evaluate the model performance
by full ranking with all item candidates, so we can add the following
settings into the test.yaml.

.. code:: yaml

   # Training and evaluation config
   eval_args:
      split: {'RS': [0.8, 0.1, 0.1]}
      group_by: user 
      order: RO 
      mode: full
   metrics: ['Recall', 'MRR', 'NDCG', 'Hit', 'Precision']
   topk: 10
   valid_metric: MRR@10

For more details of training and evaluation config, please refer to
`Training
Settings <https://recbole.io/docs/user_guide/config/training_settings.html>`__
and `Evaluation
Settings <https://recbole.io/docs/user_guide/config/evaluation_settings.html>`__.

.. _header-n269:

4. Run the model and collect the result
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

Now you have finished all the preparations, it’s time to run the model!

You can create a new python file (e.g., run.py), and write the following
code:

.. code:: python

   from recbole.quick_start import run_recbole
   run_recbole(model='CKE', dataset='ml-100k', config_file_list=['test.yaml'])

Then run the following command:

.. code:: python

   python run.py

And you will obtain the output like:

.. code:: 

   16 Jul 17:35    INFO  ml-100k
   The number of users: 944
   Average actions of users: 106.04453870625663
   The number of items: 1683
   Average actions of items: 59.45303210463734
   The number of inters: 100000
   The sparsity of the dataset: 93.70575143257098%
   Remain Fields: ['entity_id', 'user_id', 'item_id', 'head_id', 'relation_id', 'tail_id']
   The number of entities: 34713
   The number of relations: 26
   The number of triples: 91631
   The number of items that have been linked to KG: 1598
   16 Jul 17:35    INFO  [Training]: train_batch_size = [2048] negative sampling: [{'uniform': 1}]
   16 Jul 17:35    INFO  [Evaluation]: eval_batch_size = [4096] eval_args: [{'split': {'RS': [0.8, 0.1, 0.1]}, 'group_by': 'user', 'order': 'RO', 'mode': 'full'}]
   16 Jul 17:35    INFO  CKE(
     (user_embedding): Embedding(944, 64)
     (item_embedding): Embedding(1683, 64)
     (entity_embedding): Embedding(34713, 64)
     (relation_embedding): Embedding(26, 64)
     (trans_w): Embedding(26, 4096)
     (rec_loss): BPRLoss()
     (kg_loss): BPRLoss()
     (reg_loss): EmbLoss()
   )
   Trainable parameters: 2497920
   Train     0: 100%|██████████████████████████████████████████████████| 40/40 [00:06<00:00,  5.73it/s]
   16 Jul 17:36    INFO  epoch 0 training [time: 6.98s, train_loss1: 27.7243, train_loss2: 21.9423, train_loss3: 0.0436]
   Evaluate   : 100%|██████████████████████████████████████████████| 472/472 [00:00<00:00, 3589.95it/s]
   16 Jul 17:36    INFO  epoch 0 evaluating [time: 0.13s, valid_score: 0.019500]
   ......
   Train    86: 100%|██████████████████████████████████████████████████| 40/40 [00:07<00:00,  5.36it/s]
   16 Jul 17:47    INFO  epoch 86 training [time: 7.46s, train_loss1: 3.7211, train_loss2: 2.9693, train_loss3: 0.1157]
   Evaluate   : 100%|██████████████████████████████████████████████| 472/472 [00:00<00:00, 3165.16it/s]
   16 Jul 17:47    INFO  epoch 86 evaluating [time: 0.15s, valid_score: 0.376600]
   16 Jul 17:47    INFO  valid result: 
   recall@10 : 0.2083    mrr@10 : 0.3766    ndcg@10 : 0.2238    hit@10 : 0.7455    precision@10 : 0.1544
   16 Jul 17:47    INFO  Finished training, best eval result in epoch 75
   16 Jul 17:47    INFO  Loading model structure and parameters from saved\CKE-Jul-16-2022_17-35-57.pth
   Evaluate   : 100%|██████████████████████████████████████████████| 472/472 [00:00<00:00, 2936.85it/s]
   16 Jul 17:47    INFO  best valid : OrderedDict([('recall@10', 0.2115), ('mrr@10', 0.3832), ('ndcg@10', 0.2296), ('hit@10', 0.7391), ('precision@10', 0.1584)])
   16 Jul 17:47    INFO  test result: OrderedDict([('recall@10', 0.2483), ('mrr@10', 0.4895), ('ndcg@10', 0.2912), ('hit@10', 0.7709), ('precision@10', 0.1951)])

Finally you will get the model’s performance on the test set and the
model file will be saved under the /saved. Besides, RecBole allows
tracking and visualizing train loss and valid score with TensorBoard,
please read the `Use
Tensorboard <https://recbole.io/docs/user_guide/usage/use_tensorboard.html>`__
for more details.

The above is the whole process of running a model in RecBole, and you
can read other docs for depth usage.

.. _header-n179:

Quick-start From Source
--------------------------

Besides using API, you can also directly run the source code of
`RecBole <https://github.com/RUCAIBox/RecBole>`__. The whole process is
similar to Quick-start From API. You can create a yaml file called
test.yaml and set all the config as follow:

.. code:: yaml

   # dataset config : Knowledge-based Recommendation
   USER_ID_FIELD: user_id
   ITEM_ID_FIELD: item_id
   HEAD_ENTITY_ID_FIELD: head_id
   TAIL_ENTITY_ID_FIELD: tail_id
   RELATION_ID_FIELD: relation_id
   ENTITY_ID_FIELD: entity_id
   load_col:
       inter: [user_id, item_id]
       kg: [head_id, relation_id, tail_id]
       link: [item_id, entity_id]
       
   # model config
   embedding_size: 64
   kg_embedding_size: 64
   reg_weights: [1e-02,1e-02]

   # Training and evaluation config
   eval_args:
      split: {'RS': [0.8, 0.1, 0.1]}
      group_by: user
      order: RO
      mode: full
   metrics: ['Recall', 'MRR', 'NDCG', 'Hit', 'Precision']
   topk: 10
   valid_metric: MRR@10

Then run the following command:

.. code:: python

   python run_recbole.py --model=CKE --dataset=ml-100k --config_files=test.yaml

And you will get the output of running the CKE model on the ml-100k
dataset.

If you want to change the parameters, such as ``embedding_size``, just
set the additional command parameters as you need:

.. code:: python

   python run_recbole.py --model=CKE --dataset=ml-100k --config_files=test.yaml --embedding_size=100
