Customize Models
======================
Here, we present how to develop a new model, and apply it to the RecBole.

RecBole supports General, Context-aware, Sequential and Knowledge-based
recommendation.

Create a New Model Class
------------------------------
To begin with, we should create a new model implementing from one of :class:`~recbole.model.abstract_recommender.GeneralRecommender`,
:class:`~recbole.model.abstract_recommender.ContextRecommender`, :class:`~recbole.model.abstract_recommender.SequentialRecommender`,
:class:`~recbole.model.abstract_recommender.KnowledgeRecommender`.
For example, we would like to develop a general model named as NewModel and write the code to `newmodel.py`.

.. code:: python

    from recbole.model.abstract_recommender import GeneralRecommender

    class NewModel(GeneralRecommender):
        pass

Then, we need to indicate :attr:`~recbole.model.abstract_recommender.AbstractRecommender.input_type`,
RecBole supports two input types: :obj:`~recbole.utils.enum_type.InputType.POINTWISE` and :obj:`~recbole.utils.enum_type.InputType.PAIRWISE`.

:obj:`~recbole.utils.enum_type.InputType.POINTWISE` will give the :attr:`item` and the corresponding :attr:`label`, which is suitable for pointwise loss, e.g., Cross Entropy Loss.

:obj:`~recbole.utils.enum_type.InputType.PAIRWISE` will give the item :attr:`pos_item` and :attr:`neg_item`, which is suitable for pairwise loss, e.g., BPR Loss.

Suppose we want to use pairwise loss:

.. code:: python

    from recbole.utils import InputType
    from recbole.model.abstract_recommender import GeneralRecommender

    class NewModel(GeneralRecommender):

        input_type = InputType.PAIRWISE
        pass

Implement __init__()
--------------------------------
Then we redefine :meth:`__init__` method, :meth:`__init__` is used to initialize the model, including loading the dataset information, model parameters, define the model structure and initializing methods.

:meth:`__init__` input the parameters of :attr:`config`. and :attr:`dataset`, where :attr:`config` is used to input parameters,
:attr:`dataset` is leveraged to input datasets including :attr:`n_users`, :attr:`n_items`.

Here, we suppose the NewModel encode the users and items, where we use :func:`~recbole.model.init.xavier_normal_initialization` to initialize the parameters, and use inner product to compute the score.

.. code:: python

    import torch
    import torch.nn as nn

    from recbole.model.loss import BPRLoss
    from recbole.model.init import xavier_normal_initialization

    def __init__(self, config, dataset):
        super(NewModel, self).__init__(config, dataset)

        # load dataset info
        self.n_users = dataset.user_num
        self.n_items = dataset.item_num

        # load parameters info
        self.embedding_size = config['embedding_size']

        # define layers and loss
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)
        self.loss = BPRLoss()

        # parameters initialization
        self.apply(xavier_normal_initialization)


Implement calcualte_loss()
----------------------------------------
Then we define the :meth:`calculate_loss` method, :meth:`calculate_loss` is used to compute the loss,
the input parameters are :class:`~recbole.data.interaction.Interaction`, at last the method return a :class:`torch.Tensor` for computing the BP information.

.. code:: python

    import torch

    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        user_e = self.user_embedding(user)                        # [batch_size, embedding_size]
        pos_item_e = self.item_embedding(pos_item)                # [batch_size, embedding_size]
        neg_item_e = self.item_embedding(neg_item)                # [batch_size, embedding_size]
        pos_item_score = torch.mul(user_e, pos_item_e).sum(dim=1) # [batch_size]
        neg_item_score = torch.mul(user_e, neg_item_e).sum(dim=1) # [batch_size]

        loss = self.loss(pos_item_score, neg_item_score)          # []

        return loss


Implement predict()
------------------------------
At last, we define the :meth:`predict` method, which is used to compute the score for a give user-item pair.
The input is a :class:`~recbole.data.interaction.Interaction`, and the output is a score.

.. code:: python

    import torch

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        user_e = self.user_embedding(user)            # [batch_size, embedding_size]
        item_e = self.item_embedding(item)            # [batch_size, embedding_size]

        scores = torch.mul(user_e, item_e).sum(dim=1) # [batch_size]

        return scores

If you would like to evaluate the full ranking in the NewModel, RecBole also supports an accelerated predict method.

.. code:: python

   import torch

   def full_sort_predict(self, interaction):
      user = interaction[self.USER_ID]

      user_e = self.user_embedding(user)                        # [batch_size, embedding_size]
      all_item_e = self.item_embedding.weight                   # [n_items, batch_size]

      scores = torch.matmul(user_e, all_item_e.transpose(0, 1)) # [batch_size, n_items]

      return scores


This method will recall this method to accelerate the ranking.


Complete Code
------------------------
Thus the final implemented NewModel is:

.. code:: python

    import torch
    import torch.nn as nn

    from recbole.utils import InputType
    from recbole.model.abstract_recommender import GeneralRecommender
    from recbole.model.loss import BPRLoss
    from recbole.model.init import xavier_normal_initialization


    class NewModel(GeneralRecommender):

        input_type = InputType.PAIRWISE

        def __init__(self, config, dataset):
            super(NewModel, self).__init__(config, dataset)

            # load dataset info
            self.n_users = dataset.user_num
            self.n_items = dataset.item_num

            # load parameters info
            self.embedding_size = config['embedding_size']

            # define layers and loss
            self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
            self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)
            self.loss = BPRLoss()

            # parameters initialization
            self.apply(xavier_normal_initialization)

        def calculate_loss(self, interaction):
            user = interaction[self.USER_ID]
            pos_item = interaction[self.ITEM_ID]
            neg_item = interaction[self.NEG_ITEM_ID]

            user_e = self.user_embedding(user)                        # [batch_size, embedding_size]
            pos_item_e = self.item_embedding(pos_item)                # [batch_size, embedding_size]
            neg_item_e = self.item_embedding(neg_item)                # [batch_size, embedding_size]
            pos_item_score = torch.mul(user_e, pos_item_e).sum(dim=1) # [batch_size]
            neg_item_score = torch.mul(user_e, neg_item_e).sum(dim=1) # [batch_size]

            loss = self.loss(pos_item_score, neg_item_score)          # []

            return loss

        def predict(self, interaction):
            user = interaction[self.USER_ID]
            item = interaction[self.ITEM_ID]

            user_e = self.user_embedding(user)            # [batch_size, embedding_size]
            item_e = self.item_embedding(item)            # [batch_size, embedding_size]

            scores = torch.mul(user_e, item_e).sum(dim=1) # [batch_size]

            return scores

        def full_sort_predict(self, interaction):
            user = interaction[self.USER_ID]

            user_e = self.user_embedding(user)                        # [batch_size, embedding_size]
            all_item_e = self.item_embedding.weight                   # [n_items, batch_size]

            scores = torch.matmul(user_e, all_item_e.transpose(0, 1)) # [batch_size, n_items]

            return scores

Then, we can use NewModel in RecBole as follows (e.g., `run.py`):

.. code:: python

    from logging import getLogger
    from recbole.utils import init_logger, init_seed
    from recbole.trainer import Trainer
    from newmodel import NewModel
    from recbole.config import Config
    from recbole.data import create_dataset, data_preparation


    if __name__ == '__main__':

        config = Config(model=NewModel, dataset='ml-100k')
        init_seed(config['seed'], config['reproducibility'])

        # logger initialization
        init_logger(config)
        logger = getLogger()

        logger.info(config)

        # dataset filtering
        dataset = create_dataset(config)
        logger.info(dataset)

        # dataset splitting
        train_data, valid_data, test_data = data_preparation(config, dataset)

        # model loading and initialization
        model = NewModel(config, train_data.dataset).to(config['device'])
        logger.info(model)

        # trainer loading and initialization
        trainer = Trainer(config, model)

        # model training
        best_valid_score, best_valid_result = trainer.fit(train_data, valid_data)

        # model evaluation
        test_result = trainer.evaluate(test_data)

        logger.info('best valid result: {}'.format(best_valid_result))
        logger.info('test result: {}'.format(test_result))

Then, we can run NewModel:

.. code:: python

    python run.py --embedding_size=64

Note, please remember to configure the model parameters
(such as ``embedding_size``) through config files, parameter dicts or command line.
