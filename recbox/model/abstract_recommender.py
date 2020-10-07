# @Time   : 2020/6/25
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn

# UPDATE:
# @Time   : 2020/8/6, 2020/8/25
# @Author : Shanlei Mu, Yupeng Hou
# @Email  : slmu@ruc.edu.cn, houyupeng@ruc.edu.cn

import numpy as np
import torch.nn as nn

from recbox.utils import ModelType


class AbstractRecommender(nn.Module):
    r"""Base class for all models
    """
    def calculate_loss(self, interaction):
        r"""Calculate the training loss for a batch data.

        Args:
            interaction (Interaction): Interaction class of the batch.

        Returns:
            torch.Tensor: Training loss, shape: []
        """
        raise NotImplementedError

    def predict(self, interaction):
        r"""Predict the scores between users and items.

        Args:
            interaction (Interaction): Interaction class of the batch.

        Returns:
            torch.Tensor: Predicted scores for given users and items, shape: [batch_size]
        """
        raise NotImplementedError

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)


class GeneralRecommender(AbstractRecommender):
    type = ModelType.GENERAL

    def __init__(self, config, dataset):
        super(GeneralRecommender, self).__init__()

        # load dataset info
        self.USER_ID = config['USER_ID_FIELD']
        self.ITEM_ID = config['ITEM_ID_FIELD']
        self.NEG_ITEM_ID = config['NEG_PREFIX'] + self.ITEM_ID
        self.n_users = dataset.num(self.USER_ID)
        self.n_items = dataset.num(self.ITEM_ID)

        # load parameters info
        self.batch_size = config['train_batch_size']
        self.device = config['device']


class SequentialRecommender(AbstractRecommender):
    type = ModelType.SEQUENTIAL

    def __init__(self):
        super(SequentialRecommender, self).__init__()

    def gather_indexes(self, output, gather_index):
        "Gathers the vectors at the spexific positions over a minibatch"
        gather_index = gather_index.view(-1, 1,
                                         1).expand(-1, -1, output.shape[-1])
        output_tensor = output.gather(dim=1, index=gather_index)
        return output_tensor.squeeze(1)


class KnowledgeRecommender(AbstractRecommender):
    type = ModelType.KNOWLEDGE

    def __init__(self, config, dataset):
        super(KnowledgeRecommender, self).__init__()

        # load dataset info
        self.USER_ID = config['USER_ID_FIELD']
        self.ITEM_ID = config['ITEM_ID_FIELD']
        self.NEG_ITEM_ID = config['NEG_PREFIX'] + self.ITEM_ID
        self.ENTITY_ID = config['ENTITY_ID_FIELD']
        self.RELATION_ID = config['RELATION_ID_FIELD']
        self.HEAD_ENTITY_ID = config['HEAD_ENTITY_ID_FIELD']
        self.TAIL_ENTITY_ID = config['TAIL_ENTITY_ID_FIELD']
        self.NEG_TAIL_ENTITY_ID = config['NEG_PREFIX'] + self.TAIL_ENTITY_ID
        self.n_users = dataset.num(self.USER_ID)
        self.n_items = dataset.num(self.ITEM_ID)
        self.n_entities = dataset.num(self.ENTITY_ID)
        self.n_relations = dataset.num(self.RELATION_ID)

        # load parameters info
        self.batch_size = config['train_batch_size']
        self.device = config['device']
