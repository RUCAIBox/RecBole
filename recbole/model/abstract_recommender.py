# @Time   : 2020/6/25
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn

# UPDATE:
# @Time   : 2020/8/6, 2020/8/25
# @Author : Shanlei Mu, Yupeng Hou
# @Email  : slmu@ruc.edu.cn, houyupeng@ruc.edu.cn


"""
recbole.model.abstract_recommender
##################################
"""

import numpy as np
import torch
import torch.nn as nn

from recbole.utils import ModelType, InputType, FeatureSource, FeatureType
from recbole.model.layers import FMEmbedding, FMFirstOrderLinear


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

    def full_sort_predict(self, interaction):
        r"""full sort prediction function.
        Given users, calculate the scores between users and all candidate items.

        Args:
            interaction (Interaction): Interaction class of the batch.

        Returns:
            torch.Tensor: Predicted scores for given users and all candidate items,
            shape: [n_batch_users * n_candidate_items]
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
    """This is a abstract general recommender. All the general model should implement this class.
    The base general recommender class provide the basic dataset and parameters information.
    """
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
    """
    This is a abstract sequential recommender. All the sequential model should implement This class.
    """
    type = ModelType.SEQUENTIAL

    def __init__(self, config, dataset):
        super(SequentialRecommender, self).__init__()

        # load dataset info
        self.USER_ID = config['USER_ID_FIELD']
        self.ITEM_ID = config['ITEM_ID_FIELD']
        self.ITEM_SEQ = self.ITEM_ID + config['LIST_SUFFIX']
        self.ITEM_SEQ_LEN = config['ITEM_LIST_LENGTH_FIELD']
        self.POS_ITEM_ID = self.ITEM_ID
        self.NEG_ITEM_ID = config['NEG_PREFIX'] + self.ITEM_ID
        self.max_seq_length = config['MAX_ITEM_LIST_LENGTH']
        self.n_items = dataset.num(self.ITEM_ID)

    def gather_indexes(self, output, gather_index):
        """Gathers the vectors at the spexific positions over a minibatch"""
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
        output_tensor = output.gather(dim=1, index=gather_index)
        return output_tensor.squeeze(1)


class KnowledgeRecommender(AbstractRecommender):
    """This is a abstract knowledge-based recommender. All the knowledge-based model should implement this class.
    The base knowledge-based recommender class provide the basic dataset and parameters information.
    """
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


class ContextRecommender(AbstractRecommender):
    """This is a abstract context-aware recommender. All the context-aware model should implement this class.
    The base context-aware recommender class provide the basic embedding function of feature fields which also
    contains a first-order part of feature fields.
    """
    type = ModelType.CONTEXT
    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(ContextRecommender, self).__init__()

        self.field_names = dataset.fields()
        self.LABEL = config['LABEL_FIELD']
        self.embedding_size = config['embedding_size']
        self.device = config['device']
        self.double_tower = config['double_tower']
        if self.double_tower is None:
            self.double_tower = False
        self.token_field_names = []
        self.token_field_dims = []
        self.float_field_names = []
        self.float_field_dims = []
        self.token_seq_field_names = []
        self.token_seq_field_dims = []
        self.num_feature_field = 0

        if self.double_tower:
            self.user_field_names = []
            self.item_field_names = []
            for field_name in self.field_names:
                if dataset.dataset.field2source[field_name] in {FeatureSource.USER, FeatureSource.USER_ID}:
                    self.user_field_names.append(field_name)
                elif dataset.dataset.field2source[field_name] in {FeatureSource.ITEM, FeatureSource.ITEM_ID}:
                    self.item_field_names.append(field_name)
            self.field_names = self.user_field_names + self.item_field_names
            self.user_token_field_num = 0
            self.user_float_field_num = 0
            self.user_token_seq_field_num = 0
            for field_name in self.user_field_names:
                if dataset.field2type[field_name] == FeatureType.TOKEN:
                    self.user_token_field_num += 1
                elif dataset.field2type[field_name] == FeatureType.TOKEN_SEQ:
                    self.user_token_seq_field_num += 1
                else:
                    self.user_float_field_num += dataset.num(field_name)
            self.item_token_field_num = 0
            self.item_float_field_num = 0
            self.item_token_seq_field_num = 0
            for field_name in self.item_field_names:
                if dataset.field2type[field_name] == FeatureType.TOKEN:
                    self.item_token_field_num += 1
                elif dataset.field2type[field_name] == FeatureType.TOKEN_SEQ:
                    self.item_token_seq_field_num += 1
                else:
                    self.item_float_field_num += dataset.num(field_name)

        for field_name in self.field_names:
            if field_name == self.LABEL:
                continue
            if dataset.field2type[field_name] == FeatureType.TOKEN:
                self.token_field_names.append(field_name)
                self.token_field_dims.append(dataset.num(field_name))
            elif dataset.field2type[field_name] == FeatureType.TOKEN_SEQ:
                self.token_seq_field_names.append(field_name)
                self.token_seq_field_dims.append(dataset.num(field_name))
            else:
                self.float_field_names.append(field_name)
                self.float_field_dims.append(dataset.num(field_name))
            self.num_feature_field += 1
        if len(self.token_field_dims) > 0:
            self.token_field_offsets = np.array((0, *np.cumsum(self.token_field_dims)[:-1]), dtype=np.long)
            self.token_embedding_table = FMEmbedding(self.token_field_dims, self.token_field_offsets,
                                                     self.embedding_size)
        if len(self.float_field_dims) > 0:
            self.float_embedding_table = nn.Embedding(np.sum(self.float_field_dims, dtype=np.int32),
                                                      self.embedding_size)
        if len(self.token_seq_field_dims) > 0:
            self.token_seq_embedding_table = nn.ModuleList()
            for token_seq_field_dim in self.token_seq_field_dims:
                self.token_seq_embedding_table.append(nn.Embedding(token_seq_field_dim, self.embedding_size))

        self.first_order_linear = FMFirstOrderLinear(config, dataset)

    def embed_float_fields(self, float_fields, embed=True):
        """Embed the float feature columns

        Args:
            float_fields (torch.FloatTensor): The input dense tensor. shape of [batch_size, num_float_field]
            embed (bool): Return the embedding of columns or just the columns itself. default=True

        Returns:
            torch.FloatTensor: The result embedding tensor of float columns.
        """
        # input Tensor shape : [batch_size, num_float_field]
        if not embed or float_fields is None:
            return float_fields

        num_float_field = float_fields.shape[1]
        # [batch_size, num_float_field]
        index = torch.arange(0, num_float_field).unsqueeze(0).expand_as(float_fields).long().to(self.device)

        # [batch_size, num_float_field, embed_dim]
        float_embedding = self.float_embedding_table(index)
        float_embedding = torch.mul(float_embedding, float_fields.unsqueeze(2))

        return float_embedding

    def embed_token_fields(self, token_fields):
        """Embed the token feature columns

        Args:
            token_fields (torch.LongTensor): The input tensor. shape of [batch_size, num_token_field]

        Returns:
            torch.FloatTensor: The result embedding tensor of token columns.
        """
        # input Tensor shape : [batch_size, num_token_field]
        if token_fields is None:
            return None
        # [batch_size, num_token_field, embed_dim]
        token_embedding = self.token_embedding_table(token_fields)

        return token_embedding

    def embed_token_seq_fields(self, token_seq_fields, mode='mean'):
        """Embed the token feature columns

        Args:
            token_seq_fields (torch.LongTensor): The input tensor. shape of [batch_size, seq_len]
            mode (str): How to aggregate the embedding of feature in this field. default=mean

        Returns:
            torch.FloatTensor: The result embedding tensor of token sequence columns.
        """
        # input is a list of Tensor shape of [batch_size, seq_len]
        fields_result = []
        for i, token_seq_field in enumerate(token_seq_fields):
            embedding_table = self.token_seq_embedding_table[i]
            mask = token_seq_field != 0  # [batch_size, seq_len]
            mask = mask.float()
            value_cnt = torch.sum(mask, dim=1, keepdim=True)  # [batch_size, 1]

            token_seq_embedding = embedding_table(token_seq_field)  # [batch_size, seq_len, embed_dim]

            mask = mask.unsqueeze(2).expand_as(token_seq_embedding)  # [batch_size, seq_len, embed_dim]
            if mode == 'max':
                masked_token_seq_embedding = token_seq_embedding - (1 - mask) * 1e9  # [batch_size, seq_len, embed_dim]
                result = torch.max(masked_token_seq_embedding, dim=1, keepdim=True)  # [batch_size, 1, embed_dim]
            elif mode == 'sum':
                masked_token_seq_embedding = token_seq_embedding * mask.float()
                result = torch.sum(masked_token_seq_embedding, dim=1, keepdim=True)  # [batch_size, 1, embed_dim]
            else:
                masked_token_seq_embedding = token_seq_embedding * mask.float()
                result = torch.sum(masked_token_seq_embedding, dim=1)  # [batch_size, embed_dim]
                eps = torch.FloatTensor([1e-8]).to(self.device)
                result = torch.div(result, value_cnt + eps)  # [batch_size, embed_dim]
                result = result.unsqueeze(1)  # [batch_size, 1, embed_dim]
            fields_result.append(result)
        if len(fields_result) == 0:
            return None
        else:
            return torch.cat(fields_result, dim=1)  # [batch_size, num_token_seq_field, embed_dim]

    def double_tower_embed_input_fields(self, interaction):
        """Embed the whole feature columns in a double tower way.

        Args:
            interaction (Interaction): The input data collection.

        Returns:
            torch.FloatTensor: The embedding tensor of token sequence columns in the first part.
            torch.FloatTensor: The embedding tensor of float sequence columns in the first part.
            torch.FloatTensor: The embedding tensor of token sequence columns in the second part.
            torch.FloatTensor: The embedding tensor of float sequence columns in the second part.

        """
        if not self.double_tower:
            raise RuntimeError('Please check your model hyper parameters and set \'double tower\' as True')
        sparse_embedding, dense_embedding = self.embed_input_fields(interaction)
        if dense_embedding is not None:
            first_dense_embedding, second_dense_embedding = \
                torch.split(dense_embedding, [self.user_float_field_num, self.item_float_field_num], dim=1)
        else:
            first_dense_embedding, second_dense_embedding = None, None

        if sparse_embedding is not None:
            sizes = [self.user_token_seq_field_num, self.item_token_seq_field_num,
                     self.user_token_field_num, self.item_token_field_num]
            first_token_seq_embedding, second_token_seq_embedding, first_token_embedding, second_token_embedding = \
                torch.split(sparse_embedding, sizes, dim=1)
            first_sparse_embedding = torch.cat([first_token_seq_embedding, first_token_embedding], dim=1)
            second_sparse_embedding = torch.cat([second_token_seq_embedding, second_token_embedding], dim=1)
        else:
            first_sparse_embedding, second_sparse_embedding = None, None

        return first_sparse_embedding, first_dense_embedding, second_sparse_embedding, second_dense_embedding

    def embed_input_fields(self, interaction):
        """Embed the whole feature columns.

        Args:
            interaction (Interaction): The input data collection.

        Returns:
            torch.FloatTensor: The embedding tensor of token sequence columns.
            torch.FloatTensor: The embedding tensor of float sequence columns.
        """
        float_fields = []
        for field_name in self.float_field_names:
            float_fields.append(interaction[field_name]
                                if len(interaction[field_name].shape) == 2 else interaction[field_name].unsqueeze(1))
        if len(float_fields) > 0:
            float_fields = torch.cat(float_fields, dim=1)  # [batch_size, num_float_field]
        else:
            float_fields = None
        # [batch_size, num_float_field] or [batch_size, num_float_field, embed_dim] or None
        float_fields_embedding = self.embed_float_fields(float_fields)

        token_fields = []
        for field_name in self.token_field_names:
            token_fields.append(interaction[field_name].unsqueeze(1))
        if len(token_fields) > 0:
            token_fields = torch.cat(token_fields, dim=1)  # [batch_size, num_token_field]
        else:
            token_fields = None
        # [batch_size, num_token_field, embed_dim] or None
        token_fields_embedding = self.embed_token_fields(token_fields)

        token_seq_fields = []
        for field_name in self.token_seq_field_names:
            token_seq_fields.append(interaction[field_name])
        # [batch_size, num_token_seq_field, embed_dim] or None
        token_seq_fields_embedding = self.embed_token_seq_fields(token_seq_fields)

        if token_fields_embedding is None:
            sparse_embedding = token_seq_fields_embedding
        else:
            if token_seq_fields_embedding is None:
                sparse_embedding = token_fields_embedding
            else:
                sparse_embedding = torch.cat([token_fields_embedding, token_seq_fields_embedding], dim=1)

        dense_embedding = float_fields_embedding

        # sparse_embedding shape: [batch_size, num_token_seq_field+num_token_field, embed_dim] or None
        # dense_embedding shape: [batch_size, num_float_field] or [batch_size, num_float_field, embed_dim] or None
        return sparse_embedding, dense_embedding
