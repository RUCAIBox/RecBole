import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_normal_, constant_

from recbox.model.layers import MLPLayers
from recbox.model.loss import RegLoss
from recbox.model.context_aware_recommender.context_recommender import ContextRecommender


class DCN(ContextRecommender):
    def __init__(self, config, dataset):
        super(DCN, self).__init__(config, dataset)

        self.LABEL = config['LABEL_FIELD']
        self.mlp_hidden_size = config['mlp_hidden_size']
        self.cross_layer_num = config['cross_layer_num']
        self.weight_decay = config['weight_decay']
        self.dropout = config['dropout']

        self.cross_layer_parameter = [
            nn.Parameter(
                torch.empty(self.num_feature_field * self.embedding_size,
                            device=self.device))
            for _ in range(self.cross_layer_num * 2)
        ]

        self.cross_layer_w = nn.ParameterList(
            self.cross_layer_parameter[:self.cross_layer_num])
        self.cross_layer_b = nn.ParameterList(
            self.cross_layer_parameter[self.cross_layer_num:])

        size_list = [self.embedding_size * self.num_feature_field
                     ] + self.mlp_hidden_size
        in_feature_num = self.embedding_size * self.num_feature_field + self.mlp_hidden_size[
            -1]
        self.mlp_layers = MLPLayers(size_list, dropout=self.dropout, bn=True)
        self.deep_predict_layer = nn.Linear(in_feature_num, 1, bias=True)
        self.reg_loss = RegLoss()
        self.sigmoid = nn.Sigmoid()
        self.loss = nn.BCELoss()

        self.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)

    def cross_layer(self, x_0):
        x_l = x_0
        for i in range(self.cross_layer_num):
            xl_w = torch.tensordot(x_l, self.cross_layer_w[i], dims=([1], [0]))
            xl_dot = (x_0.transpose(0, 1) * xl_w).transpose(0, 1)
            x_l = xl_dot + self.cross_layer_b[i] + x_l
        return x_l

    # def cross_layer(self, x_0):
    #     x_0 = x_0.unsqueeze(2)
    #     x_l = x_0
    #     for i in range(self.cross_layer_num):
    #         xl_w = torch.tensordot(x_l, self.cross_layer_w[i], dims=([1], [0]))
    #         dot_ = torch.matmul(x_0, xl_w)
    #         x_l = dot_ + self.cross_layer_b[i] + x_l
    #     x_l = torch.squeeze(x_l, dim=2)
    #     return x_l

    def forward(self, interaction):
        # sparse_embedding shape: [batch_size, num_token_seq_field+num_token_field, embed_dim] or None
        # dense_embedding shape: [batch_size, num_float_field] or [batch_size, num_float_field, embed_dim] or None
        sparse_embedding, dense_embedding = self.embed_input_fields(
            interaction)
        all_embeddings = []
        if sparse_embedding is not None:
            all_embeddings.append(sparse_embedding)
        if dense_embedding is not None and len(dense_embedding.shape) == 3:
            all_embeddings.append(dense_embedding)

        dcn_all_embeddings = torch.cat(
            all_embeddings, dim=1)  # [batch_size, num_field, embed_dim]
        batch_size = dcn_all_embeddings.shape[0]
        dcn_all_embeddings = dcn_all_embeddings.view(batch_size, -1)

        deep_output = self.mlp_layers(dcn_all_embeddings)
        cross_output = self.cross_layer(dcn_all_embeddings)
        stack = torch.cat([cross_output, deep_output], dim=-1)
        output = self.sigmoid(self.deep_predict_layer(stack))

        return output.squeeze(1)

    def calculate_loss(self, interaction):
        label = interaction[self.LABEL]
        output = self.forward(interaction)
        l2_loss = self.weight_decay * self.reg_loss(self.cross_layer_w)
        return self.loss(output, label) + l2_loss

    def predict(self, interaction):
        return self.forward(interaction)
