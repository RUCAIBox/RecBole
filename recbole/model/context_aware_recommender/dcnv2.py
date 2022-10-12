# _*_ coding: utf-8 _*_
# @Time   : 2022/8/29
# @Author : Yifan Li
# @Email  : 295435096@qq.com

r"""
DCN V2
################################################
Reference:
    Ruoxi Wang at al. "Dcn v2: Improved deep & cross network and practical lessons for web-scale
    learning to rank systems." in WWW 2021.

Reference code:
    https://github.com/shenweichen/DeepCTR-Torch

"""

import torch
import torch.nn as nn

from recbole.model.abstract_recommender import ContextRecommender
from recbole.model.init import xavier_normal_initialization
from recbole.model.layers import MLPLayers
from recbole.model.loss import RegLoss


class DCNV2(ContextRecommender):
    r"""DCNV2 improves the cross network by extending the original weight vector to a matrix,
    significantly improves the expressiveness of DCN. It also introduces the MoE and
    low rank techniques to reduce time cost.
    """

    def __init__(self, config, dataset):
        super(DCNV2, self).__init__(config, dataset)

        # load and compute parameters info
        self.mixed = config["mixed"]
        self.structure = config["structure"]
        self.cross_layer_num = config["cross_layer_num"]
        self.embedding_size = config["embedding_size"]
        self.mlp_hidden_size = config["mlp_hidden_size"]
        self.reg_weight = config["reg_weight"]
        self.dropout_prob = config["dropout_prob"]

        if self.mixed:
            self.expert_num = config["expert_num"]
            self.low_rank = config["low_rank"]

        self.in_feature_num = self.num_feature_field * self.embedding_size

        # define cross layers and bias
        if self.mixed:
            # U: (in_feature_num, low_rank)
            self.cross_layer_u = nn.ParameterList(
                nn.Parameter(
                    torch.randn(self.expert_num, self.in_feature_num, self.low_rank)
                )
                for _ in range(self.cross_layer_num)
            )
            # V: (in_feature_num, low_rank)
            self.cross_layer_v = nn.ParameterList(
                nn.Parameter(
                    torch.randn(self.expert_num, self.in_feature_num, self.low_rank)
                )
                for _ in range(self.cross_layer_num)
            )
            # C: (low_rank, low_rank)
            self.cross_layer_c = nn.ParameterList(
                nn.Parameter(torch.randn(self.expert_num, self.low_rank, self.low_rank))
                for _ in range(self.cross_layer_num)
            )
            self.gating = nn.ModuleList(
                nn.Linear(self.in_feature_num, 1) for _ in range(self.expert_num)
            )
        else:
            # W: (in_feature_num, in_feature_num)
            self.cross_layer_w = nn.ParameterList(
                nn.Parameter(torch.randn(self.in_feature_num, self.in_feature_num))
                for _ in range(self.cross_layer_num)
            )
        # bias: (in_feature_num, 1)
        self.bias = nn.ParameterList(
            nn.Parameter(torch.zeros(self.in_feature_num, 1))
            for _ in range(self.cross_layer_num)
        )

        # define deep and predict layers
        mlp_size_list = [self.in_feature_num] + self.mlp_hidden_size
        self.mlp_layers = MLPLayers(mlp_size_list, dropout=self.dropout_prob, bn=True)
        if self.structure == "parallel":
            self.predict_layer = nn.Linear(
                self.in_feature_num + self.mlp_hidden_size[-1], 1
            )
        elif self.structure == "stacked":
            self.predict_layer = nn.Linear(self.mlp_hidden_size[-1], 1)

        # define loss and activation functions
        self.reg_loss = RegLoss()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.loss = nn.BCELoss()

        # parameters initialization
        self.apply(xavier_normal_initialization)

    def cross_network(self, x_0):
        r"""Cross network is composed of cross layers, with each layer having the following formula.

        .. math:: x_{l+1} = x_0 \odot (W_l x_l + b_l) + x_l

        :math:`x_l`, :math:`x_{l+1}` are column vectors denoting the outputs from the l -th and
        (l + 1)-th cross layers, respectively.
        :math:`W_l`, :math:`b_l` are the weight and bias parameters of the l -th layer.

        Args:
            x_0(torch.Tensor): Embedding vectors of all features, input of cross network.

        Returns:
            torch.Tensor:output of cross network, [batch_size, num_feature_field * embedding_size]
        """
        x_0 = x_0.unsqueeze(dim=2)
        x_l = x_0  # (batch_size, in_feature_num, 1)
        for i in range(self.cross_layer_num):
            xl_w = torch.matmul(self.cross_layer_w[i], x_l)
            xl_w = xl_w + self.bias[i]
            xl_dot = torch.mul(x_0, xl_w)
            x_l = xl_dot + x_l

        x_l = x_l.squeeze(dim=2)
        return x_l

    def cross_network_mix(self, x_0):
        r"""Cross network part of DCN-mix, which add MoE and nonlinear transformation in low-rank space.

        .. math::
            x_{l+1} = \sum_{i=1}^K G_i(x_l)E_i(x_l)+x_l
        .. math::
            E_i(x_l) = x_0 \odot (U_l^i \dot g(C_l^i \dot g(V_L^{iT} x_l)) + b_l)

        :math:`E_i` and :math:`G_i` represents the expert and gatings respectively,
        :math:`U_l`, :math:`C_l`, :math:`V_l` stand for low-rank decomposition of weight matrix,
        :math:`g` is the nonlinear activation function.

        Args:
            x_0(torch.Tensor): Embedding vectors of all features, input of cross network.

        Returns:
            torch.Tensor:output of mixed cross network, [batch_size, num_feature_field * embedding_size]
        """
        x_0 = x_0.unsqueeze(dim=2)
        x_l = x_0  # (batch_size, in_feature_num, 1)
        for i in range(self.cross_layer_num):
            expert_output_list = []
            gating_output_list = []
            for expert in range(self.expert_num):
                # compute gating output
                gating_output_list.append(
                    self.gating[expert](x_l.squeeze(dim=2))
                )  # (batch_size, 1)

                # project to low-rank subspace
                xl_v = torch.matmul(
                    self.cross_layer_v[i][expert].T, x_l
                )  # (batch_size, low_rank, 1)

                # nonlinear activation in subspace
                xl_c = self.tanh(xl_v)
                xl_c = torch.matmul(
                    self.cross_layer_c[i][expert], xl_c
                )  # (batch_size, low_rank, 1)
                xl_c = self.tanh(xl_c)

                # project back feature space
                xl_u = torch.matmul(
                    self.cross_layer_u[i][expert], xl_c
                )  # (batch_size, in_feature_num, 1)

                # dot with x_0
                xl_dot = xl_u + self.bias[i]
                xl_dot = torch.mul(x_0, xl_dot)

                expert_output_list.append(
                    xl_dot.squeeze(dim=2)
                )  # (batch_size, in_feature_num)

            expert_output = torch.stack(
                expert_output_list, dim=2
            )  # (batch_size, in_feature_num, expert_num)
            gating_output = torch.stack(
                gating_output_list, dim=1
            )  # (batch_size, expert_num, 1)
            moe_output = torch.matmul(
                expert_output, self.softmax(gating_output)
            )  # (batch_size, in_feature_num, 1)
            x_l = x_l + moe_output

        x_l = x_l.squeeze(dim=2)  # (batch_size, in_feature_num)
        return x_l

    def forward(self, interaction):
        dcn_all_embeddings = self.concat_embed_input_fields(
            interaction
        )  # (batch_size, num_field, embed_dim)
        batch_size = dcn_all_embeddings.shape[0]
        dcn_all_embeddings = dcn_all_embeddings.view(
            batch_size, -1
        )  # (batch_size, in_feature_num)

        if self.structure == "parallel":
            deep_output = self.mlp_layers(
                dcn_all_embeddings
            )  # (batch_size, mlp_hidden_size)
            if self.mixed:
                cross_output = self.cross_network_mix(
                    dcn_all_embeddings
                )  # (batch_size, in_feature_num)
            else:
                cross_output = self.cross_network(dcn_all_embeddings)
            concat_output = torch.cat(
                [cross_output, deep_output], dim=-1
            )  # (batch_size, in_num + mlp_size)
            output = self.sigmoid(self.predict_layer(concat_output))

            return output.squeeze(dim=1)

        elif self.structure == "stacked":
            if self.mixed:
                cross_output = self.cross_network_mix(
                    dcn_all_embeddings
                )  # (batch_size, in_feature_num)
            else:
                cross_output = self.cross_network(dcn_all_embeddings)
            deep_output = self.mlp_layers(cross_output)  # (batch_size, mlp_hidden_size)
            output = self.sigmoid(self.predict_layer(deep_output))

            return output.squeeze(dim=1)

    def calculate_loss(self, interaction):
        label = interaction[self.LABEL]
        output = self.forward(interaction)
        if self.mixed:
            reg_loss = (
                self.reg_loss(self.cross_layer_c)
                + self.reg_loss(self.cross_layer_v)
                + self.reg_loss(self.cross_layer_u)
            )
        else:
            reg_loss = self.reg_loss(self.cross_layer_w)
        l2_loss = self.reg_weight * reg_loss
        return self.loss(output, label) + l2_loss

    def predict(self, interaction):
        return self.forward(interaction)
