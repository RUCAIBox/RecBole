# _*_ coding: utf-8 _*_
# @Time : 2020/10/13
# @Author : Zhichao Feng
# @Email  : fzcbupt@gmail.com

# UPDATE
# @Time   : 2020/10/21
# @Author : Zhichao Feng
# @email  : fzcbupt@gmail.com

r"""
xDeepFM
################################################
Reference:
    Jianxun Lian at al. "xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems."
    in SIGKDD 2018.

Reference code:
    - https://github.com/Leavingseason/xDeepFM
    - https://github.com/shenweichen/DeepCTR-Torch
"""

import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_, constant_

from recbole.model.abstract_recommender import ContextRecommender
from recbole.model.layers import MLPLayers, activation_layer


class xDeepFM(ContextRecommender):
    """xDeepFM combines a CIN (Compressed Interaction Network) with a classical DNN.
    The model is able to learn certain bounded-degree feature interactions explicitly;
    Besides, it can also learn arbitrary low- and high-order feature interactions implicitly.
    """

    def __init__(self, config, dataset):
        super(xDeepFM, self).__init__(config, dataset)

        # load parameters info
        self.mlp_hidden_size = config["mlp_hidden_size"]
        self.reg_weight = config["reg_weight"]
        self.dropout_prob = config["dropout_prob"]
        self.direct = config["direct"]
        self.cin_layer_size = temp_cin_size = list(config["cin_layer_size"])

        # Check whether the size of the CIN layer is legal.
        if not self.direct:
            self.cin_layer_size = list(map(lambda x: int(x // 2 * 2), temp_cin_size))
            if self.cin_layer_size[:-1] != temp_cin_size[:-1]:
                self.logger.warning(
                    "Layer size of CIN should be even except for the last layer when direct is True."
                    "It is changed to {}".format(self.cin_layer_size)
                )

        # Create a convolutional layer for each CIN layer
        self.conv1d_list = nn.ModuleList()
        self.field_nums = [self.num_feature_field]
        for i, layer_size in enumerate(self.cin_layer_size):
            conv1d = nn.Conv1d(self.field_nums[-1] * self.field_nums[0], layer_size, 1)
            self.conv1d_list.append(conv1d)
            if self.direct:
                self.field_nums.append(layer_size)
            else:
                self.field_nums.append(layer_size // 2)

        # Create MLP layer
        size_list = (
            [self.embedding_size * self.num_feature_field] + self.mlp_hidden_size + [1]
        )
        self.mlp_layers = MLPLayers(size_list, dropout=self.dropout_prob)

        # Get the output size of CIN
        if self.direct:
            self.final_len = sum(self.cin_layer_size)
        else:
            self.final_len = (
                sum(self.cin_layer_size[:-1]) // 2 + self.cin_layer_size[-1]
            )

        self.cin_linear = nn.Linear(self.final_len, 1)
        self.sigmoid = nn.Sigmoid()
        self.loss = nn.BCEWithLogitsLoss()
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding) or isinstance(module, nn.Conv1d):
            xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)

    def reg_loss(self, parameters):
        """Calculate the L2 normalization loss of parameters in a certain layer.

        Returns:
            loss(torch.FloatTensor): The L2 Loss tensor. shape of [1,]
        """
        reg_loss = 0
        for name, parm in parameters:
            if name.endswith("weight"):
                reg_loss = reg_loss + parm.norm(2)
        return reg_loss

    def calculate_reg_loss(self):
        """Calculate the final L2 normalization loss of model parameters.
        Including weight matrices of mlp layers, linear layer and convolutional layers.

        Returns:
            loss(torch.FloatTensor): The L2 Loss tensor. shape of [1,]
        """
        l2_reg = 0
        l2_reg = l2_reg + self.reg_loss(self.mlp_layers.named_parameters())
        l2_reg = l2_reg + self.reg_loss(self.first_order_linear.named_parameters())
        for conv1d in self.conv1d_list:
            l2_reg += self.reg_loss(conv1d.named_parameters())
        return l2_reg

    def compressed_interaction_network(self, input_features, activation="ReLU"):
        r"""For k-th CIN layer, the output :math:`X_k` is calculated via

        .. math::
            x_{h,*}^{k} = \sum_{i=1}^{H_k-1} \sum_{j=1}^{m}W_{i,j}^{k,h}(X_{i,*}^{k-1} \circ x_{j,*}^0)

        :math:`H_k` donates the number of feature vectors in the k-th layer,
        :math:`1 \le h \le H_k`.
        :math:`\circ` donates the Hadamard product.

        And Then, We apply sum pooling on each feature map of the hidden layer.
        Finally, All pooling vectors from hidden layers are concatenated.

        Args:
            input_features(torch.Tensor): [batch_size, field_num, embed_dim]. Embedding vectors of all features.
            activation(str): name of activation function.

        Returns:
            torch.Tensor: [batch_size, num_feature_field * embedding_size]. output of CIN layer.
        """
        batch_size, _, embedding_size = input_features.shape
        hidden_nn_layers = [input_features]
        final_result = []
        for i, layer_size in enumerate(self.cin_layer_size):
            z_i = torch.einsum(
                "bhd,bmd->bhmd", hidden_nn_layers[-1], hidden_nn_layers[0]
            )
            z_i = z_i.view(
                batch_size, self.field_nums[0] * self.field_nums[i], embedding_size
            )
            z_i = self.conv1d_list[i](z_i)

            # Pass the CIN intermediate result through the activation function.
            if activation.lower() == "identity":
                output = z_i
            else:
                activate_func = activation_layer(activation)
                if activate_func is None:
                    output = z_i
                else:
                    output = activate_func(z_i)

            # Get the output of the hidden layer.
            if self.direct:
                direct_connect = output
                next_hidden = output
            else:
                if i != len(self.cin_layer_size) - 1:
                    next_hidden, direct_connect = torch.split(
                        output, 2 * [layer_size // 2], 1
                    )
                else:
                    direct_connect = output
                    next_hidden = 0

            final_result.append(direct_connect)
            hidden_nn_layers.append(next_hidden)
        result = torch.cat(final_result, dim=1)
        result = torch.sum(result, -1)
        return result

    def forward(self, interaction):
        # Get the output of CIN.
        xdeepfm_input = self.concat_embed_input_fields(
            interaction
        )  # [batch_size, num_field, embed_dim]
        cin_output = self.compressed_interaction_network(xdeepfm_input)
        cin_output = self.cin_linear(cin_output)

        # Get the output of MLP layer.
        batch_size = xdeepfm_input.shape[0]
        dnn_output = self.mlp_layers(xdeepfm_input.view(batch_size, -1))

        # Get predicted score.
        y_p = self.first_order_linear(interaction) + cin_output + dnn_output

        return y_p.squeeze(1)

    def calculate_loss(self, interaction):
        label = interaction[self.LABEL]
        output = self.forward(interaction)
        l2_reg = self.calculate_reg_loss()
        return self.loss(output, label) + self.reg_weight * l2_reg

    def predict(self, interaction):
        return self.sigmoid(self.forward(interaction))
