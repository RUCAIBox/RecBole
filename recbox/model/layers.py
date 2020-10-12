# -*- coding: utf-8 -*-
# @Time   : 2020/6/27 16:40
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn
# @File   : layers.py

# UPDATE:
# @Time   : 2020/8/24 14:58, 2020/9/16, 2020/9/21, 2020/10/9
# @Author : Yujie Lu, Xingyu Pan, Zhichao Feng, Hui Wang
# @Email  : yujielu1998@gmail.com, panxy@ruc.edu.cn, fzcbupt@gmail.com, hui.wang@ruc.edu.cn

"""
Common Layers in recommender system
"""

from logging import getLogger
import numpy as np
import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as fn
from torch.nn.init import normal_


class MLPLayers(nn.Module):
    """ MLPLayers

    Args:
        - layers(list): a list contains the size of each layer in mlp layers
        - dropout(float): probability of an element to be zeroed. Default: 0
        - activation(str): activation function after each layer in mlp layers. Default: 'relu'
                      candidates: 'sigmoid', 'tanh', 'relu', 'leekyrelu', 'none'

    Shape:
        - Input: (N, *, H_{in}) where * means any number of additional dimensions
          H_{in} must equal to the first value in `layers`
        - Output: (N, *, H_{out}) where H_{out} equals to the last value in `layers`

    Examples::

        >> m = MLPLayers([64, 32, 16], 0.2, 'relu')
        >> input = torch.randn(128, 64)
        >> output = m(input)
        >> print(output.size())
        >> torch.Size([128, 16])
    """

    def __init__(self, layers, dropout=0, activation='relu', bn=False, init_method=None):
        super(MLPLayers, self).__init__()
        self.layers = layers
        self.dropout = dropout
        self.activation = activation
        self.use_bn = bn
        self.init_method = init_method
        self.logger = getLogger()

        mlp_modules = []
        for idx, (input_size, output_size) in enumerate(zip(self.layers[:-1], self.layers[1:])):
            mlp_modules.append(nn.Dropout(p=self.dropout))
            mlp_modules.append(nn.Linear(input_size, output_size))
            if self.use_bn:
                mlp_modules.append(nn.BatchNorm1d(num_features=output_size))
            if self.activation.lower() == 'sigmoid':
                mlp_modules.append(nn.Sigmoid())
            elif self.activation.lower() == 'tanh':
                mlp_modules.append(nn.Tanh())
            elif self.activation.lower() == 'relu':
                mlp_modules.append(nn.ReLU())
            elif self.activation.lower() == 'leakyrelu':
                mlp_modules.append(nn.LeakyReLU())
            elif self.activation.lower() == 'dice':
                mlp_modules.append(Dice(output_size))
            elif self.activation.lower() == 'none':
                pass
            else:
                self.logger.warning('Received unrecognized activation function, set default activation function')
        self.mlp_layers = nn.Sequential(*mlp_modules)
        if self.init_method is not None:
            self.apply(self.init_weights)

    def init_weights(self, module):
        # We just initialize the module with normal distribution as the paper said
        if isinstance(module, nn.Linear):
            if self.init_method == 'norm':
                normal_(module.weight.data, 0, 0.01)
            if module.bias is not None:
                module.bias.data.fill_(0.0)

    def forward(self, input_feature):
        return self.mlp_layers(input_feature)


class FMEmbedding(nn.Module):
    """
        Input shape
        - A 3D tensor with shape:``(batch_size,field_size)``.

        Output shape
        - 3D tensor with shape: ``(batch_size,field_size,embed_dim)``.
    """

    def __init__(self, field_dims, offsets, embed_dim):
        super(FMEmbedding, self).__init__()
        self.embedding = nn.Embedding(sum(field_dims), embed_dim)
        self.offsets = offsets

    def forward(self, input_x):
        input_x = input_x + input_x.new_tensor(self.offsets).unsqueeze(0)
        output = self.embedding(input_x)
        return output


class BaseFactorizationMachine(nn.Module):
    """
        Input shape
        - A 3D tensor with shape:``(batch_size,field_size,embed_dim)``.

        Output shape
        - 3D tensor with shape: ``(batch_size,1)`` or (batch_size, embed_dim).
    """

    def __init__(self, reduce_sum=True):
        super(BaseFactorizationMachine, self).__init__()
        self.reduce_sum = reduce_sum

    def forward(self, input_x):
        square_of_sum = torch.sum(input_x, dim=1) ** 2
        sum_of_square = torch.sum(input_x ** 2, dim=1)
        output = square_of_sum - sum_of_square
        if self.reduce_sum:
            output = torch.sum(output, dim=1, keepdim=True)
        output = 0.5 * output
        return output


class BiGNNLayer(nn.Module):
    """Propagate a layer of Bi-interaction GNN

    .. math::
            output = (L+I)EW_1 + LE \otimes EW_2
    """

    def __init__(self, in_dim, out_dim):
        super(BiGNNLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.linear = torch.nn.Linear(in_features=in_dim, out_features=out_dim)
        self.interActTransform = torch.nn.Linear(in_features=in_dim, out_features=out_dim)

    def forward(self, lap_matrix, eye_matrix, features):
        # for GCF ajdMat is a (N+M) by (N+M) mat
        # lap_matrix L = D^-1(A)D^-1 # 拉普拉斯矩阵
        x = torch.sparse.mm(lap_matrix, features)

        inter_part1 = self.linear(features + x)
        inter_feature = torch.mul(x, features)
        inter_part2 = self.interActTransform(inter_feature)

        return inter_part1 + inter_part2


class AttLayer(nn.Module):
    """Calculate the attention signal(weight) according the input tensor.
    Args:
        infeatures (torch.FloatTensor): A 3D input tensor with shape of[batch_size, M, embed_dim].

    Returns:
        torch.FloatTensor: Attention weight of input. shape of [batch_size, M].

    """

    def __init__(self, in_dim, att_dim):
        super(AttLayer, self).__init__()
        self.in_dim = in_dim
        self.att_dim = att_dim
        self.w = torch.nn.Linear(in_features=in_dim, out_features=att_dim, bias=False)
        self.h = nn.Parameter(torch.randn(att_dim), requires_grad=True)

    def forward(self, infeatures):
        att_singal = self.w(infeatures)  # [batch_size, M, att_dim]
        att_singal = fn.relu(att_singal)  # [batch_size, M, att_dim]

        att_singal = torch.mul(att_singal, self.h)  # [batch_size, M, att_dim]
        att_singal = torch.sum(att_singal, dim=2)  # [batch_size, M]
        att_singal = fn.softmax(att_singal, dim=1)  # [batch_size, M]

        return att_singal


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_head == 0
        self.d_model = d_model
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.W_Q = nn.Linear(self.d_model, self.d_k * self.n_head, bias=False)
        self.W_K = nn.Linear(self.d_model, self.d_k * self.n_head, bias=False)
        self.W_V = nn.Linear(self.d_model, self.d_v * self.n_head, bias=False)
        self.fc = nn.Linear(self.n_head * self.d_v, self.d_model, bias=False)
        self.layernorm = nn.LayerNorm(self.d_model)

    def scale_dot_product_attention(self, Q, K, V, mask=None):
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
        if mask is not None:
            scores.masked_fill_(mask, -1e9)  # Fills elements of self tensor with value where mask is True.
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn

    def forward(self, input_Q, input_K, input_V, mask=None):
        residual, batch_size = input_Q, input_Q.size(0)

        Q = self.W_Q(input_Q).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)
        K = self.W_K(input_K).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)
        V = self.W_V(input_V).view(batch_size, -1, self.n_head, self.d_v).transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.n_head, 1, 1)

        context, attn = self.scale_dot_product_attention(Q, K, V, mask)
        context = context.transpose(1, 2).reshape(batch_size, -1, self.n_head * self.d_v)
        output = self.fc(context)
        return self.layernorm(output + residual), attn


class Dice(nn.Module):
    r"""Dice activation function

    .. math::
        f(s)=p(s) \cdot s+(1-p(s)) \cdot \alpha s,
        p(s)=\frac{1} {1 + e^{-\frac{s-E[s]} {\sqrt {Var[s] + \epsilon}}}}

    """

    def __init__(self, emb_size):
        super(Dice, self).__init__()

        self.sigmoid = nn.Sigmoid()
        self.alpha = torch.zeros((emb_size,))

    def forward(self, score):
        self.alpha = self.alpha.to(score.device)
        score_p = self.sigmoid(score)

        return self.alpha * (1 - score_p) * score + score_p * score


class SequenceAttLayer(nn.Module):
    """attention Layer. Get the representation of each user in the batch.

    Args:
        queries(torch.Tensor): candidate ads, [B, H], H means embedding_size * feat_num
        keys(torch.Tensor): user_hist, [B, T, H]
        keys_length(torch.Tensor): mask, [B]

    Returns:
        torch.Tensor: result

    """

    def __init__(self, mask_mat, att_hidden_size=(80, 40), activation='sigmoid', softmax_stag=False,
                 return_seq_weight=True):
        super(SequenceAttLayer, self).__init__()
        self.att_hidden_size = att_hidden_size
        self.activation = activation
        self.softmax_stag = softmax_stag
        self.return_seq_weight = return_seq_weight
        self.mask_mat = mask_mat
        self.att_mlp_layers = MLPLayers(self.att_hidden_size, activation='Sigmoid', bn=False)
        self.dense = nn.Linear(self.att_hidden_size[-1], 1)

    def forward(self, queries, keys, keys_length):
        embbedding_size = queries.shape[-1]  # H
        hist_len = keys.shape[1]  # T
        queries = queries.repeat(1, hist_len)

        queries = queries.view(-1, hist_len, embbedding_size)

        # MLP Layer
        input_tensor = torch.cat([queries, keys, queries - keys, queries * keys], dim=-1)
        output = self.att_mlp_layers(input_tensor)
        output = torch.transpose(self.dense(output), -1, -2)

        # get mask
        output = output.squeeze(1)
        mask = self.mask_mat.repeat(output.size(0), 1)
        mask = (mask >= keys_length.unsqueeze(1))

        # mask
        if self.softmax_stag:
            mask_value = -np.inf
        else:
            mask_value = 0.0

        output = output.masked_fill(mask=mask, value=torch.tensor(mask_value))
        output = output.unsqueeze(1)
        output = output / (embbedding_size ** 0.5)

        # get the weight of each user's history list about the target item
        if self.softmax_stag:
            output = fn.softmax(output, dim=2)  # [B, 1, T]

        if not self.return_seq_weight:
            output = torch.matmul(output, keys)  # [B, 1, H]

        return output

# Transformer Layers
# Adapted from https://github.com/huggingface/transformers/blob/master/src/transformers/modeling_bert.py
def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different
        (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) *
        (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": fn.relu, "swish": swish}


class SelfAttention(nn.Module):
    def __init__(self, config):
        super(SelfAttention, self).__init__()
        if config['hidden_size'] % config['num_attention_heads'] != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config['hidden_size'], config['num_attention_heads']))

        self.num_attention_heads = config['num_attention_heads']
        self.attention_head_size = int(config['hidden_size'] / config['num_attention_heads'])
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config['hidden_size'], self.all_head_size)
        self.key = nn.Linear(config['hidden_size'], self.all_head_size)
        self.value = nn.Linear(config['hidden_size'], self.all_head_size)

        self.attn_dropout = nn.Dropout(config['dropout_prob'])

        self.dense = nn.Linear(config['hidden_size'], config['hidden_size'])
        self.LayerNorm = nn.LayerNorm(config['hidden_size'], eps=1e-12)
        self.out_dropout = nn.Dropout(config['dropout_prob'])

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_tensor, attention_mask):
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # [batch_size heads seq_len seq_len] scores
        # [batch_size 1 1 seq_len]
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.

        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class Intermediate(nn.Module):
    def __init__(self, config):
        super(Intermediate, self).__init__()
        self.dense_1 = nn.Linear(config['hidden_size'], config['hidden_size'] * 4)
        if isinstance(config['hidden_act'], str):
            self.intermediate_act_fn = ACT2FN[config['hidden_act']]
        else:
            self.intermediate_act_fn = config['hidden_act']

        self.dense_2 = nn.Linear(config['hidden_size'] * 4, config['hidden_size'])
        self.LayerNorm = nn.LayerNorm(config['hidden_size'], eps=1e-12)
        self.dropout = nn.Dropout(config['dropout_prob'])

    def forward(self, input_tensor):
        hidden_states = self.dense_1(input_tensor)
        hidden_states = self.intermediate_act_fn(hidden_states)

        hidden_states = self.dense_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class TransformerLayer(nn.Module):
    def __init__(self, config):
        super(TransformerLayer, self).__init__()
        self.attention = SelfAttention(config)
        self.intermediate = Intermediate(config)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        return intermediate_output


class TransformerEncoder(nn.Module):
    def __init__(self, config):
        super(TransformerEncoder, self).__init__()
        layer = TransformerLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer)
                                    for _ in range(config['num_hidden_layers'])])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers