# _*_ coding: utf-8 _*_
# @Time   : 2023/1/20
# @Author : Wanli Yang
# @Email  : 2013774@mail.nankai.edu.cn

r"""
KD_DAGFM
################################################
Reference:
    Zhen Tian et al. "Directed Acyclic Graph Factorization Machines for CTR Prediction via Knowledge Distillation."
    in WSDM 2023.
Reference code:
    https://github.com/chenyuwuxin/DAGFM
"""

import torch
from torch import nn
from torch.nn.init import xavier_normal_
from copy import deepcopy

from recbole.model.init import xavier_normal_initialization
from recbole.model.abstract_recommender import ContextRecommender


class KD_DAGFM(ContextRecommender):
    r"""KD_DAGFM is a context-based recommendation model. The model is based on directed acyclic graph and knowledge
    distillation. It can learn arbitrary feature interactions from the complex teacher networks and achieve
    approximately lossless model performance. It can also greatly reduce the computational resource costs.
    """

    def __init__(self, config, dataset):
        super(KD_DAGFM, self).__init__(config, dataset)

        # load parameters info
        self.phase = config["phase"]
        self.alpha = config["alpha"]
        self.beta = config["beta"]

        # add element to config for the initialization of teacher&student network
        config["feature_num"] = self.num_feature_field

        # initialize teacher&student network
        self.student_network = DAGFM(config)
        self.teacher_network = eval(f"{config['teacher']}")(
            self.get_teacher_config(config)
        )

        # initialize loss function
        self.loss_fn = nn.BCELoss()

        # get warm up parameters
        if self.phase != "teacher_training":
            if "warm_up" not in config:
                raise ValueError("Must have warm up!")
            else:
                save_info = torch.load(config["warm_up"])
                self.load_state_dict(save_info["state_dict"])
        else:
            self.apply(xavier_normal_initialization)

    # get config of teacher network from config
    def get_teacher_config(self, config):
        teacher_cfg = deepcopy(config)
        for key in config.final_config_dict:
            if key.startswith("t_"):
                teacher_cfg[key[2:]] = config[key]
        return teacher_cfg

    def FeatureInteraction(self, feature):
        if self.phase == "teacher_training":
            return self.teacher_network.FeatureInteraction(feature)
        elif self.phase == "distillation" or self.phase == "finetuning":
            return self.student_network.FeatureInteraction(feature)
        else:
            return ValueError("Phase invalid!")

    def forward(self, interaction):
        dagfm_all_embeddings = self.concat_embed_input_fields(
            interaction
        )  # [batch_size, num_field, embed_dim]
        if self.phase == "teacher_training" or self.phase == "finetuning":
            return self.FeatureInteraction(dagfm_all_embeddings)
        elif self.phase == "distillation":
            dagfm_all_embeddings = dagfm_all_embeddings.data
            if self.training:
                self.t_pred = self.teacher_network(dagfm_all_embeddings)
            return self.FeatureInteraction(dagfm_all_embeddings)
        else:
            raise ValueError("Phase invalid!")

    def calculate_loss(self, interaction):
        if self.phase == "teacher_training" or self.phase == "finetuning":
            prediction = self.forward(interaction)
            loss = self.loss_fn(
                prediction.squeeze(-1),
                interaction[self.LABEL].squeeze(-1).to(self.device),
            )
        elif self.phase == "distillation":
            self.teacher_network.eval()
            s_pred = self.forward(interaction)
            ctr_loss = self.loss_fn(
                s_pred.squeeze(-1), interaction[self.LABEL].squeeze(-1).to(self.device)
            )
            kd_loss = torch.mean(
                (self.teacher_network.logits.data - self.student_network.logits) ** 2
            )
            loss = self.alpha * ctr_loss + self.beta * kd_loss
        else:
            raise ValueError("Phase invalid!")
        return loss

    def predict(self, interaction):
        return self.forward(interaction)


class DAGFM(nn.Module):
    def __init__(self, config):
        super(DAGFM, self).__init__()
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # load parameters info
        self.type = config["type"]
        self.depth = config["depth"]
        field_num = config["feature_num"]
        embedding_size = config["embedding_size"]

        # initialize parameters according to the type
        if self.type == "inner":
            self.p = nn.ParameterList(
                [
                    nn.Parameter(torch.randn(field_num, field_num, embedding_size))
                    for _ in range(self.depth)
                ]
            )
            for _ in range(self.depth):
                xavier_normal_(self.p[_], gain=1.414)
        elif self.type == "outer":
            self.p = nn.ParameterList(
                [
                    nn.Parameter(torch.randn(field_num, field_num, embedding_size))
                    for _ in range(self.depth)
                ]
            )
            self.q = nn.ParameterList(
                [
                    nn.Parameter(torch.randn(field_num, field_num, embedding_size))
                    for _ in range(self.depth)
                ]
            )
            for _ in range(self.depth):
                xavier_normal_(self.p[_], gain=1.414)
                xavier_normal_(self.q[_], gain=1.414)
        self.adj_matrix = torch.zeros(field_num, field_num, embedding_size).to(
            self.device
        )
        for i in range(field_num):
            for j in range(i, field_num):
                self.adj_matrix[i, j, :] += 1
        self.connect_layer = nn.Parameter(torch.eye(field_num).float())
        self.linear = nn.Linear(field_num * (self.depth + 1), 1)

    def FeatureInteraction(self, feature):
        init_state = self.connect_layer @ feature
        h0, ht = init_state, init_state
        state = [torch.sum(init_state, dim=-1)]
        for i in range(self.depth):
            if self.type == "inner":
                aggr = torch.einsum("bfd,fsd->bsd", ht, self.p[i] * self.adj_matrix)
                ht = h0 * aggr
            elif self.type == "outer":
                term = torch.einsum("bfd,fsd->bfs", ht, self.p[i] * self.adj_matrix)
                aggr = torch.einsum("bfs,fsd->bsd", term, self.q[i])
                ht = h0 * aggr
            state.append(torch.sum(ht, dim=-1))

        state = torch.cat(state, dim=-1)
        self.logits = self.linear(state)
        self.outputs = torch.sigmoid(self.logits)
        return self.outputs


# teacher network CrossNet
class CrossNet(nn.Module):
    def __init__(self, config):
        super(CrossNet, self).__init__()

        # load parameters info
        self.depth = config["depth"]
        self.embedding_size = config["embedding_size"]
        self.feature_num = config["feature_num"]
        self.in_feature_num = self.feature_num * self.embedding_size
        self.cross_layer_w = nn.ParameterList(
            nn.Parameter(torch.randn(self.in_feature_num, self.in_feature_num))
            for _ in range(self.depth)
        )
        self.bias = nn.ParameterList(
            nn.Parameter(torch.zeros(self.in_feature_num, 1)) for _ in range(self.depth)
        )
        self.linear = nn.Linear(self.in_feature_num, 1)
        nn.init.normal_(self.linear.weight)

    def FeatureInteraction(self, x_0):
        x_0 = x_0.reshape(x_0.shape[0], -1)
        x_0 = x_0.unsqueeze(dim=2)
        x_l = x_0  # (batch_size, in_feature_num, 1)
        for i in range(self.depth):
            xl_w = torch.matmul(self.cross_layer_w[i], x_l)
            xl_w = xl_w + self.bias[i]
            xl_dot = torch.mul(x_0, xl_w)
            x_l = xl_dot + x_l
        x_l = x_l.squeeze(dim=2)
        self.logits = self.linear(x_l)
        self.outputs = torch.sigmoid(self.logits)
        return self.outputs

    def forward(self, feature):
        return self.FeatureInteraction(feature)


class CINComp(nn.Module):
    def __init__(self, indim, outdim, config):
        super(CINComp, self).__init__()
        basedim = config["feature_num"]
        self.conv = nn.Conv1d(indim * basedim, outdim, 1)

    def forward(self, feature, base):
        return self.conv(
            (feature[:, :, None, :] * base[:, None, :, :]).reshape(
                feature.shape[0], feature.shape[1] * base.shape[1], -1
            )
        )


# teacher network CIN
class CIN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cinlist = [config["feature_num"]] + config["cin"]
        self.cin = nn.ModuleList(
            [
                CINComp(self.cinlist[i], self.cinlist[i + 1], config)
                for i in range(0, len(self.cinlist) - 1)
            ]
        )
        self.linear = nn.Parameter(torch.zeros(sum(self.cinlist) - self.cinlist[0], 1))
        nn.init.normal_(self.linear, mean=0, std=0.01)
        self.backbone = ["cin", "linear"]
        self.loss_fn = nn.BCELoss()
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

    def FeatureInteraction(self, feature):
        base = feature
        x = feature
        p = []
        for comp in self.cin:
            x = comp(x, base)
            p.append(torch.sum(x, dim=-1))
        p = torch.cat(p, dim=-1)
        self.logits = p @ self.linear
        self.outputs = torch.sigmoid(self.logits)
        return self.outputs

    def forward(self, feature):
        return self.FeatureInteraction(feature)
