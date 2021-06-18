import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_normal_

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.loss import BPRLoss
from recbole.utils import InputType

torch.autograd.set_detect_anomaly(True)


class SINE(SequentialRecommender):
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(SINE, self).__init__(config, dataset)

        # load dataset info
        self.n_users = dataset.user_num
        self.n_items = dataset.item_num

        # load parameters info
        self.device = config["device"]
        self.embedding_size = config['embedding_size']
        self.loss_type = config['loss_type']
        self.layer_norm_eps = config['layer_norm_eps']

        if self.loss_type == 'BPR':
            self.loss_fct = BPRLoss()
        elif self.loss_type == 'CE':
            self.loss_fct = nn.CrossEntropyLoss()
        elif self.loss_type == 'NLL':
            self.loss_fct = nn.NLLLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE', 'NLL']!")

        self.D = config['embedding_size']
        self.L = config['prototype_size']  # 500 for movie-len dataset
        self.k = config['interest_size']  # 4 for movie-len dataset
        self.M = self.n_items
        self.tau = config['tau_ratio']  # 0.1 in paper

        self.initializer_range = 0.01
        self.w1 = torch.tensor(np.random.normal(0, self.initializer_range, (self.D, self.D)), dtype=torch.float32,
                               requires_grad=True).to(
            self.device)
        self.w2 = torch.tensor(np.random.normal(0, self.initializer_range, self.D), dtype=torch.float32,
                               requires_grad=True).to(
            self.device)
        self.w3 = torch.tensor(np.random.normal(0, self.initializer_range, (self.D, self.D)), dtype=torch.float32,
                               requires_grad=True).to(
            self.device)
        self.w4 = torch.tensor(np.random.normal(0, self.initializer_range, (self.D, self.D)), dtype=torch.float32,
                               requires_grad=True).to(self.device)
        self.w5 = torch.tensor(np.random.normal(0, self.initializer_range, self.D), dtype=torch.float32,
                               requires_grad=True).to(self.device)

        self.C = nn.Embedding(self.L, self.D)

        self.w_k_1 = torch.tensor(np.random.normal(0, self.initializer_range, (self.k, self.D, self.D)),
                                  dtype=torch.float32,
                                  requires_grad=True).to(self.device)
        self.w_k_2 = torch.tensor(np.random.normal(0, self.initializer_range, (self.k, self.D)), dtype=torch.float32,
                                  requires_grad=True).to(self.device)
        self.item_embedding = nn.Embedding(self.M, self.D, padding_idx=0)
        self.ln2 = nn.LayerNorm(self.embedding_size, eps=self.layer_norm_eps)
        self.ln4 = nn.LayerNorm(self.embedding_size, eps=self.layer_norm_eps)

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]

        if self.loss_type == 'BPR':
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [B]
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # [B]
            loss = self.loss_fct(pos_score, neg_score)
            return loss
        elif self.loss_type == 'CE':
            test_item_emb = self.item_embedding.weight
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items)
            return loss
        else:
            test_item_emb = self.item_embedding.weight
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            logits = F.log_softmax(logits, dim=1)
            loss = self.loss_fct(logits, pos_items)
            return loss + self.calculate_reg_loss() / 2

    def calculate_reg_loss(self):
        C_mean = torch.mean(self.C.weight, dim=1, keepdim=True)
        C_reg = (self.C.weight - C_mean)
        C_reg = C_reg.matmul(C_reg.T) / self.D
        return (torch.norm(C_reg) ** 2 - torch.norm(torch.diag(C_reg)) ** 2) / 2

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq, item_seq_len)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        return scores

    def forward(self, item_seq, item_seq_len):
        x_u = self.item_embedding(item_seq).to(self.device)  # [B, N, D]

        # concept activation
        # sort by inner product
        x = torch.matmul(x_u, self.w1)
        x = torch.tanh(x)
        x = torch.matmul(x, self.w2)
        a = F.softmax(x, dim=1)
        z_u = torch.matmul(a.unsqueeze(2).transpose(1, 2), x_u).transpose(1, 2)
        s_u = torch.matmul(self.C.weight, z_u)
        s_u = s_u.squeeze(2)
        idx = s_u.argsort(1)[:, -self.k:]
        s_u_idx = s_u.sort(1)[0][:, -self.k:]
        c_u = self.C(idx)
        sigs = torch.sigmoid(s_u_idx.unsqueeze(2).repeat(1, 1, self.embedding_size))
        C_u = c_u.mul(sigs)

        # intention assignment
        # use matrix multiplication instead of cos()
        w3_x_u = x_u.matmul(self.w3)
        w3_x_u_norm = F.normalize(w3_x_u, p=2, dim=2)
        C_u_norm = self.ln2(C_u)
        P_k_t = torch.bmm(w3_x_u_norm, C_u_norm.transpose(1, 2))
        P_k_t_b = F.softmax(P_k_t, dim=2)
        P_k_t_b_t = P_k_t_b.transpose(1, 2)

        # attention weighting
        a_k = x_u.unsqueeze(1).repeat(1, self.k, 1, 1).matmul(self.w_k_1)
        P_t_k = F.softmax(torch.tanh(a_k).matmul(self.w_k_2.reshape(self.k, self.embedding_size, 1)).squeeze(3), dim=2)

        # interest embedding generation
        mul_p = P_k_t_b_t.mul(P_t_k)
        x_u_re = x_u.unsqueeze(1).repeat(1, self.k, 1, 1)
        mul_p_re = mul_p.unsqueeze(3).repeat(1, 1, 1, self.embedding_size)
        delta_k = x_u_re.mul(mul_p_re).sum(2)
        delta_k = F.normalize(delta_k, p=2, dim=2)

        # prototype sequence
        x_u_bar = P_k_t_b.matmul(C_u)
        C_apt = F.softmax(torch.tanh(x_u_bar.matmul(self.w4)).matmul(self.w5), dim=1).reshape(-1, 1, self.max_seq_length).matmul(x_u_bar)
        C_apt = self.ln4(C_apt)

        # aggregation weight
        e_k = delta_k.bmm(C_apt.reshape(-1, self.embedding_size, 1)) / self.tau
        e_k_u = F.softmax(e_k.squeeze(2), dim=1)
        v_u = e_k_u.unsqueeze(2).repeat(1, 1, self.embedding_size).mul(delta_k).sum(dim=1)

        return v_u

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B, n_items]
        return scores
