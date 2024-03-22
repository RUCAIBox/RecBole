# -*- coding: utf-8 -*-
# @Time    : 2023/10/27
# @Author  : Kesha Ou
# @Email   : keishaou@gmail.com

r"""
FEARec
################################################

Reference:
    Xinyu Du et al. "Frequency Enhanced Hybrid Attention Network for Sequential Recommendation."
    In SIGIR 2023.

Reference code:
    https://github.com/sudaada/FEARec

"""


import random

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import math

import torch.nn.functional as fn


from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.loss import BPRLoss
from recbole.data.interaction import Interaction


class FEARec(SequentialRecommender):
    def __init__(self, config, dataset):
        super(FEARec, self).__init__(config, dataset)

        # load parameters info
        self.dataset = dataset
        self.config = config
        self.n_layers = config["n_layers"]
        self.n_heads = config["n_heads"]
        self.hidden_size = config["hidden_size"]  # same as embedding_size
        self.inner_size = config[
            "inner_size"
        ]  # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = config["hidden_dropout_prob"]
        self.attn_dropout_prob = config["attn_dropout_prob"]
        self.hidden_act = config["hidden_act"]
        self.layer_norm_eps = config["layer_norm_eps"]

        self.lmd = config["lmd"]
        self.lmd_sem = config["lmd_sem"]

        self.initializer_range = config["initializer_range"]
        self.loss_type = config["loss_type"]
        self.same_item_index = self.get_same_item_index(dataset)

        # define layers and loss
        self.item_embedding = nn.Embedding(
            self.n_items, self.hidden_size, padding_idx=0
        )
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        self.item_encoder = FEAEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
            config=self.config,
        )

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        if self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        elif self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        self.ssl = config["contrast"]
        self.tau = config["tau"]
        self.sim = config["sim"]
        self.fredom = config["fredom"]
        self.fredom_type = config["fredom_type"]
        self.batch_size = config["train_batch_size"]
        self.mask_default = self.mask_correlated_samples(batch_size=self.batch_size)
        self.aug_nce_fct = nn.CrossEntropyLoss()
        self.sem_aug_nce_fct = nn.CrossEntropyLoss()

        # parameters initialization
        self.apply(self._init_weights)

    def get_same_item_index(self, dataset):
        same_target_index = {}
        target_item = dataset.inter_feat[self.ITEM_ID].numpy()

        for index, item_id in enumerate(target_item):
            all_index_same_id = np.where(target_item == item_id)[0]
            same_target_index[item_id] = all_index_same_id

        return same_target_index

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            # module.weight.data = self.truncated_normal_(tensor=module.weight.data, mean=0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def truncated_normal_(self, tensor, mean=0, std=0.09):
        with torch.no_grad():
            size = tensor.shape
            tmp = tensor.new_empty(size + (4,)).normal_()
            valid = (tmp < 2) & (tmp > -2)
            ind = valid.max(-1, keepdim=True)[1]
            tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
            tensor.data.mul_(std).add_(mean)
            return tensor

    def get_attention_mask(self, item_seq):
        """Generate left-to-right uni-directional attention mask for multi-head attention."""
        attention_mask = (item_seq > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(
            2
        )  # torch.int64
        # mask for left-to-right unidirectional
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long().to(item_seq.device)

        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype
        )  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def get_bi_attention_mask(self, item_seq):
        """Generate bidirectional attention mask for multi-head attention."""
        attention_mask = (item_seq > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(
            2
        )  # torch.int64
        # bidirectional mask
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype
        )  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def forward(self, item_seq, item_seq_len):
        position_ids = torch.arange(
            item_seq.size(1), dtype=torch.long, device=item_seq.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        item_emb = self.item_embedding(item_seq)
        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)
        # extended_attention_mask = self.get_bi_attention_mask(item_seq)

        trm_output = self.item_encoder(
            input_emb, extended_attention_mask, output_all_encoded_layers=True
        )
        output = trm_output[-1]
        output = self.gather_indexes(output, item_seq_len - 1)

        return output  # [B H]

    @staticmethod
    def alignment(x, y):
        x, y = F.normalize(x, dim=-1), F.normalize(y, dim=-1)
        return (x - y).norm(p=2, dim=1).pow(2).mean()

    @staticmethod
    def uniformity(x):
        x = F.normalize(x, dim=-1)
        x = abs(x)
        return torch.pdist(x, p=2).pow(2).mul(-2).exp().mean().log()

    def calculate_loss(self, interaction):
        same_item_index = self.same_item_index
        sem_pos_lengths = []
        sem_pos_seqs = []
        dataset = self.dataset
        target_items = interaction[self.ITEM_ID]
        for i, item_id in enumerate(target_items):
            item_id = item_id.item()
            targets_index = same_item_index[item_id]
            lens = len(targets_index)
            if lens == 0:
                print("error")
            remaining_indices = targets_index.copy()
            while len(remaining_indices) > 0:
                sample_index = random.choice(remaining_indices)
                remaining_indices = remaining_indices[remaining_indices != sample_index]
                cur_item_list = interaction[self.ITEM_SEQ][i].to("cpu")
                sample_item_list = dataset.inter_feat[self.ITEM_SEQ][sample_index]
                are_equal = torch.equal(cur_item_list, sample_item_list)
                sample_item_length = dataset.inter_feat[self.ITEM_SEQ_LEN][sample_index]

                if not are_equal or len(remaining_indices) == 0:
                    sem_pos_lengths.append(sample_item_length)
                    sem_pos_seqs.append(sample_item_list)
                    break

        sem_pos_lengths = torch.stack(sem_pos_lengths).to(self.device)
        sem_pos_seqs = torch.stack(sem_pos_seqs).to(self.device)

        interaction.update(
            Interaction({"sem_aug": sem_pos_seqs, "sem_aug_lengths": sem_pos_lengths})
        )

        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]
        if self.loss_type == "BPR":
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [B]
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # [B]
            loss = self.loss_fct(pos_score, neg_score)
        else:  # self.loss_type = 'CE'
            test_item_emb = self.item_embedding.weight
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items)

        # Unsupervised NCE
        if self.ssl in ["us", "un"]:
            aug_seq_output = self.forward(item_seq, item_seq_len)
            nce_logits, nce_labels = self.info_nce(
                seq_output,
                aug_seq_output,
                temp=self.tau,
                batch_size=item_seq_len.shape[0],
                sim=self.sim,
            )

            loss += self.lmd * self.aug_nce_fct(nce_logits, nce_labels)

        # Supervised NCE
        if self.ssl in ["us", "su"]:
            sem_aug, sem_aug_lengths = (
                interaction["sem_aug"],
                interaction["sem_aug_lengths"],
            )
            sem_aug_seq_output = self.forward(sem_aug, sem_aug_lengths)

            sem_nce_logits, sem_nce_labels = self.info_nce(
                seq_output,
                sem_aug_seq_output,
                temp=self.tau,
                batch_size=item_seq_len.shape[0],
                sim=self.sim,
            )

            loss += self.lmd_sem * self.aug_nce_fct(sem_nce_logits, sem_nce_labels)

        if self.ssl == "us_x":
            aug_seq_output = self.forward(item_seq, item_seq_len)
            sem_aug, sem_aug_lengths = (
                interaction["sem_aug"],
                interaction["sem_aug_lengths"],
            )

            sem_aug_seq_output = self.forward(sem_aug, sem_aug_lengths)
            sem_nce_logits, sem_nce_labels = self.info_nce(
                aug_seq_output,
                sem_aug_seq_output,
                temp=self.tau,
                batch_size=item_seq_len.shape[0],
                sim=self.sim,
            )

            loss += self.lmd_sem * self.aug_nce_fct(sem_nce_logits, sem_nce_labels)

            # frequency domain loss
            if self.fredom:
                seq_output_f = torch.fft.rfft(seq_output, dim=1, norm="ortho")
                aug_seq_output_f = torch.fft.rfft(aug_seq_output, dim=1, norm="ortho")
                sem_aug_seq_output_f = torch.fft.rfft(
                    sem_aug_seq_output, dim=1, norm="ortho"
                )
                if self.fredom_type in ["us", "un"]:
                    loss += 0.1 * abs(seq_output_f - aug_seq_output_f).flatten().mean()
                if self.fredom_type in ["us", "su"]:
                    loss += (
                        0.1 * abs(seq_output_f - sem_aug_seq_output_f).flatten().mean()
                    )
                if self.fredom_type == "us_x":
                    loss += (
                        0.1
                        * abs(aug_seq_output_f - sem_aug_seq_output_f).flatten().mean()
                    )

        return loss

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def info_nce(self, z_i, z_j, temp, batch_size, sim="dot"):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        N = 2 * batch_size

        z = torch.cat((z_i, z_j), dim=0)

        if sim == "cos":
            sim = (
                nn.functional.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)
                / temp
            )
        elif sim == "dot":
            sim = torch.mm(z, z.T) / temp

        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        if batch_size != self.batch_size:
            mask = self.mask_correlated_samples(batch_size)
        else:
            mask = self.mask_default
        negative_samples = sim[mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        return logits, labels

    def decompose(self, z_i, z_j, origin_z, batch_size):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        N = 2 * batch_size

        z = torch.cat((z_i, z_j), dim=0)

        # pairwise l2 distace
        sim = torch.cdist(z, z, p=2)

        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        alignment = positive_samples.mean()

        # pairwise l2 distace
        sim = torch.cdist(origin_z, origin_z, p=2)
        mask = torch.ones((batch_size, batch_size), dtype=bool)
        mask = mask.fill_diagonal_(0)
        negative_samples = sim[mask].reshape(batch_size, -1)
        uniformity = torch.log(torch.exp(-2 * negative_samples).mean())

        return alignment, uniformity

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq, item_seq_len)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B n_items]
        return scores


class HybridAttention(nn.Module):
    """
    Hybrid Attention layer: combine time domain self-attention layer and frequency domain attention layer.

    Args:
        input_tensor (torch.Tensor): the input of the multi-head Hybrid Attention layer
        attention_mask (torch.Tensor): the attention mask for input tensor

    Returns:
        hidden_states (torch.Tensor): the output of the multi-head Hybrid Attention layer

    """

    def __init__(
        self,
        n_heads,
        hidden_size,
        hidden_dropout_prob,
        attn_dropout_prob,
        layer_norm_eps,
        i,
        config,
    ):
        super(HybridAttention, self).__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, n_heads)
            )

        self.factor = config["topk_factor"]
        self.scale = None
        self.mask_flag = True
        self.output_attention = False
        self.dropout = nn.Dropout(0.1)
        self.config = config
        self.num_attention_heads = n_heads
        self.attention_head_size = int(hidden_size / n_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query_layer = nn.Linear(hidden_size, self.all_head_size)
        self.key_layer = nn.Linear(hidden_size, self.all_head_size)
        self.value_layer = nn.Linear(hidden_size, self.all_head_size)
        self.attn_dropout = nn.Dropout(attn_dropout_prob)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)
        self.filter_mixer = None
        self.global_ratio = config["global_ratio"]
        self.n_layers = config["n_layers"]
        if self.global_ratio > (1 / self.n_layers):
            print(
                "{}>{}:{}".format(
                    self.global_ratio,
                    1 / self.n_layers,
                    self.global_ratio > (1 / self.n_layers),
                )
            )
            self.filter_mixer = "G"
        else:
            print(
                "{}>{}:{}".format(
                    self.global_ratio,
                    1 / self.n_layers,
                    self.global_ratio > (1 / self.n_layers),
                )
            )
            self.filter_mixer = "L"
        self.max_item_list_length = config["MAX_ITEM_LIST_LENGTH"]
        self.dual_domain = config["dual_domain"]
        self.slide_step = (
            (self.max_item_list_length // 2 + 1) * (1 - self.global_ratio)
        ) // (self.n_layers - 1)
        self.local_ratio = 1 / self.n_layers
        self.filter_size = self.local_ratio * (self.max_item_list_length // 2 + 1)

        if self.filter_mixer == "G":
            self.w = self.global_ratio
            self.s = self.slide_step

        if self.filter_mixer == "L":
            self.w = self.local_ratio
            self.s = self.filter_size

        self.left = int(
            ((self.max_item_list_length // 2 + 1) * (1 - self.w)) - (i * self.s)
        )
        self.right = int((self.max_item_list_length // 2 + 1) - i * self.s)

        self.q_index = list(range(self.left, self.right))
        self.k_index = list(range(self.left, self.right))
        self.v_index = list(range(self.left, self.right))
        # if sample in time domain
        self.std = config["std"]
        if self.std:
            self.time_q_index = self.q_index
            self.time_k_index = self.k_index
            self.time_v_index = self.v_index
        else:
            self.time_q_index = list(range(self.max_item_list_length // 2 + 1))
            self.time_k_index = list(range(self.max_item_list_length // 2 + 1))
            self.time_v_index = list(range(self.max_item_list_length // 2 + 1))

        print("modes_q={}, index_q={}".format(len(self.q_index), self.q_index))
        print("modes_k={}, index_k={}".format(len(self.k_index), self.k_index))
        print("modes_v={}, index_v={}".format(len(self.v_index), self.v_index))

        if self.config["dual_domain"]:
            self.spatial_ratio = self.config["spatial_ratio"]

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        # return x.permute(0, 2, 1, 3)
        return x

    def time_delay_agg_training(self, values, corr):
        """
        SpeedUp version of Autocorrelation (a batch-normalization style design)
        This is for the training phase.
        """
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        # find top k
        top_k = int(self.factor * math.log(length))
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)
        index = torch.topk(torch.mean(mean_value, dim=0), top_k, dim=-1)[1]
        weights = torch.stack([mean_value[:, index[i]] for i in range(top_k)], dim=-1)
        # update corr
        tmp_corr = torch.softmax(weights, dim=-1)
        # aggregation
        tmp_values = values
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            pattern = torch.roll(tmp_values, -int(index[i]), -1)
            delays_agg = delays_agg + pattern * (
                tmp_corr[:, i]
                .unsqueeze(1)
                .unsqueeze(1)
                .unsqueeze(1)
                .repeat(1, head, channel, length)
            )
        return delays_agg

    def time_delay_agg_inference(self, values, corr):
        """
        SpeedUp version of Autocorrelation (a batch-normalization style design)
        This is for the inference phase.
        """
        batch = values.shape[0]
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        # index init
        init_index = (
            torch.arange(length)
            .unsqueeze(0)
            .unsqueeze(0)
            .unsqueeze(0)
            .repeat(batch, head, channel, 1)
            .to(values.device)
        )
        # find top k
        top_k = int(self.factor * math.log(length))
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)
        weights, delay = torch.topk(mean_value, top_k, dim=-1)
        # update corr
        tmp_corr = torch.softmax(weights, dim=-1)
        # aggregation
        tmp_values = values.repeat(1, 1, 1, 2)
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            tmp_delay = init_index + delay[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(
                1
            ).repeat(1, head, channel, length)
            pattern = torch.gather(tmp_values, dim=-1, index=tmp_delay)
            delays_agg = delays_agg + pattern * (
                tmp_corr[:, i]
                .unsqueeze(1)
                .unsqueeze(1)
                .unsqueeze(1)
                .repeat(1, head, channel, length)
            )
        return delays_agg

    def forward(self, input_tensor, attention_mask):
        mixed_query_layer = self.query_layer(input_tensor)
        mixed_key_layer = self.key_layer(input_tensor)
        mixed_value_layer = self.value_layer(input_tensor)

        queries = self.transpose_for_scores(mixed_query_layer)
        keys = self.transpose_for_scores(mixed_key_layer)
        values = self.transpose_for_scores(mixed_value_layer)

        # B, H, L, E = query_layer.shape
        # AutoFormer
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        if L > S:
            zeros = torch.zeros_like(queries[:, : (L - S), :]).float()
            values = torch.cat([values, zeros], dim=1)
            keys = torch.cat([keys, zeros], dim=1)
        else:
            values = values[:, :L, :, :]
            keys = keys[:, :L, :, :]

            # period-based dependencies
        q_fft = torch.fft.rfft(queries.permute(0, 2, 3, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(keys.permute(0, 2, 3, 1).contiguous(), dim=-1)

        # put into an empty box for sampling
        q_fft_box = torch.zeros(
            B, H, E, len(self.q_index), device=q_fft.device, dtype=torch.cfloat
        )
        q_fft_box = q_fft[:, :, :, self.q_index]

        k_fft_box = torch.zeros(
            B, H, E, len(self.k_index), device=q_fft.device, dtype=torch.cfloat
        )
        k_fft_box = k_fft[:, :, :, self.q_index]
        res = q_fft_box * torch.conj(k_fft_box)

        if self.config["use_filter"]:
            # filter
            weight = torch.view_as_complex(self.complex_weight)
            res = res * weight

        box_res = torch.zeros(
            B, H, E, L // 2 + 1, device=q_fft.device, dtype=torch.cfloat
        )
        box_res[:, :, :, self.q_index] = res

        corr = torch.fft.irfft(box_res, dim=-1)

        # time delay agg
        if self.training:
            V = self.time_delay_agg_training(
                values.permute(0, 2, 3, 1).contiguous(), corr
            ).permute(0, 3, 1, 2)
        else:
            V = self.time_delay_agg_inference(
                values.permute(0, 2, 3, 1).contiguous(), corr
            ).permute(0, 3, 1, 2)

        new_context_layer_shape = V.size()[:-2] + (self.all_head_size,)
        context_layer = V.view(*new_context_layer_shape)

        if self.dual_domain:
            # put into an empty box for sampling
            # q
            q_fft_box = torch.zeros(
                B, H, E, len(self.time_q_index), device=q_fft.device, dtype=torch.cfloat
            )
            q_fft_box = q_fft[:, :, :, self.time_q_index]
            spatial_q = torch.zeros(
                B, H, E, L // 2 + 1, device=q_fft.device, dtype=torch.cfloat
            )
            spatial_q[:, :, :, self.time_q_index] = q_fft_box

            # k
            k_fft_box = torch.zeros(
                B, H, E, len(self.time_k_index), device=q_fft.device, dtype=torch.cfloat
            )
            k_fft_box = k_fft[:, :, :, self.time_k_index]
            spatial_k = torch.zeros(
                B, H, E, L // 2 + 1, device=k_fft.device, dtype=torch.cfloat
            )
            spatial_k[:, :, :, self.time_k_index] = k_fft_box

            # v
            v_fft = torch.fft.rfft(values.permute(0, 2, 3, 1).contiguous(), dim=-1)
            # put into an empty box for sampling
            v_fft_box = torch.zeros(
                B, H, E, len(self.time_v_index), device=v_fft.device, dtype=torch.cfloat
            )
            v_fft_box = v_fft[:, :, :, self.time_v_index]
            spatial_v = torch.zeros(
                B, H, E, L // 2 + 1, device=v_fft.device, dtype=torch.cfloat
            )
            spatial_v[:, :, :, self.time_v_index] = v_fft_box

            queries = torch.fft.irfft(spatial_q, dim=-1)
            keys = torch.fft.irfft(spatial_k, dim=-1)
            values = torch.fft.irfft(spatial_v, dim=-1)

            queries = queries.permute(0, 1, 3, 2)
            keys = keys.permute(0, 1, 3, 2)
            values = values.permute(0, 1, 3, 2)

            attention_scores = torch.matmul(queries, keys.transpose(-1, -2))
            attention_scores = attention_scores / math.sqrt(self.attention_head_size)

            attention_scores = attention_scores + attention_mask
            attention_probs = nn.Softmax(dim=-1)(attention_scores)
            attention_probs = self.attn_dropout(attention_probs)
            qkv = torch.matmul(attention_probs, values)
            context_layer_spatial = qkv.permute(0, 2, 1, 3).contiguous()
            new_context_layer_shape = context_layer_spatial.size()[:-2] + (
                self.all_head_size,
            )
            context_layer_spatial = context_layer_spatial.view(*new_context_layer_shape)
            context_layer = (
                1 - self.spatial_ratio
            ) * context_layer + self.spatial_ratio * context_layer_spatial

        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class FeedForward(nn.Module):
    """
    Point-wise feed-forward layer is implemented by two dense layers.

    Args:
        input_tensor (torch.Tensor): the input of the point-wise feed-forward layer

    Returns:
        hidden_states (torch.Tensor): the output of the point-wise feed-forward layer

    """

    def __init__(
        self, hidden_size, inner_size, hidden_dropout_prob, hidden_act, layer_norm_eps
    ):
        super(FeedForward, self).__init__()
        self.dense_1 = nn.Linear(hidden_size, inner_size)
        self.intermediate_act_fn = self.get_hidden_act(hidden_act)

        self.dense_2 = nn.Linear(inner_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def get_hidden_act(self, act):
        ACT2FN = {
            "gelu": self.gelu,
            "relu": fn.relu,
            "swish": self.swish,
            "tanh": torch.tanh,
            "sigmoid": torch.sigmoid,
        }
        return ACT2FN[act]

    def gelu(self, x):
        """Implementation of the gelu activation function.

        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results)::

            0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

        Also see https://arxiv.org/abs/1606.08415
        """
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    def swish(self, x):
        return x * torch.sigmoid(x)

    def forward(self, input_tensor):
        hidden_states = self.dense_1(input_tensor)
        hidden_states = self.intermediate_act_fn(hidden_states)

        hidden_states = self.dense_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class FEABlock(nn.Module):
    """
    One transformer layer consists of a multi-head self-attention layer and a point-wise feed-forward layer.

    Args:
        hidden_states (torch.Tensor): the input of the multi-head self-attention sublayer
        attention_mask (torch.Tensor): the attention mask for the multi-head self-attention sublayer

    Returns:
        feedforward_output (torch.Tensor): The output of the point-wise feed-forward sublayer,
                                           is the output of the transformer layer.

    """

    def __init__(
        self,
        n_heads,
        hidden_size,
        intermediate_size,
        hidden_dropout_prob,
        attn_dropout_prob,
        hidden_act,
        layer_norm_eps,
        n,
        config,
    ):
        super(FEABlock, self).__init__()
        self.hybrid_attention = HybridAttention(
            n_heads,
            hidden_size,
            hidden_dropout_prob,
            attn_dropout_prob,
            layer_norm_eps,
            n,
            config,
        )
        self.feed_forward = FeedForward(
            hidden_size,
            intermediate_size,
            hidden_dropout_prob,
            hidden_act,
            layer_norm_eps,
        )

    def forward(self, hidden_states, attention_mask):
        attention_output = self.hybrid_attention(hidden_states, attention_mask)
        feedforward_output = self.feed_forward(attention_output)

        return feedforward_output


class FEAEncoder(nn.Module):
    r"""One TransformerEncoder consists of several TransformerLayers.

    - n_layers(num): num of transformer layers in transformer encoder. Default: 2
    - n_heads(num): num of attention heads for multi-head attention layer. Default: 2
    - hidden_size(num): the input and output hidden size. Default: 64
    - inner_size(num): the dimensionality in feed-forward layer. Default: 256
    - hidden_dropout_prob(float): probability of an element to be zeroed. Default: 0.5
    - attn_dropout_prob(float): probability of an attention score to be zeroed. Default: 0.5
    - hidden_act(str): activation function in feed-forward layer. Default: 'gelu'
                  candidates: 'gelu', 'relu', 'swish', 'tanh', 'sigmoid'
    - layer_norm_eps(float): a value added to the denominator for numerical stability. Default: 1e-12

    """

    def __init__(
        self,
        n_layers=2,
        n_heads=2,
        hidden_size=64,
        inner_size=256,
        hidden_dropout_prob=0.5,
        attn_dropout_prob=0.5,
        hidden_act="gelu",
        layer_norm_eps=1e-12,
        config=None,
    ):
        super(FEAEncoder, self).__init__()
        self.n_layers = n_layers
        self.layer = nn.ModuleList()
        for n in range(self.n_layers):
            self.layer_ramp = FEABlock(
                n_heads,
                hidden_size,
                inner_size,
                hidden_dropout_prob,
                attn_dropout_prob,
                hidden_act,
                layer_norm_eps,
                n,
                config,
            )
            self.layer.append(self.layer_ramp)

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        """
        Args:
            hidden_states (torch.Tensor): the input of the TransformerEncoder
            attention_mask (torch.Tensor): the attention mask for the input hidden_states
            output_all_encoded_layers (Bool): whether output all transformer layers' output

        Returns:
            all_encoder_layers (list): if output_all_encoded_layers is True, return a list consists of all transformer
            layers' output, otherwise return a list only consists of the output of last transformer layer.

        """
        all_encoder_layers = []

        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers
