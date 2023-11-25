# -*- coding: utf-8 -*-
# @Time   : 2020/8/17 19:38
# @Author : Yujie Lu
# @Email  : yujielu1998@gmail.com

# UPDATE:
# @Time   : 2020/8/19, 2020/10/2
# @Author : Yupeng Hou, Yujie Lu
# @Email  : houyupeng@ruc.edu.cn, yujielu1998@gmail.com

# UPDATE:
# @Time   : 2023/11/24
# @Author : Haw-Shiuan Chang
# @Email  : ken77921@gmail.com

r"""
GRU4Rec + Softmax-CPR
################################################

Reference:
    Yong Kiam Tan et al. "Improved Recurrent Neural Networks for Session-based Recommendations." in DLRS 2016.
    Haw-Shiuan Chang, Nikhil Agarwal, and Andrew McCallum "To Copy, or not to Copy; That is a Critical Issue of the Output Softmax Layer in Neural Sequential Recommenders" in WSDM 2024


"""

import torch
from torch import nn
from torch.nn.init import xavier_uniform_, xavier_normal_

import torch.nn.functional as F
import math
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.loss import BPRLoss
import sys


def gelu(x):
    return (
        0.5
        * x
        * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    )


class GRU4RecCPR(SequentialRecommender):
    r"""GRU4Rec is a model that incorporate RNN for recommendation.

    Note:

        Regarding the innovation of this article,we can only achieve the data augmentation mentioned
        in the paper and directly output the embedding of the item,
        in order that the generation method we used is common to other sequential models.
    """

    def __init__(self, config, dataset):
        super(GRU4RecCPR, self).__init__(config, dataset)

        # load parameters info
        self.hidden_size = config["hidden_size"]
        self.embedding_size = config["embedding_size"]
        self.loss_type = config["loss_type"]
        self.num_layers = config["num_layers"]
        self.dropout_prob = config["dropout_prob"]

        self.n_facet_all = config["n_facet_all"]  # added for mfs
        self.n_facet = config["n_facet"]  # added for mfs
        self.n_facet_window = config["n_facet_window"]  # added for mfs
        self.n_facet_hidden = min(
            config["n_facet_hidden"], config["num_layers"]
        )  # config['n_facet_hidden'] #added for mfs
        self.n_facet_MLP = config["n_facet_MLP"]  # added for mfs
        self.n_facet_context = config["n_facet_context"]  # added for dynamic partioning
        self.n_facet_reranker = config[
            "n_facet_reranker"
        ]  # added for dynamic partioning
        self.n_facet_emb = config["n_facet_emb"]  # added for dynamic partioning
        assert self.n_facet_MLP <= 0  # -1 or 0
        assert self.n_facet_window <= 0
        self.n_facet_window = -self.n_facet_window
        self.n_facet_MLP = -self.n_facet_MLP
        self.softmax_nonlinear = "None"  # added for mfs
        self.use_out_emb = config["use_out_emb"]  # added for mfs
        self.only_compute_loss = True  # added for mfs

        self.dense = nn.Linear(self.hidden_size, self.embedding_size)
        out_size = self.embedding_size

        self.n_embd = out_size

        self.use_proj_bias = config["use_proj_bias"]  # added for mfs
        self.weight_mode = config["weight_mode"]  # added for mfs
        self.context_norm = config["context_norm"]  # added for mfs
        self.post_remove_context = config["post_remove_context"]  # added for mfs
        self.reranker_merging_mode = config["reranker_merging_mode"]  # added for mfs
        self.partition_merging_mode = config["partition_merging_mode"]  # added for mfs
        self.reranker_CAN_NUM = [
            int(x) for x in str(config["reranker_CAN_NUM"]).split(",")
        ]
        assert self.use_proj_bias is not None
        self.candidates_from_previous_reranker = True
        if self.weight_mode == "max_logits":
            self.n_facet_effective = 1
        else:
            self.n_facet_effective = self.n_facet

        assert (
            self.n_facet
            + self.n_facet_context
            + self.n_facet_reranker * len(self.reranker_CAN_NUM)
            + self.n_facet_emb
            == self.n_facet_all
        )
        assert self.n_facet_emb == 0 or self.n_facet_emb == 2

        hidden_state_input_ratio = 1 + self.n_facet_MLP  # 1 + 1
        self.MLP_linear = nn.Linear(
            self.n_embd * (self.n_facet_hidden * (self.n_facet_window + 1)),
            self.n_embd * self.n_facet_MLP,
        )  # (hid_dim*2) -> (hid_dim)
        total_lin_dim = self.n_embd * hidden_state_input_ratio
        self.project_arr = nn.ModuleList(
            [
                nn.Linear(total_lin_dim, self.n_embd, bias=self.use_proj_bias)
                for i in range(self.n_facet_all)
            ]
        )

        self.project_emb = nn.Linear(self.n_embd, self.n_embd, bias=self.use_proj_bias)
        if len(self.weight_mode) > 0:
            self.weight_facet_decoder = nn.Linear(
                self.n_embd * hidden_state_input_ratio, self.n_facet_effective
            )
            self.weight_global = nn.Parameter(torch.ones(self.n_facet_effective))

        self.c = 123

        # define layers and loss
        self.emb_dropout = nn.Dropout(self.dropout_prob)
        self.gru_layers = nn.GRU(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bias=False,
            batch_first=True,
        )
        self.item_embedding = nn.Embedding(
            self.n_items, self.embedding_size, padding_idx=0
        )

        if self.use_out_emb:
            self.out_item_embedding = nn.Linear(out_size, self.n_items, bias=False)
        else:
            self.out_item_embedding = self.item_embedding
            self.out_item_embedding.bias = None

        if self.loss_type == "BPR":
            print("current softmax-cpr code does not support BPR loss")
            sys.exit(0)
        elif self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight)
        elif isinstance(module, nn.GRU):
            xavier_uniform_(module.weight_hh_l0)
            xavier_uniform_(module.weight_ih_l0)

    def forward(self, item_seq, item_seq_len):
        item_seq_emb = self.item_embedding(item_seq)
        item_seq_emb_dropout = self.emb_dropout(item_seq_emb)
        gru_output, _ = self.gru_layers(item_seq_emb_dropout)
        gru_output = self.dense(gru_output)
        return gru_output

    def get_facet_emb(self, input_emb, i):
        return self.project_arr[i](input_emb)

    def calculate_loss_prob(self, interaction, only_compute_prob=False):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        last_layer_hs = self.forward(item_seq, item_seq_len)
        all_hidden_states = [last_layer_hs]
        if self.loss_type != "CE":
            print(
                "current softmax-cpr code does not support BPR or the losses other than cross entropy"
            )
            sys.exit(0)
        else:  # self.loss_type = 'CE'
            test_item_emb = self.out_item_embedding.weight
            test_item_bias = self.out_item_embedding.bias

            """mfs code starts"""
            device = all_hidden_states[0].device

            hidden_emb_arr = []
            # h_facet_hidden -> H, n_face_window -> W, here 1 and 0
            for i in range(self.n_facet_hidden):
                # print('all_hidden_states length is {}. i is {}'.format(len(all_hidden_states), i))
                hidden_states = all_hidden_states[
                    -(i + 1)
                ]  # i-th hidden-state embedding from the top
                device = hidden_states.device
                hidden_emb_arr.append(hidden_states)
                for j in range(self.n_facet_window):
                    (
                        bsz,
                        seq_len,
                        hidden_size,
                    ) = (
                        hidden_states.size()
                    )  # bsz -> , seq_len -> , hidden_size -> 768 in GPT-small?
                    if j + 1 < hidden_states.size(1):
                        shifted_hidden = torch.cat(
                            (
                                torch.zeros((bsz, (j + 1), hidden_size), device=device),
                                hidden_states[:, : -(j + 1), :],
                            ),
                            dim=1,
                        )
                    else:
                        shifted_hidden = torch.zeros(
                            (bsz, hidden_states.size(1), hidden_size), device=device
                        )
                    hidden_emb_arr.append(shifted_hidden)
            # hidden_emb_arr -> (W*H, bsz, seq_len, hidden_size)

            # n_facet_MLP -> 1
            if self.n_facet_MLP > 0:
                stacked_hidden_emb_raw_arr = torch.cat(
                    hidden_emb_arr, dim=-1
                )  # (bsz, seq_len, W*H*hidden_size)
                hidden_emb_MLP = self.MLP_linear(
                    stacked_hidden_emb_raw_arr
                )  # bsz, seq_len, hidden_size
                stacked_hidden_emb_arr_raw = torch.cat(
                    [hidden_emb_arr[0], gelu(hidden_emb_MLP)], dim=-1
                )  # bsz, seq_len, 2*hidden_size
            else:
                stacked_hidden_emb_arr_raw = hidden_emb_arr[0]

            # Only use the hidden state corresponding to the last word
            stacked_hidden_emb_arr = stacked_hidden_emb_arr_raw[:, -1, :].unsqueeze(
                dim=1
            )

            # list of linear projects per facet
            projected_emb_arr = []
            # list of final logits per facet
            facet_lm_logits_arr = []

            rereanker_candidate_token_ids_arr = []
            for i in range(self.n_facet):
                projected_emb = self.get_facet_emb(
                    stacked_hidden_emb_arr, i
                )  # (bsz, seq_len, hidden_dim)
                projected_emb_arr.append(projected_emb)
                lm_logits = F.linear(projected_emb, test_item_emb, test_item_bias)
                facet_lm_logits_arr.append(lm_logits)
                if (
                    i < self.n_facet_reranker
                    and not self.candidates_from_previous_reranker
                ):
                    candidate_token_ids = []
                    for j in range(len(self.reranker_CAN_NUM)):
                        _, candidate_token_ids_ = torch.topk(
                            lm_logits, self.reranker_CAN_NUM[j]
                        )
                        candidate_token_ids.append(candidate_token_ids_)
                    rereanker_candidate_token_ids_arr.append(candidate_token_ids)

            for i in range(self.n_facet_reranker):
                for j in range(len(self.reranker_CAN_NUM)):
                    projected_emb = self.get_facet_emb(
                        stacked_hidden_emb_arr,
                        self.n_facet + i * len(self.reranker_CAN_NUM) + j,
                    )  # (bsz, seq_len, hidden_dim)
                    projected_emb_arr.append(projected_emb)

            for i in range(self.n_facet_context):
                projected_emb = self.get_facet_emb(
                    stacked_hidden_emb_arr,
                    self.n_facet
                    + self.n_facet_reranker * len(self.reranker_CAN_NUM)
                    + i,
                )  # (bsz, seq_len, hidden_dim)
                projected_emb_arr.append(projected_emb)

            # to generate context-based embeddings for words in input
            for i in range(self.n_facet_emb):
                projected_emb = self.get_facet_emb(
                    stacked_hidden_emb_arr_raw,
                    self.n_facet
                    + self.n_facet_context
                    + self.n_facet_reranker * len(self.reranker_CAN_NUM)
                    + i,
                )  # (bsz, seq_len, hidden_dim)
                projected_emb_arr.append(projected_emb)

            for i in range(self.n_facet_reranker):
                bsz, seq_len, hidden_size = projected_emb_arr[i].size()
                for j in range(len(self.reranker_CAN_NUM)):
                    if self.candidates_from_previous_reranker:
                        _, candidate_token_ids = torch.topk(
                            facet_lm_logits_arr[i], self.reranker_CAN_NUM[j]
                        )  # (bsz, seq_len, topk)
                    else:
                        candidate_token_ids = rereanker_candidate_token_ids_arr[i][j]
                    logit_hidden_reranker_topn = (
                        projected_emb_arr[
                            self.n_facet + i * len(self.reranker_CAN_NUM) + j
                        ]
                        .unsqueeze(dim=2)
                        .expand(bsz, seq_len, self.reranker_CAN_NUM[j], hidden_size)
                        * test_item_emb[candidate_token_ids, :]
                    ).sum(
                        dim=-1
                    )  # (bsz, seq_len, emb_size) x (bsz, seq_len, topk, emb_size) -> (bsz, seq_len, topk)
                    if test_item_bias is not None:
                        logit_hidden_reranker_topn += test_item_bias[
                            candidate_token_ids
                        ]
                    if self.reranker_merging_mode == "add":
                        # print("inside reranker")
                        facet_lm_logits_arr[i].scatter_add_(
                            2, candidate_token_ids, logit_hidden_reranker_topn
                        )  # (bsz, seq_len, vocab_size) <- (bsz, seq_len, topk) x (bsz, seq_len, topk)
                    else:
                        facet_lm_logits_arr[i].scatter_(
                            2, candidate_token_ids, logit_hidden_reranker_topn
                        )  # (bsz, seq_len, vocab_size) <- (bsz, seq_len, topk) x (bsz, seq_len, topk)

            for i in range(self.n_facet_context):
                bsz, seq_len_1, hidden_size = projected_emb_arr[i].size()
                bsz, seq_len_2 = item_seq.size()
                logit_hidden_context = (
                    projected_emb_arr[
                        self.n_facet
                        + self.n_facet_reranker * len(self.reranker_CAN_NUM)
                        + i
                    ]
                    .unsqueeze(dim=2)
                    .expand(-1, -1, seq_len_2, -1)
                    * test_item_emb[item_seq, :]
                    .unsqueeze(dim=1)
                    .expand(-1, seq_len_1, -1, -1)
                ).sum(dim=-1)
                if test_item_bias is not None:
                    logit_hidden_context += (
                        test_item_bias[item_seq]
                        .unsqueeze(dim=1)
                        .expand(-1, seq_len_1, -1)
                    )
                logit_hidden_pointer = 0
                if self.n_facet_emb == 2:
                    logit_hidden_pointer = (
                        projected_emb_arr[-2][:, -1, :]
                        .unsqueeze(dim=1)
                        .unsqueeze(dim=1)
                        .expand(-1, seq_len_1, seq_len_2, -1)
                        * projected_emb_arr[-1]
                        .unsqueeze(dim=1)
                        .expand(-1, seq_len_1, -1, -1)
                    ).sum(dim=-1)

                item_seq_expand = item_seq.unsqueeze(dim=1).expand(-1, seq_len_1, -1)
                only_new_logits = torch.zeros_like(facet_lm_logits_arr[i])
                if self.context_norm:
                    only_new_logits.scatter_add_(
                        dim=2,
                        index=item_seq_expand,
                        src=logit_hidden_context + logit_hidden_pointer,
                    )
                    item_count = torch.zeros_like(only_new_logits) + 1e-15
                    item_count.scatter_add_(
                        dim=2,
                        index=item_seq_expand,
                        src=torch.ones_like(item_seq_expand).to(dtype=item_count.dtype),
                    )
                    only_new_logits = only_new_logits / item_count
                else:
                    only_new_logits.scatter_add_(
                        dim=2, index=item_seq_expand, src=logit_hidden_context
                    )
                    item_count = torch.zeros_like(only_new_logits) + 1e-15
                    item_count.scatter_add_(
                        dim=2,
                        index=item_seq_expand,
                        src=torch.ones_like(item_seq_expand).to(dtype=item_count.dtype),
                    )
                    only_new_logits = only_new_logits / item_count
                    only_new_logits.scatter_add_(
                        dim=2, index=item_seq_expand, src=logit_hidden_pointer
                    )

                if self.partition_merging_mode == "replace":
                    facet_lm_logits_arr[i].scatter_(
                        dim=2,
                        index=item_seq_expand,
                        src=torch.zeros_like(item_seq_expand).to(
                            dtype=facet_lm_logits_arr[i].dtype
                        ),
                    )
                    facet_lm_logits_arr[i] = facet_lm_logits_arr[i] + only_new_logits
                elif self.partition_merging_mode == "add":
                    facet_lm_logits_arr[i] = facet_lm_logits_arr[i] + only_new_logits
                elif self.partition_merging_mode == "half":
                    item_in_context = torch.ones_like(only_new_logits)
                    item_in_context.scatter_(
                        dim=2,
                        index=item_seq_expand,
                        src=2
                        * torch.ones_like(item_seq_expand).to(dtype=item_count.dtype),
                    )
                    facet_lm_logits_arr[i] = (
                        facet_lm_logits_arr[i] / item_in_context + only_new_logits
                    )

            weight = None
            if self.weight_mode == "dynamic":
                weight = self.weight_facet_decoder(stacked_hidden_emb_arr).softmax(
                    dim=-1
                )  # hidden_dim*hidden_input_state_ration -> n_facet_effective
            elif self.weight_mode == "static":
                weight = self.weight_global.softmax(
                    dim=-1
                )  # torch.ones(n_facet_effective)
            elif self.weight_mode == "max_logits":
                stacked_facet_lm_logits = torch.stack(facet_lm_logits_arr, dim=0)
                facet_lm_logits_arr = [stacked_facet_lm_logits.amax(dim=0)]

            prediction_prob = 0

            for i in range(self.n_facet_effective):
                facet_lm_logits = facet_lm_logits_arr[i]
                if self.softmax_nonlinear == "sigsoftmax":  #'None' here
                    facet_lm_logits_sig = torch.exp(
                        facet_lm_logits - facet_lm_logits.max(dim=-1, keepdim=True)[0]
                    ) * (1e-20 + torch.sigmoid(facet_lm_logits))
                    facet_lm_logits_softmax = (
                        facet_lm_logits_sig
                        / facet_lm_logits_sig.sum(dim=-1, keepdim=True)
                    )
                elif self.softmax_nonlinear == "None":
                    facet_lm_logits_softmax = facet_lm_logits.softmax(
                        dim=-1
                    )  # softmax over final logits
                if self.weight_mode == "dynamic":
                    prediction_prob += facet_lm_logits_softmax * weight[
                        :, :, i
                    ].unsqueeze(-1)
                elif self.weight_mode == "static":
                    prediction_prob += facet_lm_logits_softmax * weight[i]
                else:
                    prediction_prob += (
                        facet_lm_logits_softmax / self.n_facet_effective
                    )  # softmax over final logits/1
            if not only_compute_prob:
                inp = torch.log(prediction_prob.view(-1, self.n_items) + 1e-8)
                pos_items = interaction[self.POS_ITEM_ID]
                loss_raw = self.loss_fct(inp, pos_items.view(-1))
                loss = loss_raw.mean()
            else:
                loss = None
            return loss, prediction_prob.squeeze(dim=1)

    def calculate_loss(self, interaction):
        loss, prediction_prob = self.calculate_loss_prob(interaction)
        return loss

    def predict(self, interaction):
        print(
            "Current softmax cpr code does not support negative sampling in an efficient way just like RepeatNet.",
            file=sys.stderr,
        )
        assert False  # If you can accept slow running time, uncomment this line.
        loss, prediction_prob = self.calculate_loss_prob(
            interaction, only_compute_prob=True
        )
        if self.post_remove_context:
            item_seq = interaction[self.ITEM_SEQ]
            prediction_prob.scatter_(1, item_seq, 0)
        test_item = interaction[self.ITEM_ID]
        prediction_prob = prediction_prob.unsqueeze(-1)
        # batch_size * num_items * 1
        scores = self.gather_indexes(prediction_prob, test_item).squeeze(-1)
        return scores

    def full_sort_predict(self, interaction):
        loss, prediction_prob = self.calculate_loss_prob(interaction)
        item_seq = interaction[self.ITEM_SEQ]
        if self.post_remove_context:
            prediction_prob.scatter_(1, item_seq, 0)
        return prediction_prob
