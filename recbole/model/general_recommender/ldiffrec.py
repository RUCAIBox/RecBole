# -*- coding: utf-8 -*-
# @Time   : 2023/10/6
# @Author : Enze Liu
# @Email  : enzeeliu@foxmail.com

r"""
DiffRec
################################################
Reference:
    Wenjie Wang et al. "Diffusion Recommender Model." in SIGIR 2023.

Reference code:
    https://github.com/YiyanXu/DiffRec
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from recbole.model.init import xavier_normal_initialization
from recbole.model.layers import MLPLayers
from recbole.model.general_recommender.diffrec import (
    DiffRec,
    DNN,
    ModelMeanType,
    mean_flat,
)


class AutoEncoder(nn.Module):
    r"""
    Guassian Diffusion for large-scale recommendation.
    """

    def __init__(
        self,
        item_emb,
        n_cate,
        in_dims,
        out_dims,
        device,
        act_func,
        reparam=True,
        dropout=0.1,
    ):
        super(AutoEncoder, self).__init__()

        self.item_emb = item_emb
        self.n_cate = n_cate
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.act_func = act_func
        self.n_item = len(item_emb)
        self.reparam = reparam
        self.dropout = nn.Dropout(dropout)

        if n_cate == 1:  # no clustering
            in_dims_temp = (
                [self.n_item + 1] + self.in_dims[:-1] + [self.in_dims[-1] * 2]
            )
            out_dims_temp = [self.in_dims[-1]] + self.out_dims + [self.n_item + 1]

            self.encoder = MLPLayers(in_dims_temp, activation=self.act_func)
            self.decoder = MLPLayers(
                out_dims_temp, activation=self.act_func, last_activation=False
            )

        else:
            from kmeans_pytorch import kmeans

            self.cluster_ids, _ = kmeans(
                X=item_emb, num_clusters=n_cate, distance="euclidean", device=device
            )
            # cluster_ids(labels): [0, 1, 2, 2, 1, 0, 0, ...]
            category_idx = []
            for i in range(n_cate):
                idx = np.argwhere(self.cluster_ids.numpy() == i).flatten().tolist()
                category_idx.append(torch.tensor(idx, dtype=int) + 1)
            self.category_idx = category_idx  # [cate1: [iid1, iid2, ...], cate2: [iid3, iid4, ...], cate3: [iid5, iid6, ...]]
            self.category_map = torch.cat(tuple(category_idx), dim=-1)  # map
            self.category_len = [
                len(self.category_idx[i]) for i in range(n_cate)
            ]  # item num in each category
            print("category length: ", self.category_len)
            assert sum(self.category_len) == self.n_item

            ##### Build the Encoder and Decoder #####
            encoders = []
            decode_dim = []
            for i in range(n_cate):
                if i == n_cate - 1:
                    latent_dims = list(self.in_dims - np.array(decode_dim).sum(axis=0))
                else:
                    latent_dims = [
                        int(self.category_len[i] / self.n_item * self.in_dims[j])
                        for j in range(len(self.in_dims))
                    ]
                    latent_dims = [
                        latent_dims[j] if latent_dims[j] != 0 else 1
                        for j in range(len(self.in_dims))
                    ]
                in_dims_temp = (
                    [self.category_len[i]] + latent_dims[:-1] + [latent_dims[-1] * 2]
                )
                encoders.append(MLPLayers(in_dims_temp, activation=self.act_func))
                decode_dim.append(latent_dims)

            self.encoder = nn.ModuleList(encoders)
            print("Latent dims of each category: ", decode_dim)

            self.decode_dim = [decode_dim[i][::-1] for i in range(len(decode_dim))]

            if len(out_dims) == 0:  # one-layer decoder: [encoder_dim_sum, n_item]
                out_dim = self.in_dims[-1]
                self.decoder = MLPLayers([out_dim, self.n_item], activation=None)
            else:  # multi-layer decoder: [encoder_dim, hidden_size, cate_num]
                # decoder_modules = [[] for _ in range(n_cate)]
                decoders = []
                for i in range(n_cate):
                    out_dims_temp = self.decode_dim[i] + [self.category_len[i]]
                    decoders.append(
                        MLPLayers(
                            out_dims_temp,
                            activation=self.act_func,
                            last_activation=False,
                        )
                    )
                self.decoder = nn.ModuleList(decoders)

        self.apply(xavier_normal_initialization)

    def Encode(self, batch):
        batch = self.dropout(batch)
        if self.n_cate == 1:
            hidden = self.encoder(batch)
            mu = hidden[:, : self.in_dims[-1]]
            logvar = hidden[:, self.in_dims[-1] :]

            if self.training and self.reparam:
                latent = self.reparamterization(mu, logvar)
            else:
                latent = mu

            kl_divergence = -0.5 * torch.mean(
                torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
            )

            return batch, latent, kl_divergence

        else:
            batch_cate = []
            for i in range(self.n_cate):
                batch_cate.append(batch[:, self.category_idx[i]])
            # [batch_size, n_items] -> [[batch_size, n1_items], [batch_size, n2_items], [batch_size, n3_items]]
            latent_mu = []
            latent_logvar = []
            for i in range(self.n_cate):
                hidden = self.encoder[i](batch_cate[i])
                latent_mu.append(hidden[:, : self.decode_dim[i][0]])
                latent_logvar.append(hidden[:, self.decode_dim[i][0] :])
            # latent: [[batch_size, latent_size1], [batch_size, latent_size2], [batch_size, latent_size3]]

            mu = torch.cat(tuple(latent_mu), dim=-1)
            logvar = torch.cat(tuple(latent_logvar), dim=-1)
            if self.training and self.reparam:
                latent = self.reparamterization(mu, logvar)
            else:
                latent = mu

            kl_divergence = -0.5 * torch.mean(
                torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
            )

            return torch.cat(tuple(batch_cate), dim=-1), latent, kl_divergence

    def reparamterization(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def Decode(self, batch):
        if len(self.out_dims) == 0 or self.n_cate == 1:  # one-layer decoder
            return self.decoder(batch)
        else:
            batch_cate = []
            start = 0
            for i in range(self.n_cate):
                end = start + self.decode_dim[i][0]
                batch_cate.append(batch[:, start:end])
                start = end
            pred_cate = []
            for i in range(self.n_cate):
                pred_cate.append(self.decoder[i](batch_cate[i]))
            pred = torch.cat(tuple(pred_cate), dim=-1)

            return pred


class LDiffRec(DiffRec):
    r"""
    L-DiffRec clusters items into groups, compresses the interaction vector over each group into a
    low-dimensional latent vector via a group-specific VAE, and conducts the forward and reverse
    diffusion processes in the latent space.
    """

    def __init__(self, config, dataset):
        super(LDiffRec, self).__init__(config, dataset)
        self.n_cate = config["n_cate"]
        self.reparam = config["reparam"]
        self.ae_act_func = config["ae_act_func"]
        self.in_dims = config["in_dims"]
        self.out_dims = config["out_dims"]

        # control loss in training
        self.update_count = 0
        self.update_count_vae = 0
        self.lamda = config["lamda"]
        self.anneal_cap = config["anneal_cap"]
        self.anneal_steps = config["anneal_steps"]
        self.vae_anneal_cap = config["vae_anneal_cap"]
        self.vae_anneal_steps = config["vae_anneal_steps"]

        out_dims = self.out_dims
        in_dims = self.in_dims[::-1]
        emb_path = os.path.join(dataset.dataset_path, f"item_emb.npy")
        if self.n_cate > 1:
            if not os.path.exists(emb_path):
                self.logger.exception(
                    "The item embedding file must be given when n_cate>1."
                )
            item_emb = torch.from_numpy(np.load(emb_path, allow_pickle=True))
        else:
            item_emb = torch.zeros((self.n_items - 1, 64))
        self.autoencoder = AutoEncoder(
            item_emb,
            self.n_cate,
            in_dims,
            out_dims,
            self.device,
            self.ae_act_func,
            self.reparam,
        ).to(self.device)

        self.latent_size = in_dims[-1]
        dims = [self.latent_size] + config["dims_dnn"] + [self.latent_size]
        self.mlp = DNN(
            dims=dims,
            emb_size=self.emb_size,
            time_type="cat",
            norm=self.norm,
            act_func=self.mlp_act_func,
        ).to(self.device)

    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        batch = self.get_rating_matrix(user)

        batch_cate, batch_latent, vae_kl = self.autoencoder.Encode(batch)

        # calculate loss in diffusion
        batch_size, device = batch_latent.size(0), batch_latent.device
        ts, pt = self.sample_timesteps(batch_size, device, "importance")
        noise = torch.randn_like(batch_latent)
        if self.noise_scale != 0.0:
            x_t = self.q_sample(batch_latent, ts, noise)
        else:
            x_t = batch_latent

        model_output = self.mlp(x_t, ts)
        target = {
            ModelMeanType.START_X: batch_latent,
            ModelMeanType.EPSILON: noise,
        }[self.mean_type]

        assert model_output.shape == target.shape == batch_latent.shape

        mse = mean_flat((target - model_output) ** 2)

        reloss = self.reweight_loss(
            batch_latent, x_t, mse, ts, target, model_output, device
        )

        if self.mean_type == ModelMeanType.START_X:
            batch_latent_recon = model_output
        else:
            batch_latent_recon = self._predict_xstart_from_eps(x_t, ts, model_output)

        self.update_Lt_history(ts, reloss)

        diff_loss = (reloss / pt).mean()

        batch_recon = self.autoencoder.Decode(batch_latent_recon)

        if self.anneal_steps > 0:
            lamda = max(
                (1.0 - self.update_count / self.anneal_steps) * self.lamda,
                self.anneal_cap,
            )
        else:
            lamda = max(self.lamda, self.anneal_cap)

        if self.vae_anneal_steps > 0:
            anneal = min(
                self.vae_anneal_cap, 1.0 * self.update_count_vae / self.vae_anneal_steps
            )
        else:
            anneal = self.vae_anneal_cap

        self.update_count_vae += 1
        self.update_count += 1
        vae_loss = compute_loss(batch_recon, batch_cate) + anneal * vae_kl

        loss = lamda * diff_loss + vae_loss

        return loss

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        batch = self.get_rating_matrix(user)
        _, batch_latent, _ = self.autoencoder.Encode(batch)
        batch_latent_recon = super(LDiffRec, self).p_sample(batch_latent)
        prediction = self.autoencoder.Decode(
            batch_latent_recon
        )  # [batch_size, n1_items + n2_items + n3_items]
        if self.n_cate > 1:
            transform = torch.zeros((prediction.shape[0], prediction.shape[1] + 1)).to(
                prediction.device
            )
            transform[:, self.autoencoder.category_map] = prediction
        else:
            transform = prediction
        return transform

    def predict(self, interaction):
        item = interaction[self.ITEM_ID]
        x_t = self.full_sort_predict(interaction)
        scores = x_t[torch.arange(len(item)).to(self.device), item]
        return scores


def compute_loss(recon_x, x):
    return -torch.mean(
        torch.sum(F.log_softmax(recon_x, 1) * x, -1)
    )  # multinomial log likelihood in MultVAE
