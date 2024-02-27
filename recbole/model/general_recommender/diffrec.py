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

import enum
import math
import copy
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from recbole.model.init import xavier_normal_initialization
from recbole.utils.enum_type import InputType
from recbole.model.abstract_recommender import AutoEncoderMixin, GeneralRecommender
from recbole.model.layers import MLPLayers
import typing


class ModelMeanType(enum.Enum):
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon


class DNN(nn.Module):
    r"""
    A deep neural network for the reverse diffusion preocess.
    """

    def __init__(
        self,
        dims: typing.List,
        emb_size: int,
        time_type="cat",
        act_func="tanh",
        norm=False,
        dropout=0.5,
    ):
        super(DNN, self).__init__()
        self.dims = dims
        self.time_type = time_type
        self.time_emb_dim = emb_size
        self.norm = norm

        self.emb_layer = nn.Linear(self.time_emb_dim, self.time_emb_dim)

        if self.time_type == "cat":
            # Concatenate timestep embedding with input
            self.dims[0] += self.time_emb_dim
        else:
            raise ValueError(
                "Unimplemented timestep embedding type %s" % self.time_type
            )

        self.mlp_layers = MLPLayers(
            layers=self.dims, dropout=0, activation=act_func, last_activation=False
        )
        self.drop = nn.Dropout(dropout)

        self.apply(xavier_normal_initialization)

    def forward(self, x, timesteps):
        time_emb = timestep_embedding(timesteps, self.time_emb_dim).to(x.device)
        emb = self.emb_layer(time_emb)
        if self.norm:
            x = F.normalize(x)
        x = self.drop(x)
        h = torch.cat([x, emb], dim=-1)
        h = self.mlp_layers(h)
        return h


class DiffRec(GeneralRecommender, AutoEncoderMixin):
    r"""
    DiffRec is a generative recommender model which infers users' interaction probabilities in a denoising manner.
    Note that DiffRec simultaneously ranks all items for each user.
    We implement the the DiffRec model with only user dataloader.
    """

    input_type = InputType.LISTWISE

    def __init__(self, config, dataset):
        super(DiffRec, self).__init__(config, dataset)

        if config["mean_type"] == "x0":
            self.mean_type = ModelMeanType.START_X
        elif config["mean_type"] == "eps":
            self.mean_type = ModelMeanType.EPSILON
        else:
            raise ValueError("Unimplemented mean type %s" % config["mean_type"])
        self.time_aware = config["time-aware"]
        self.w_max = config["w_max"]
        self.w_min = config["w_min"]
        self.build_histroy_items(dataset)

        self.noise_schedule = config["noise_schedule"]
        self.noise_scale = config["noise_scale"]
        self.noise_min = config["noise_min"]
        self.noise_max = config["noise_max"]
        self.steps = config["steps"]
        self.beta_fixed = config["beta_fixed"]
        self.emb_size = config["embedding_size"]
        self.norm = config["norm"]  # True or False
        self.reweight = config["reweight"]  # reweight the loss for different timesteps
        if self.noise_scale == 0.0:
            self.reweight = False
        self.sampling_noise = config[
            "sampling_noise"
        ]  # whether sample noise during predict
        self.sampling_steps = config["sampling_steps"]
        self.mlp_act_func = config["mlp_act_func"]
        assert self.sampling_steps <= self.steps, "Too much steps in inference."

        self.history_num_per_term = config["history_num_per_term"]
        self.Lt_history = torch.zeros(
            self.steps, self.history_num_per_term, dtype=torch.float64
        ).to(self.device)
        self.Lt_count = torch.zeros(self.steps, dtype=int).to(self.device)

        dims = [self.n_items] + config["dims_dnn"] + [self.n_items]

        self.mlp = DNN(
            dims=dims,
            emb_size=self.emb_size,
            time_type="cat",
            norm=self.norm,
            act_func=self.mlp_act_func,
        ).to(self.device)

        if self.noise_scale != 0.0:
            self.betas = torch.tensor(self.get_betas(), dtype=torch.float64).to(
                self.device
            )
            if self.beta_fixed:
                self.betas[0] = (
                    0.00001  # Deep Unsupervised Learning using Noneequilibrium Thermodynamics 2.4.1
                )
                # The variance \beta_1 of the first step is fixed to a small constant to prevent overfitting.
            assert len(self.betas.shape) == 1, "betas must be 1-D"
            assert (
                len(self.betas) == self.steps
            ), "num of betas must equal to diffusion steps"
            assert (self.betas > 0).all() and (
                self.betas <= 1
            ).all(), "betas out of range"

            self.calculate_for_diffusion()

    def build_histroy_items(self, dataset):
        r"""
        Add time-aware reweighting to the original user-item interaction matrix when config['time-aware'] is True.
        """
        if not self.time_aware:
            super().build_histroy_items(dataset)
        else:
            inter_feat = copy.deepcopy(dataset.inter_feat)
            inter_feat.sort(dataset.time_field)
            user_ids, item_ids = (
                inter_feat[dataset.uid_field].numpy(),
                inter_feat[dataset.iid_field].numpy(),
            )

            w_max = self.w_max
            w_min = self.w_min
            values = np.zeros(len(inter_feat))

            row_num = dataset.user_num
            row_ids, col_ids = user_ids, item_ids

            for uid in range(1, row_num + 1):
                uindex = np.argwhere(user_ids == uid).flatten()
                int_num = len(uindex)
                weight = np.linspace(w_min, w_max, int_num)
                values[uindex] = weight

            history_len = np.zeros(row_num, dtype=np.int64)
            for row_id in row_ids:
                history_len[row_id] += 1

            max_inter_num = np.max(history_len)
            col_num = max_inter_num

            history_matrix = np.zeros((row_num, col_num), dtype=np.int64)
            history_value = np.zeros((row_num, col_num))
            history_len[:] = 0

            for row_id, value, col_id in zip(row_ids, values, col_ids):
                if history_len[row_id] >= col_num:
                    continue
                history_matrix[row_id, history_len[row_id]] = col_id
                history_value[row_id, history_len[row_id]] = value
                history_len[row_id] += 1

            self.history_item_id = torch.LongTensor(history_matrix)
            self.history_item_value = torch.FloatTensor(history_value)
            self.history_item_id = self.history_item_id.to(self.device)
            self.history_item_value = self.history_item_value.to(self.device)

    def get_betas(self):
        r"""
        Given the schedule name, create the betas for the diffusion process.
        """
        if self.noise_schedule == "linear" or self.noise_schedule == "linear-var":
            start = self.noise_scale * self.noise_min
            end = self.noise_scale * self.noise_max
            if self.noise_schedule == "linear":
                return np.linspace(start, end, self.steps, dtype=np.float64)
            else:
                return betas_from_linear_variance(
                    self.steps, np.linspace(start, end, self.steps, dtype=np.float64)
                )
        elif self.noise_schedule == "cosine":
            return betas_for_alpha_bar(
                self.steps, lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
            )
        # Deep Unsupervised Learning using Noneequilibrium Thermodynamics 2.4.1
        elif self.noise_schedule == "binomial":
            ts = np.arange(self.steps)
            betas = [1 / (self.steps - t + 1) for t in ts]
            return betas
        else:
            raise NotImplementedError(f"unknown beta schedule: {self.noise_schedule}!")

    def calculate_for_diffusion(self):
        r"""
        Calculate the coefficients for the diffusion process.
        """
        alphas = 1.0 - self.betas
        # [alpha_{1}, ..., alpha_{1}*...*alpha_{T}] shape (steps,)
        self.alphas_cumprod = torch.cumprod(alphas, axis=0).to(self.device)
        # alpha_{t-1}
        self.alphas_cumprod_prev = torch.cat(
            [torch.tensor([1.0]).to(self.device), self.alphas_cumprod[:-1]]
        ).to(self.device)
        # alpha_{t+1}
        self.alphas_cumprod_next = torch.cat(
            [self.alphas_cumprod[1:], torch.tensor([0.0]).to(self.device)]
        ).to(self.device)
        assert self.alphas_cumprod_prev.shape == (self.steps,)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

        self.posterior_log_variance_clipped = torch.log(
            torch.cat(
                [self.posterior_variance[1].unsqueeze(0), self.posterior_variance[1:]]
            )
        )
        # Eq.10 coef for x_theta
        self.posterior_mean_coef1 = (
            self.betas
            * torch.sqrt(self.alphas_cumprod_prev)
            / (1.0 - self.alphas_cumprod)
        )
        # Eq.10 coef for x_t
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * torch.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )

    def p_sample(self, x_start):
        r"""
        Generate users' interaction probabilities in a denoising manner.
        Args:
            x_start (torch.FloatTensor): the input tensor that contains user's history interaction matrix,
                                         for DiffRec shape: [batch_size, n_items]
                                         for LDiffRec shape: [batch_size, hidden_size]
        Returns:
            torch.FloatTensor: the interaction probabilities,
                               for DiffRec shape: [batch_size, n_items]
                               for LDiffRec shape: [batch_size, hidden_size]
        """
        steps = self.sampling_steps
        if steps == 0:
            x_t = x_start
        else:
            t = torch.tensor([steps - 1] * x_start.shape[0]).to(x_start.device)
            x_t = self.q_sample(x_start, t)

        indices = list(range(self.steps))[::-1]

        if self.noise_scale == 0.0:
            for i in indices:
                t = torch.tensor([i] * x_t.shape[0]).to(x_start.device)
                x_t = self.mlp(x_t, t)
            return x_t

        for i in indices:
            t = torch.tensor([i] * x_t.shape[0]).to(x_start.device)
            out = self.p_mean_variance(x_t, t)
            if self.sampling_noise:
                noise = torch.randn_like(x_t)
                nonzero_mask = (
                    (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))
                )  # no noise when t == 0
                x_t = (
                    out["mean"]
                    + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
                )
            else:
                x_t = out["mean"]
        return x_t

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        x_start = self.get_rating_matrix(user)
        scores = self.p_sample(x_start)
        return scores

    def predict(self, interaction):
        item = interaction[self.ITEM_ID]
        x_t = self.full_sort_predict(interaction)
        scores = x_t[torch.arange(len(item)).to(self.device), item]
        return scores

    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        x_start = self.get_rating_matrix(user)

        batch_size, device = x_start.size(0), x_start.device
        ts, pt = self.sample_timesteps(batch_size, device, "importance")
        noise = torch.randn_like(x_start)
        if self.noise_scale != 0.0:
            x_t = self.q_sample(x_start, ts, noise)
        else:
            x_t = x_start

        model_output = self.mlp(x_t, ts)
        target = {
            ModelMeanType.START_X: x_start,
            ModelMeanType.EPSILON: noise,
        }[self.mean_type]

        assert model_output.shape == target.shape == x_start.shape

        mse = mean_flat((target - model_output) ** 2)

        reloss = self.reweight_loss(x_start, x_t, mse, ts, target, model_output, device)
        self.update_Lt_history(ts, reloss)

        # importance sampling
        reloss /= pt
        mean_loss = reloss.mean()
        return mean_loss

    def reweight_loss(self, x_start, x_t, mse, ts, target, model_output, device):
        if self.reweight:
            if self.mean_type == ModelMeanType.START_X:
                # Eq.11
                weight = self.SNR(ts - 1) - self.SNR(ts)
                # Eq.12
                weight = torch.where((ts == 0), 1.0, weight)
                loss = mse
            elif self.mean_type == ModelMeanType.EPSILON:
                weight = (1 - self.alphas_cumprod[ts]) / (
                    (1 - self.alphas_cumprod_prev[ts]) ** 2 * (1 - self.betas[ts])
                )
                weight = torch.where((ts == 0), 1.0, weight)
                likelihood = mean_flat(
                    (x_start - self._predict_xstart_from_eps(x_t, ts, model_output))
                    ** 2
                    / 2.0
                )
                loss = torch.where((ts == 0), likelihood, mse)
        else:
            weight = torch.tensor([1.0] * len(target)).to(device)
            loss = mse
        reloss = weight * loss
        return reloss

    def update_Lt_history(self, ts, reloss):
        # update Lt_history & Lt_count
        for t, loss in zip(ts, reloss):
            if self.Lt_count[t] == self.history_num_per_term:
                Lt_history_old = self.Lt_history.clone()
                self.Lt_history[t, :-1] = Lt_history_old[t, 1:]
                self.Lt_history[t, -1] = loss.detach()
            else:
                try:
                    self.Lt_history[t, self.Lt_count[t]] = loss.detach()
                    self.Lt_count[t] += 1
                except:
                    print(t)
                    print(self.Lt_count[t])
                    print(loss)
                    raise ValueError

    def sample_timesteps(
        self, batch_size, device, method="uniform", uniform_prob=0.001
    ):
        if method == "importance":  # importance sampling
            if not (self.Lt_count == self.history_num_per_term).all():
                return self.sample_timesteps(batch_size, device, method="uniform")

            Lt_sqrt = torch.sqrt(torch.mean(self.Lt_history**2, axis=-1))
            pt_all = Lt_sqrt / torch.sum(Lt_sqrt)
            pt_all *= 1 - uniform_prob
            pt_all += uniform_prob / len(pt_all)  # ensure the least prob > uniform_prob

            assert pt_all.sum(-1) - 1.0 < 1e-5

            t = torch.multinomial(pt_all, num_samples=batch_size, replacement=True)
            pt = pt_all.gather(dim=0, index=t) * len(pt_all)

            return t, pt

        elif method == "uniform":  # uniform sampling
            t = torch.randint(0, self.steps, (batch_size,), device=device).long()
            pt = torch.ones_like(t).float()

            return t, pt

        else:
            raise ValueError

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            self._extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape)
            * x_start
            + self._extract_into_tensor(
                self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
            )
            * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        r"""
        Compute the mean and variance of the diffusion posterior:
            q(x_{t-1} | x_t, x_0)
        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            self._extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + self._extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self._extract_into_tensor(
            self.posterior_variance, t, x_t.shape
        )
        posterior_log_variance_clipped = self._extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t):
        r"""
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.
        """
        B, C = x.shape[:2]
        assert t.shape == (B,)
        model_output = self.mlp(x, t)

        model_variance = self.posterior_variance
        model_log_variance = self.posterior_log_variance_clipped

        model_variance = self._extract_into_tensor(model_variance, t, x.shape)
        model_log_variance = self._extract_into_tensor(model_log_variance, t, x.shape)

        if self.mean_type == ModelMeanType.START_X:
            pred_xstart = model_output
        elif self.mean_type == ModelMeanType.EPSILON:
            pred_xstart = self._predict_xstart_from_eps(x, t, eps=model_output)
        else:
            raise NotImplementedError(self.mean_type)

        model_mean, _, _ = self.q_posterior_mean_variance(
            x_start=pred_xstart, x_t=x, t=t
        )

        assert (
            model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        )

        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            self._extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape)
            * x_t
            - self._extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
            * eps
        )

    def SNR(self, t):
        r"""
        Compute the signal-to-noise ratio for a single timestep.
        """
        self.alphas_cumprod = self.alphas_cumprod.to(t.device)
        return self.alphas_cumprod[t] / (1 - self.alphas_cumprod[t])

    def _extract_into_tensor(self, arr, timesteps, broadcast_shape):
        r"""
        Extract values from a 1-D torch tensor for a batch of indices.

        Args:
            arr (torch.Tensor): the 1-D torch tensor.
            timesteps (torch.Tensor): a tensor of indices into the array to extract.
            broadcast_shape (torch.Size): a larger shape of K dimensions with the batch
                                dimension equal to the length of timesteps.
        Returns:
            torch.Tensor: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
        """
        # res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
        arr = arr.to(timesteps.device)
        res = arr[timesteps].float()
        while len(res.shape) < len(broadcast_shape):
            res = res[..., None]
        return res.expand(broadcast_shape)


def betas_from_linear_variance(steps, variance, max_beta=0.999):
    alpha_bar = 1 - variance
    betas = []
    betas.append(1 - alpha_bar[0])
    for i in range(1, steps):
        betas.append(min(1 - alpha_bar[i] / alpha_bar[i - 1], max_beta))
    return np.array(betas)


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    r"""
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    Args:
        num_diffusion_timesteps (int): the number of betas to produce.
        alpha_bar (Callable): a lambda that takes an argument t from 0 to 1 and
                   produces the cumulative product of (1-beta) up to that
                   part of the diffusion process.
        max_beta (int): the maximum beta to use; use values lower than 1 to
                  prevent singularities.
    Returns:
        np.ndarray: a 1-D array of beta values.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


def normal_kl(mean1, logvar1, mean2, logvar2):
    r"""
    Compute the KL divergence between two gaussians.

    Shapes are automatically broadcasted, so batches can be compared to
    scalars, among other use cases.
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, torch.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    # Force variances to be Tensors. Broadcasting helps convert scalars to
    # Tensors, but it does not work for torch.exp().
    logvar1, logvar2 = [
        x if isinstance(x, torch.Tensor) else torch.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + torch.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
    )


def mean_flat(tensor):
    r"""
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def timestep_embedding(timesteps, dim, max_period=10000):
    r"""
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional. (N,)
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """

    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32)
        / half
    ).to(
        timesteps.device
    )  # shape (dim//2,)
    args = timesteps[:, None].float() * freqs[None]  # (N, dim//2)
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # (N, (dim//2)*2)
    if dim % 2:
        # zero pad in the last dimension to ensure shape (N, dim)
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding
