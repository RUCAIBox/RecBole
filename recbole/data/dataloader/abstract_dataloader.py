# @Time   : 2020/7/7
# @Author : Yupeng Hou
# @Email  : houyupeng@ruc.edu.cn

# UPDATE
# @Time   : 2022/7/8, 2020/10/22, 2020/9/23, 2022/7/6
# @Author : Zhen Tian, Yupeng Hou, Yushuo Chen, Gaowei Zhang
# @email  : chenyuwuxinn@gmail.com, houyupeng@ruc.edu.cn, chenyushuo@ruc.edu.cn, zgw15630559577@163.com

"""
recbole.data.dataloader.abstract_dataloader
################################################
"""

import math
import copy
from logging import getLogger

import torch

from recbole.data.interaction import Interaction
from recbole.utils import InputType, FeatureType, FeatureSource, ModelType
from recbole.data.transform import construct_transform

start_iter = False


class AbstractDataLoader(torch.utils.data.DataLoader):
    """:class:`AbstractDataLoader` is an abstract object which would return a batch of data which is loaded by
    :class:`~recbole.data.interaction.Interaction` when it is iterated.
    And it is also the ancestor of all other dataloader.

    Args:
        config (Config): The config of dataloader.
        dataset (Dataset): The dataset of dataloader.
        sampler (Sampler): The sampler of dataloader.
        shuffle (bool, optional): Whether the dataloader will be shuffle after a round. Defaults to ``False``.

    Attributes:
        _dataset (Dataset): The dataset of this dataloader.
        shuffle (bool): If ``True``, dataloader will shuffle before every epoch.
        pr (int): Pointer of dataloader.
        step (int): The increment of :attr:`pr` for each batch.
        _batch_size (int): The max interaction number for all batch.
    """

    def __init__(self, config, dataset, sampler, shuffle=False):
        self.shuffle = shuffle
        self.config = config
        self._dataset = dataset
        self._sampler = sampler
        self._batch_size = self.step = self.model = None
        self._init_batch_size_and_step()
        index_sampler = None
        self.generator = torch.Generator()
        self.generator.manual_seed(config["seed"])
        self.transform = construct_transform(config)
        self.is_sequential = config["MODEL_TYPE"] == ModelType.SEQUENTIAL
        if not config["single_spec"]:
            index_sampler = torch.utils.data.distributed.DistributedSampler(
                list(range(self.sample_size)), shuffle=shuffle, drop_last=False
            )
            self.step = max(1, self.step // config["world_size"])
            shuffle = False
        super().__init__(
            dataset=list(range(self.sample_size)),
            batch_size=self.step,
            collate_fn=self.collate_fn,
            num_workers=config["worker"],
            shuffle=shuffle,
            sampler=index_sampler,
            generator=self.generator,
        )

    def _init_batch_size_and_step(self):
        """Initializing :attr:`step` and :attr:`batch_size`."""
        raise NotImplementedError(
            "Method [init_batch_size_and_step] should be implemented"
        )

    def update_config(self, config):
        """Update configure of dataloader, such as :attr:`batch_size`, :attr:`step` etc.

        Args:
            config (Config): The new config of dataloader.
        """
        self.config = config
        self._init_batch_size_and_step()

    def set_batch_size(self, batch_size):
        """Reset the batch_size of the dataloader, but it can't be called when dataloader is being iterated.

        Args:
            batch_size (int): the new batch_size of dataloader.
        """
        self._batch_size = batch_size

    def collate_fn(self):
        """Collect the sampled index, and apply neg_sampling or other methods to get the final data."""
        raise NotImplementedError("Method [collate_fn] must be implemented.")

    def __iter__(self):
        global start_iter
        start_iter = True
        res = super().__iter__()
        start_iter = False
        return res

    def __getattribute__(self, __name: str):
        global start_iter
        if not start_iter and __name == "dataset":
            __name = "_dataset"
        return super().__getattribute__(__name)


class NegSampleDataLoader(AbstractDataLoader):
    """:class:`NegSampleDataLoader` is an abstract class which can sample negative examples by ratio.
    It has two neg-sampling method, the one is 1-by-1 neg-sampling (pair wise),
    and the other is 1-by-multi neg-sampling (point wise).

    Args:
        config (Config): The config of dataloader.
        dataset (Dataset): The dataset of dataloader.
        sampler (Sampler): The sampler of dataloader.
        shuffle (bool, optional): Whether the dataloader will be shuffle after a round. Defaults to ``False``.
    """

    def __init__(self, config, dataset, sampler, shuffle=True):
        self.logger = getLogger()
        super().__init__(config, dataset, sampler, shuffle=shuffle)

    def _set_neg_sample_args(self, config, dataset, dl_format, neg_sample_args):
        self.uid_field = dataset.uid_field
        self.iid_field = dataset.iid_field
        self.dl_format = dl_format
        self.neg_sample_args = neg_sample_args
        self.times = 1
        if (
            self.neg_sample_args["distribution"] in ["uniform", "popularity"]
            and self.neg_sample_args["sample_num"] != "none"
        ):
            self.neg_sample_num = self.neg_sample_args["sample_num"]

            if self.dl_format == InputType.POINTWISE:
                self.times = 1 + self.neg_sample_num
                self.sampling_func = self._neg_sample_by_point_wise_sampling

                self.label_field = config["LABEL_FIELD"]
                dataset.set_field_property(
                    self.label_field, FeatureType.FLOAT, FeatureSource.INTERACTION, 1
                )
            elif self.dl_format == InputType.PAIRWISE:
                self.times = self.neg_sample_num
                self.sampling_func = self._neg_sample_by_pair_wise_sampling

                self.neg_prefix = config["NEG_PREFIX"]
                self.neg_item_id = self.neg_prefix + self.iid_field

                columns = (
                    [self.iid_field]
                    if dataset.item_feat is None
                    else dataset.item_feat.columns
                )
                for item_feat_col in columns:
                    neg_item_feat_col = self.neg_prefix + item_feat_col
                    dataset.copy_field_property(neg_item_feat_col, item_feat_col)
            else:
                raise ValueError(
                    f"`neg sampling by` with dl_format [{self.dl_format}] not been implemented."
                )

        elif (
            self.neg_sample_args["distribution"] != "none"
            and self.neg_sample_args["sample_num"] != "none"
        ):
            raise ValueError(
                f'`neg_sample_args` [{self.neg_sample_args["distribution"]}] is not supported!'
            )

    def _neg_sampling(self, inter_feat):
        if self.neg_sample_args.get("dynamic", False):
            candidate_num = self.neg_sample_args["candidate_num"]
            user_ids = inter_feat[self.uid_field].numpy()
            item_ids = inter_feat[self.iid_field].numpy()
            neg_candidate_ids = self._sampler.sample_by_user_ids(
                user_ids, item_ids, self.neg_sample_num * candidate_num
            )
            self.model.eval()
            interaction = copy.deepcopy(inter_feat).to(self.model.device)
            interaction = interaction.repeat(self.neg_sample_num * candidate_num)
            neg_item_feat = Interaction(
                {self.iid_field: neg_candidate_ids.to(self.model.device)}
            )
            interaction.update(neg_item_feat)
            scores = self.model.predict(interaction).reshape(candidate_num, -1)
            indices = torch.max(scores, dim=0)[1].detach()
            neg_candidate_ids = neg_candidate_ids.reshape(candidate_num, -1)
            neg_item_ids = neg_candidate_ids[
                indices, [i for i in range(neg_candidate_ids.shape[1])]
            ].view(-1)
            self.model.train()
            return self.sampling_func(inter_feat, neg_item_ids)
        elif (
            self.neg_sample_args["distribution"] != "none"
            and self.neg_sample_args["sample_num"] != "none"
        ):
            user_ids = inter_feat[self.uid_field].numpy()
            item_ids = inter_feat[self.iid_field].numpy()
            neg_item_ids = self._sampler.sample_by_user_ids(
                user_ids, item_ids, self.neg_sample_num
            )
            return self.sampling_func(inter_feat, neg_item_ids)
        else:
            return inter_feat

    def _neg_sample_by_pair_wise_sampling(self, inter_feat, neg_item_ids):
        inter_feat = inter_feat.repeat(self.times)
        neg_item_feat = Interaction({self.iid_field: neg_item_ids})
        neg_item_feat = self._dataset.join(neg_item_feat)
        neg_item_feat.add_prefix(self.neg_prefix)
        inter_feat.update(neg_item_feat)
        return inter_feat

    def _neg_sample_by_point_wise_sampling(self, inter_feat, neg_item_ids):
        pos_inter_num = len(inter_feat)
        new_data = inter_feat.repeat(self.times)
        new_data[self.iid_field][pos_inter_num:] = neg_item_ids
        new_data = self._dataset.join(new_data)
        labels = torch.zeros(pos_inter_num * self.times)
        labels[:pos_inter_num] = 1.0
        new_data.update(Interaction({self.label_field: labels}))
        return new_data

    def get_model(self, model):
        self.model = model
