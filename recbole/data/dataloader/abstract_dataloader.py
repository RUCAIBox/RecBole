# @Time   : 2020/7/7
# @Author : Yupeng Hou
# @Email  : houyupeng@ruc.edu.cn

# UPDATE
# @Time   : 2020/10/22, 2020/9/23
# @Author : Yupeng Hou, Yushuo Chen
# @email  : houyupeng@ruc.edu.cn, chenyushuo@ruc.edu.cn

"""
recbole.data.dataloader.abstract_dataloader
################################################
"""

import math
from logging import getLogger

import torch

from recbole.data.interaction import Interaction
from recbole.utils import InputType, FeatureType, FeatureSource


class AbstractDataLoader(object):
    """:class:`AbstractDataLoader` is an abstract object which would return a batch of data which is loaded by
    :class:`~recbole.data.interaction.Interaction` when it is iterated.
    And it is also the ancestor of all other dataloader.

    Args:
        config (Config): The config of dataloader.
        dataset (Dataset): The dataset of dataloader.
        batch_size (int, optional): The batch_size of dataloader. Defaults to ``1``.
        dl_format (InputType, optional): The input type of dataloader. Defaults to
            :obj:`~recbole.utils.enum_type.InputType.POINTWISE`.
        shuffle (bool, optional): Whether the dataloader will be shuffle after a round. Defaults to ``False``.

    Attributes:
        dataset (Dataset): The dataset of this dataloader.
        shuffle (bool): If ``True``, dataloader will shuffle before every epoch.
        real_time (bool): If ``True``, dataloader will do data pre-processing,
            such as neg-sampling and data-augmentation.
        pr (int): Pointer of dataloader.
        step (int): The increment of :attr:`pr` for each batch.
        batch_size (int): The max interaction number for all batch.
    """
    dl_type = None

    def __init__(self, config, dataset, batch_size=1, dl_format=InputType.POINTWISE, shuffle=False):
        self.config = config
        self.logger = getLogger()
        self.dataset = dataset
        self.batch_size = batch_size
        self.step = batch_size
        self.dl_format = dl_format
        self.shuffle = shuffle
        self.pr = 0
        self.real_time = config['real_time_process']
        if self.real_time is None:
            self.real_time = True

        self.setup()
        if not self.real_time:
            self.data_preprocess()

    def setup(self):
        """This function can be used to deal with some problems after essential args are initialized,
        such as the batch-size-adaptation when neg-sampling is needed, and so on. By default, it will do nothing.
        """
        pass

    def data_preprocess(self):
        """This function is used to do some data preprocess, such as pre-data-augmentation.
        By default, it will do nothing.
        """
        pass

    def __len__(self):
        return math.ceil(self.pr_end / self.step)

    def __iter__(self):
        if self.shuffle:
            self._shuffle()
        return self

    def __next__(self):
        if self.pr >= self.pr_end:
            self.pr = 0
            raise StopIteration()
        return self._next_batch_data()

    @property
    def pr_end(self):
        """This property marks the end of dataloader.pr which is used in :meth:`__next__()`."""
        raise NotImplementedError('Method [pr_end] should be implemented')

    def _shuffle(self):
        """Shuffle the order of data, and it will be called by :meth:`__iter__()` if self.shuffle is True.
        """
        raise NotImplementedError('Method [shuffle] should be implemented.')

    def _next_batch_data(self):
        """Assemble next batch of data in form of Interaction, and return these data.

        Returns:
            Interaction: The next batch of data.
        """
        raise NotImplementedError('Method [next_batch_data] should be implemented.')

    def set_batch_size(self, batch_size):
        """Reset the batch_size of the dataloader, but it can't be called when dataloader is being iterated.

        Args:
            batch_size (int): the new batch_size of dataloader.
        """
        if self.pr != 0:
            raise PermissionError('Cannot change dataloader\'s batch_size while iteration')
        if self.batch_size != batch_size:
            self.batch_size = batch_size
            self.logger.warning(f'Batch size is changed to {batch_size}.')

    def upgrade_batch_size(self, batch_size):
        """Upgrade the batch_size of the dataloader, if input batch_size is bigger than current batch_size.

        Args:
            batch_size (int): the new batch_size of dataloader.
        """
        if self.batch_size < batch_size:
            self.set_batch_size(batch_size)


class NewAbstractDataLoader:
    """:class:`AbstractDataLoader` is an abstract object which would return a batch of data which is loaded by
    :class:`~recbole.data.interaction.Interaction` when it is iterated.
    And it is also the ancestor of all other dataloader.

    Args:
        config (Config): The config of dataloader.
        dataset (Dataset): The dataset of dataloader.
        shuffle (bool, optional): Whether the dataloader will be shuffle after a round. Defaults to ``False``.

    Attributes:
        dataset (Dataset): The dataset of this dataloader.
        shuffle (bool): If ``True``, dataloader will shuffle before every epoch.
        pr (int): Pointer of dataloader.
        step (int): The increment of :attr:`pr` for each batch.
        batch_size (int): The max interaction number for all batch.
    """
    dl_type = None

    def __init__(self, config, dataset, sampler, shuffle=False):
        self.config = config
        self.logger = getLogger()
        self.dataset = dataset
        self.sampler = sampler
        self.batch_size = self.step = None
        self.shuffle = shuffle
        self.pr = 0
        self._init_batch_size_and_step()

    def _init_batch_size_and_step(self):
        raise NotImplementedError('Method [_init_batch_size_and_step] should be implemented')

    def __len__(self):
        return math.ceil(self.pr_end / self.step)

    def __iter__(self):
        if self.shuffle:
            self._shuffle()
        return self

    def __next__(self):
        if self.pr >= self.pr_end:
            self.pr = 0
            raise StopIteration()
        return self._next_batch_data()

    @property
    def pr_end(self):
        """This property marks the end of dataloader.pr which is used in :meth:`__next__()`."""
        raise NotImplementedError('Method [pr_end] should be implemented')

    def _shuffle(self):
        """Shuffle the order of data, and it will be called by :meth:`__iter__()` if self.shuffle is True.
        """
        raise NotImplementedError('Method [shuffle] should be implemented.')

    def _next_batch_data(self):
        """Assemble next batch of data in form of Interaction, and return these data.

        Returns:
            Interaction: The next batch of data.
        """
        raise NotImplementedError('Method [next_batch_data] should be implemented.')

    def set_batch_size(self, batch_size):
        """Reset the batch_size of the dataloader, but it can't be called when dataloader is being iterated.

        Args:
            batch_size (int): the new batch_size of dataloader.
        """
        if self.pr != 0:
            raise PermissionError('Cannot change dataloader\'s batch_size while iteration')
        if self.batch_size != batch_size:
            self.batch_size = batch_size


class NewNegSampleDataLoader(NewAbstractDataLoader):
    def __init__(self, config, dataset, sampler, shuffle=True):
        super().__init__(config, dataset, sampler, shuffle=shuffle)

    def _set_neg_sample_args(self, config, dataset, dl_format, neg_sample_args):
        self.uid_field = dataset.uid_field
        self.iid_field = dataset.iid_field
        self.dl_format = dl_format
        self.neg_sample_args = neg_sample_args
        if self.neg_sample_args['strategy'] == 'by':
            self.neg_sample_by = self.neg_sample_args['by']

            if self.dl_format == InputType.POINTWISE:
                self.times = 1 + self.neg_sample_by
                self.sampling_func = self._neg_sample_by_point_wise_sampling

                self.label_field = config['LABEL_FIELD']
                dataset.set_field_property(self.label_field, FeatureType.FLOAT, FeatureSource.INTERACTION, 1)
            elif self.dl_format == InputType.PAIRWISE:
                self.times = self.neg_sample_by
                self.sampling_func = self._neg_sample_by_pair_wise_sampling

                self.neg_prefix = config['NEG_PREFIX']
                self.neg_item_id = self.neg_prefix + self.iid_field

                columns = [self.iid_field] if dataset.item_feat is None else dataset.item_feat.columns
                for item_feat_col in columns:
                    neg_item_feat_col = self.neg_prefix + item_feat_col
                    dataset.copy_field_property(neg_item_feat_col, item_feat_col)
            else:
                raise ValueError(f'`neg sampling by` with dl_format [{self.dl_format}] not been implemented.')

        elif self.neg_sample_args['strategy'] != 'none':
            raise ValueError(f'`neg_sample_args` [{self.neg_sample_args["strategy"]}] is not supported!')

    def _neg_sampling(self, inter_feat):
        if self.neg_sample_args['strategy'] == 'by':
            user_ids = inter_feat[self.uid_field]
            neg_item_ids = self.sampler.sample_by_user_ids(user_ids, self.neg_sample_by)
            return self.sampling_func(inter_feat, neg_item_ids)
        else:
            return inter_feat

    def _neg_sample_by_pair_wise_sampling(self, inter_feat, neg_item_ids):
        inter_feat = inter_feat.repeat(self.times)
        neg_item_feat = Interaction({self.iid_field: neg_item_ids})
        neg_item_feat = self.dataset.join(neg_item_feat)
        neg_item_feat.add_prefix(self.neg_prefix)
        inter_feat.update(neg_item_feat)
        return inter_feat

    def _neg_sample_by_point_wise_sampling(self, inter_feat, neg_item_ids):
        pos_inter_num = len(inter_feat)
        new_data = inter_feat.repeat(self.times)
        new_data[self.iid_field][pos_inter_num:] = neg_item_ids
        new_data = self.dataset.join(new_data)
        labels = torch.zeros(pos_inter_num * self.times)
        labels[:pos_inter_num] = 1.0
        new_data.update(Interaction({self.label_field: labels}))
        return new_data
