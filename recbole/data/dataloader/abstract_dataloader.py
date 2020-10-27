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

from recbole.utils import InputType


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

    def __init__(self, config, dataset,
                 batch_size=1, dl_format=InputType.POINTWISE, shuffle=False):
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

        self.join = self.dataset.join
        self.history_item_matrix = self.dataset.history_item_matrix
        self.history_user_matrix = self.dataset.history_user_matrix
        self.inter_matrix = self.dataset.inter_matrix

        for dataset_attr in self.dataset._dataloader_apis:
            try:
                flag = hasattr(self.dataset, dataset_attr)
                if flag:
                    setattr(self, dataset_attr, getattr(self.dataset, dataset_attr))
            except:
                continue

        self.setup()
        if not self.real_time:
            self.data_preprocess()

    def setup(self):
        """This function can be used to deal with some problems after essential args are initialized,
        such as the batch-size-adaptation when neg-sampling is needed, and so on. By default, it will do nothing.
        """
        pass

    def data_preprocess(self):
        """This function is used to do some data preprocess, such as pre-neg-sampling and pre-data-augmentation.
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
            self.logger.warning('Batch size is changed to {}'.format(batch_size))

    def get_user_feature(self):
        """It is similar to :meth:`~recbole.data.dataset.dataset.Dataset.get_user_feature`, but it will return an
        :class:`~recbole.data.interaction.Interaction` of user feature instead of a :class:`pandas.DataFrame`.

        Returns:
            Interaction: The interaction of user feature.
        """
        user_df = self.dataset.get_user_feature()
        return self._dataframe_to_interaction(user_df)

    def get_item_feature(self):
        """It is similar to :meth:`~recbole.data.dataset.dataset.Dataset.get_item_feature`, but it will return an
        :class:`~recbole.data.interaction.Interaction` of item feature instead of a :class:`pandas.DataFrame`.

        Returns:
            Interaction: The interaction of item feature.
        """
        item_df = self.dataset.get_item_feature()
        return self._dataframe_to_interaction(item_df)
