# @Time   : 2020/9/23
# @Author : Yushuo Chen
# @Email  : chenyushuo@ruc.edu.cn

# UPDATE
# @Time   : 2020/9/23
# @Author : Yushuo Chen
# @email  : chenyushuo@ruc.edu.cn

"""
recbole.data.dataloader.user_dataloader
################################################
"""

import numpy as np
import torch

from recbole.data.dataloader import AbstractDataLoader
from recbole.utils.enum_type import DataLoaderType, InputType
from recbole.data.interaction import Interaction


class UserDataLoader(AbstractDataLoader):
    """:class:`UserDataLoader` will return a batch of data which only contains user-id when it is iterated.

    Args:
        config (Config): The config of dataloader.
        dataset (Dataset): The dataset of dataloader.
        batch_size (int, optional): The batch_size of dataloader. Defaults to ``1``.
        dl_format (InputType, optional): The input type of dataloader. Defaults to
            :obj:`~recbole.utils.enum_type.InputType.POINTWISE`.
        shuffle (bool, optional): Whether the dataloader will be shuffle after a round. Defaults to ``False``.

    Attributes:
        shuffle (bool): Whether the dataloader will be shuffle after a round.
            However, in :class:`UserDataLoader`, it's guaranteed to be ``True``.
    """
    dl_type = DataLoaderType.ORIGIN

    def __init__(self, config, dataset,
                 batch_size=1, dl_format=InputType.POINTWISE, shuffle=False):
        self.uid_field = dataset.uid_field

        super().__init__(config=config, dataset=dataset,
                         batch_size=batch_size, dl_format=dl_format, shuffle=shuffle)

    def setup(self):
        """Make sure that the :attr:`shuffle` is True. If :attr:`shuffle` is False, it will be changed to True
        and give a warning to user.
        """
        if self.shuffle is False:
            self.shuffle = True
            self.logger.warning('UserDataLoader must shuffle the data')

    @property
    def pr_end(self):
        return len(self.dataset.user_feat)

    def _shuffle(self):
        self.dataset.user_feat.shuffle()

    def _next_batch_data(self):
        cur_data = self.dataset.user_feat[self.pr: self.pr + self.step]
        self.pr += self.step
        return cur_data


class AutoEncoderUserDataloader(AbstractDataLoader):
    """:class:`AutoEncoderUserDataloader` is a general-dataloader that only load user id.
    For the result of every batch, we only return batch size of user id.

    Args:
        config (Config): The config of dataloader.
        dataset (Dataset): The dataset of dataloader.
        batch_size (int, optional): The batch_size of dataloader. Defaults to ``1``.
        dl_format (InputType, optional): The input type of dataloader. Defaults to
            :obj:`~recbole.utils.enum_type.InputType.POINTWISE`.
        shuffle (bool, optional): Whether the dataloader will be shuffle after a round. Defaults to ``False``.
    """
    def __init__(self, config, dataset,
                 batch_size=1, dl_format=InputType.POINTWISE, shuffle=False):

        self.uid_field = dataset.uid_field
        self.uid2index = np.unique(dataset.inter_feat[self.uid_field].numpy())

        super().__init__(config, dataset, batch_size=batch_size, dl_format=dl_format, shuffle=shuffle)

    @property
    def pr_end(self):
        return len(self.uid2index)

    def _shuffle(self):
        np.random.shuffle(self.uid2index)

    def _next_batch_data(self):
        cur_data = self.uid2index[self.pr: self.pr + self.step]
        self.pr += self.step
        return Interaction({self.uid_field: torch.tensor(cur_data)})