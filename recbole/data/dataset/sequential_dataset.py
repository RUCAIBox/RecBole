# @Time   : 2020/9/16
# @Author : Yushuo Chen
# @Email  : chenyushuo@ruc.edu.cn

# UPDATE:
# @Time   : 2020/9/16
# @Author : Yushuo Chen
# @Email  : chenyushuo@ruc.edu.cn

"""
recbole.data.sequential_dataset
###############################
"""

import numpy as np
import pandas as pd
import copy

from recbole.data.dataset import Dataset


class SequentialDataset(Dataset):
    """:class:`SequentialDataset` is based on :class:`~recbole.data.dataset.dataset.Dataset`,
    and provides augmentation interface to adapt to Sequential Recommendation,
    which can accelerate the data loader.

    Attributes:
        uid_list (numpy.ndarray): List of user id after augmentation.

        item_list_index (numpy.ndarray): List of indexes of item sequence after augmentation.

        target_index (numpy.ndarray): List of indexes of target item id after augmentation.

        item_list_length (numpy.ndarray): List of item sequences' length after augmentation.

    """
    def __init__(self, config, saved_dataset=None):
        super().__init__(config, saved_dataset=saved_dataset)

    def prepare_data_augmentation(self):
        """Augmentation processing for sequential dataset.

        E.g., ``u1`` has purchase sequence ``<i1, i2, i3, i4>``,
        then after augmentation, we will generate three cases.

        ``u1, <i1> | i2``

        (Which means given user_id ``u1`` and item_seq ``<i1>``,
        we need to predict the next item ``i2``.)

        The other cases are below:

        ``u1, <i1, i2> | i3``

        ``u1, <i1, i2, i3> | i4``

        Returns:
            Tuple of ``self.uid_list``, ``self.item_list_index``,
            ``self.target_index``, ``self.item_list_length``.
            See :class:`SequentialDataset`'s attributes for details.

        Note:
            Actually, we do not realy generate these new item sequences.
            One user's item sequence is stored only once in memory.
            We store the index (slice) of each item sequence after augmentation,
            which saves memory and accelerates a lot.
        """
        self.logger.debug('prepare_data_augmentation')
        if hasattr(self, 'uid_list'):
            return self.uid_list, self.item_list_index, self.target_index, self.item_list_length

        self._check_field('uid_field', 'time_field')
        max_item_list_len = self.config['MAX_ITEM_LIST_LENGTH']
        self.sort(by=[self.uid_field, self.time_field], ascending=True)
        last_uid = None
        uid_list, item_list_index, target_index, item_list_length = [], [], [], []
        seq_start = 0
        for i, uid in enumerate(self.inter_feat[self.uid_field].values):
            if last_uid != uid:
                last_uid = uid
                seq_start = i
            else:
                if i - seq_start > max_item_list_len:
                    seq_start += 1
                uid_list.append(uid)
                item_list_index.append(slice(seq_start, i))
                target_index.append(i)
                item_list_length.append(i - seq_start)

        self.uid_list = np.array(uid_list)
        self.item_list_index = np.array(item_list_index)
        self.target_index = np.array(target_index)
        self.item_list_length = np.array(item_list_length)
        return self.uid_list, self.item_list_index, self.target_index, self.item_list_length

    def leave_one_out(self, group_by, leave_one_num=1):
        self.logger.debug('leave one out, group_by=[{}], leave_one_num=[{}]'.format(group_by, leave_one_num))
        if group_by is None:
            raise ValueError('leave one out strategy require a group field')

        self.prepare_data_augmentation()
        grouped_index = pd.DataFrame(self.uid_list).groupby(by=0).groups.values()
        next_index = self._split_index_by_leave_one_out(grouped_index, leave_one_num)
        next_ds = []
        for index in next_index:
            ds = copy.copy(self)
            for field in ['uid_list', 'item_list_index', 'target_index', 'item_list_length']:
                setattr(ds, field, np.array(getattr(ds, field)[index]))
            next_ds.append(ds)
        return next_ds
