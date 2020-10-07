# @Time   : 2020/9/16
# @Author : Yushuo Chen
# @Email  : chenyushuo@ruc.edu.cn

# UPDATE:
# @Time   : 2020/9/16
# @Author : Yushuo Chen
# @Email  : chenyushuo@ruc.edu.cn

import numpy as np
import pandas as pd
import copy

from recbox.data.dataset import Dataset


class SequentialDataset(Dataset):
    def __init__(self, config, saved_dataset=None):
        super().__init__(config, saved_dataset=saved_dataset)

    def prepare_data_augmentation(self):
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
        self.logger.debug(
            'leave one out, group_by=[{}], leave_one_num=[{}]'.format(
                group_by, leave_one_num))
        if group_by is None:
            raise ValueError('leave one out strategy require a group field')

        self.prepare_data_augmentation()
        grouped_index = pd.DataFrame(
            self.uid_list).groupby(by=0).groups.values()
        next_index = self._split_index_by_leave_one_out(
            grouped_index, leave_one_num)
        next_ds = []
        for index in next_index:
            ds = copy.copy(self)
            for field in [
                    'uid_list', 'item_list_index', 'target_index',
                    'item_list_length'
            ]:
                setattr(ds, field, np.array(getattr(ds, field)[index]))
            next_ds.append(ds)
        return next_ds
