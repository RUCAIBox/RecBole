# @Time   : 2020/9/16
# @Author : Yushuo Chen
# @Email  : chenyushuo@ruc.edu.cn

# UPDATE:
# @Time   : 2020/9/16, 2021/7/1, 2021/7/8
# @Author : Yushuo Chen, Xingyu Pan, Yupeng Hou
# @Email  : chenyushuo@ruc.edu.cn, xy_pan@foxmail.com, houyupeng@ruc.edu.cn

"""
recbole.data.sequential_dataset
###############################
"""

import copy

import numpy as np

from recbole.data.dataset import Dataset


class SequentialDataset(Dataset):
    """:class:`SequentialDataset` is based on :class:`~recbole.data.dataset.dataset.Dataset`,
    and provides augmentation interface to adapt to Sequential Recommendation,
    which can accelerate the data loader.

    Attributes:
        augmentation (bool): Whether the interactions should be augmented in RecBole.

        uid_list (numpy.ndarray): List of user id after augmentation.

        item_list_index (numpy.ndarray): List of indexes of item sequence after augmentation.

        target_index (numpy.ndarray): List of indexes of target item id after augmentation.

        item_list_length (numpy.ndarray): List of item sequences' length after augmentation.

    """

    def __init__(self, config):
        self.augmentation = config['augmentation']
        super().__init__(config)

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

        Note:
            Actually, we do not really generate these new item sequences.
            One user's item sequence is stored only once in memory.
            We store the index (slice) of each item sequence after augmentation,
            which saves memory and accelerates a lot.
        """
        self.logger.debug('prepare_data_augmentation')

        self._check_field('uid_field', 'time_field')
        max_item_list_len = self.config['MAX_ITEM_LIST_LENGTH']
        self.sort(by=[self.uid_field, self.time_field], ascending=True)
        last_uid = None
        uid_list, item_list_index, target_index, item_list_length = [], [], [], []
        seq_start = 0
        for i, uid in enumerate(self.inter_feat[self.uid_field].numpy()):
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
        self.item_list_length = np.array(item_list_length, dtype=np.int64)
        self.mask = np.ones(len(self.inter_feat), dtype=np.bool)

    def leave_one_out(self, group_by, leave_one_num=1):
        self.logger.debug(f'Leave one out, group_by=[{group_by}], leave_one_num=[{leave_one_num}].')
        if group_by is None:
            raise ValueError('Leave one out strategy require a group field.')
        if group_by != self.uid_field:
            raise ValueError('Sequential models require group by user.')

        self.prepare_data_augmentation()
        grouped_index = self._grouped_index(self.uid_list)
        next_index = self._split_index_by_leave_one_out(grouped_index, leave_one_num)

        self._drop_unused_col()
        next_ds = []
        for index in next_index:
            ds = copy.copy(self)
            for field in ['uid_list', 'item_list_index', 'target_index', 'item_list_length']:
                setattr(ds, field, np.array(getattr(ds, field)[index]))
            setattr(ds, 'mask', np.ones(len(self.inter_feat), dtype=np.bool))
            next_ds.append(ds)
        next_ds[0].mask[self.target_index[next_index[1] + next_index[2]]] = False
        next_ds[1].mask[self.target_index[next_index[2]]] = False
        return next_ds

    def inter_matrix(self, form='coo', value_field=None):
        """Get sparse matrix that describe interactions between user_id and item_id.
        Sparse matrix has shape (user_num, item_num).
        For a row of <src, tgt>, ``matrix[src, tgt] = 1`` if ``value_field`` is ``None``,
        else ``matrix[src, tgt] = self.inter_feat[src, tgt]``.

        Args:
            form (str, optional): Sparse matrix format. Defaults to ``coo``.
            value_field (str, optional): Data of sparse matrix, which should exist in ``df_feat``.
                Defaults to ``None``.

        Returns:
            scipy.sparse: Sparse matrix in form ``coo`` or ``csr``.
        """
        if not self.uid_field or not self.iid_field:
            raise ValueError('dataset does not exist uid/iid, thus can not converted to sparse matrix.')

        self.logger.warning(
            'Load interaction matrix may lead to label leakage from testing phase, this implementation '
            'only provides the interactions corresponding to specific phase'
        )
        local_inter_feat = self.inter_feat[self.mask]  # TODO: self.mask will applied to _history_matrix() in future
        return self._create_sparse_matrix(local_inter_feat, self.uid_field, self.iid_field, form, value_field)

    def build(self):

        self._change_feat_format()

        
        ordering_args = self.config['eval_args']['order']
        if ordering_args == 'RO':
            raise ValueError('Ordering strategy `shuffle` is not supported in sequential models.')

        group_by = self.config['eval_args']['group_by']
        if group_by != 'user':
            raise ValueError('The data splitting for Sequential models must be grouped by user.')
            
        split_args = self.config['eval_args']['split']
        if split_args is None:
            raise ValueError('The split_args in eval_args should not be None.')
        if isinstance(split_args, dict) != True:
            raise ValueError(f'The split_args [{split_args}] should be a dict.')

        split_mode = list(split_args.keys())[0] 
        if split_mode == 'LS':
            return self.leave_one_out(group_by=self.config['USER_ID_FIELD'], leave_one_num=split_args['LS'])
        else:
            ValueError('Sequential models require `LS` (leave one out) split strategy.')
