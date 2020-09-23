# @Time   : 2020/7/7
# @Author : Yupeng Hou
# @Email  : houyupeng@ruc.edu.cn

# UPDATE
# @Time   : 2020/9/16, 2020/9/16
# @Author : Yupeng Hou, Yushuo Chen
# @email  : houyupeng@ruc.edu.cn, chenyushuo@ruc.edu.cn

import math

from recbox.utils import InputType


class AbstractDataLoader(object):
    dl_type = None

    def __init__(self, config, dataset,
                 batch_size=1, dl_format=InputType.POINTWISE, shuffle=False):
        self.config = config
        self.dataset = dataset
        self.batch_size = batch_size
        self.step = batch_size
        self.dl_format = dl_format
        self.shuffle = shuffle
        self.pr = 0
        self.real_time = config['real_time_process'] or True

        self.join = self.dataset.join
        self.history_item_matrix = self.dataset.history_item_matrix
        self.history_user_matrix = self.dataset.history_user_matrix
        self.inter_matrix = self.dataset.inter_matrix

        optional_attrs = [
            'kg_graph', 'ckg_graph', 'relation_num', 'entity_num',
            'head_entities', 'tail_entities', 'relations', 'entities',
            'net_matrix'
        ]
        for op_attr in optional_attrs:
            if hasattr(self.dataset, op_attr):
                setattr(self, op_attr, getattr(self.dataset, op_attr))

        self.logger = self.dataset.logger
        self.num = self.dataset.num
        self.fields = self.dataset.fields
        self.get_preload_weight = self.dataset.get_preload_weight
        self.field2type = self.dataset.field2type
        self.field2source = self.dataset.field2source
        self.field2id_token = self.dataset.field2id_token
        if self.dataset.uid_field:
            self.user_num = self.dataset.user_num
        if self.dataset.iid_field:
            self.item_num = self.dataset.item_num
        self._dataframe_to_interaction = self.dataset._dataframe_to_interaction
        self._dict_to_interaction = self.dataset._dict_to_interaction

        self.setup()
        if not self.real_time:
            self.data_preprocess()

    def setup(self):
        pass

    def data_preprocess(self):
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
        raise NotImplementedError('Method [pr_end] should be implemented')

    def _shuffle(self):
        raise NotImplementedError('Method [shuffle] should be implemented.')

    def _next_batch_data(self):
        raise NotImplementedError('Method [next_batch_data] should be implemented.')

    def set_batch_size(self, batch_size):
        if self.pr != 0:
            raise PermissionError('Cannot change dataloader\'s batch_size while iteration')
        if self.batch_size != batch_size:
            self.batch_size = batch_size
            self.logger.warning('Batch size is changed to {}'.format(batch_size))

    def get_user_feature(self):
        user_df = self.dataset.get_user_feature()
        return self._dataframe_to_interaction(user_df)

    def get_item_feature(self):
        item_df = self.dataset.get_item_feature()
        return self._dataframe_to_interaction(item_df)
