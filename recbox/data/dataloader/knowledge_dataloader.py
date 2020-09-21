# @Time   : 2020/7/7
# @Author : Yupeng Hou
# @Email  : houyupeng@ruc.edu.cn

# UPDATE
# @Time   : 2020/9/16, 2020/9/21, 2020/8/31
# @Author : Yupeng Hou, Yushuo Chen, Kaiyuan Li
# @email  : houyupeng@ruc.edu.cn, chenyushuo@ruc.edu.cn, tsotfsk@outlook.com


from recbox.data.dataloader import AbstractDataLoader, GeneralNegSampleDataLoader
from recbox.utils import InputType, KGDataLoaderState


class KGDataLoader(AbstractDataLoader):

    def __init__(self, config, dataset, sampler,
                 batch_size=1, dl_format=InputType.PAIRWISE, shuffle=False):
        self.sampler = sampler
        self.neg_sample_num = 1

        self.neg_prefix = config['NEG_PREFIX']
        self.hid_field = dataset.head_entity_field
        self.tid_field = dataset.tail_entity_field

        # kg negative cols
        self.neg_tid_field = self.neg_prefix + self.tid_field
        dataset.copy_field_property(self.neg_tid_field, self.tid_field)

        super().__init__(config, dataset,
                         batch_size=batch_size, dl_format=dl_format, shuffle=shuffle)

    def setup(self):
        if self.shuffle is False:
            self.shuffle = True
            self.logger.warning('kg based dataloader must shuffle the data')

    @property
    def pr_end(self):
        # TODO 这个地方应该是取kg_data的len
        return len(self.dataset.kg_feat)

    def _shuffle(self):
        # TODO 这个地方应该是取kg_data的len
        self.dataset.kg_feat = self.dataset.kg_feat.sample(frac=1).reset_index(drop=True)

    def _next_batch_data(self):
        # TODO 这个地方应该取的kg_data
        cur_data = self.dataset.kg_feat[self.pr: self.pr + self.step]
        self.pr += self.step
        if self.real_time:
            cur_data = self._neg_sampling(cur_data)
        return self._dataframe_to_interaction(cur_data)

    def data_preprocess(self):
        # TODO 这个地方应该是kg_data
        self.dataset.kg_feat = self._neg_sampling(self.dataset.kg_feat)

    def _neg_sampling(self, kg_feat):
        hids = kg_feat[self.hid_field].to_list()
        neg_tids = self.sampler.sample_by_entity_ids(hids, self.neg_sample_num)
        kg_feat.insert(len(kg_feat.columns), self.neg_tid_field, neg_tids)
        return kg_feat


class KnowledgeBasedDataLoader(AbstractDataLoader):

    def __init__(self, config, dataset, sampler, kg_sampler, neg_sample_args,
                 batch_size=1, dl_format=InputType.POINTWISE, shuffle=False):

        # using sampler
        self.general_dataloader = GeneralNegSampleDataLoader(config=config, dataset=dataset,
                                                             sampler=sampler, neg_sample_args=neg_sample_args,
                                                             batch_size=batch_size, dl_format=dl_format,
                                                             shuffle=shuffle)

        # using kg_sampler and dl_format is pairwise
        self.kg_dataloader = KGDataLoader(config, dataset, kg_sampler,
                                          batch_size=batch_size, dl_format=InputType.PAIRWISE, shuffle=shuffle)

        self.main_dataloader = self.general_dataloader

        super().__init__(config, dataset,
                         batch_size=batch_size, dl_format=dl_format, shuffle=shuffle)

    @property
    def pr(self):
        return self.main_dataloader.pr

    @pr.setter
    def pr(self, value):
        self.main_dataloader.pr = value

    def __iter__(self):
        if not hasattr(self, 'state') or not hasattr(self, 'main_dataloader'):
            raise ValueError('The dataloader\'s state and main_dataloader must be set '
                             'when using the kg based dataloader')
        return super().__iter__()

    def _shuffle(self):
        self.main_dataloader._shuffle()
        if self.state == KGDataLoaderState.RSKG:
            self.kg_dataloader._shuffle()

    def __next__(self):
        if self.pr >= self.pr_end:
            self.pr = 0
            # After the rec data ends, the kg data pointer needs to be cleared to zero
            if self.state == KGDataLoaderState.RSKG:
                self.kg_dataloader.pr = 0
            raise StopIteration()
        return self._next_batch_data()

    def __len__(self):
        return len(self.main_dataloader)

    @property
    def pr_end(self):
        return self.main_dataloader.pr_end

    def _next_batch_data(self):
        if self.state == KGDataLoaderState.KG:
            return self.kg_dataloader._next_batch_data()
        elif self.state == KGDataLoaderState.RS:
            return self.general_dataloader._next_batch_data()
        elif self.state == KGDataLoaderState.RSKG:
            kg_data = self.kg_dataloader._next_batch_data()
            rec_data = self.general_dataloader._next_batch_data()
            rec_data.update(kg_data)
            return rec_data

    def set_mode(self, state):
        if state not in set(KGDataLoaderState):
            raise NotImplementedError('kg data loader has no state named [{}]'.format(self.state))
        self.state = state
        if self.state in [KGDataLoaderState.RS, KGDataLoaderState.RSKG]:
            self.main_dataloader = self.general_dataloader
        elif self.state == KGDataLoaderState.KG:
            self.main_dataloader = self.kg_dataloader
