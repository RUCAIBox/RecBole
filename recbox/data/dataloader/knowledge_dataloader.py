# @Time   : 2020/7/7
# @Author : Yupeng Hou
# @Email  : houyupeng@ruc.edu.cn

# UPDATE
# @Time   : 2020/9/18, 2020/9/17, 2020/8/31
# @Author : Yupeng Hou, Yushuo Chen, Kaiyuan Li
# @email  : houyupeng@ruc.edu.cn, chenyushuo@ruc.edu.cn, tsotfsk@outlook.com


from recbox.data.dataloader.abstract_dataloader import AbstractDataLoader
from recbox.data.dataloader.general_dataloader import GeneralNegSampleDataLoader
from recbox.data.dataloader.neg_sample_mixin import NegSampleMixin
from recbox.utils import InputType, KGDataLoaderState


class KGDataLoader(NegSampleMixin, AbstractDataLoader):

    def __init__(self, config, dataset, sampler, neg_sample_args,
                 batch_size=1, dl_format=InputType.PAIRWISE, shuffle=False):
        if neg_sample_args['strategy'] != 'by':
            raise ValueError('neg_sample strategy in KnowledgeBasedDataLoader() should be `by`')
        if dl_format != InputType.PAIRWISE or neg_sample_args['by'] != 1:
            raise ValueError('kg based dataloader must be pairwise and can only neg sample by 1')
        if shuffle is False:
            raise ValueError('kg based dataloader must shuffle the data')

        self.batch_size = batch_size
        self.neg_sample_by = neg_sample_args['by']
        self.times = 1

        neg_prefix = config['NEG_PREFIX']
        tid_field = config['TAIL_ENTITY_ID_FIELD']

        # kg negative cols
        neg_kg_col = neg_prefix + tid_field
        dataset.copy_field_property(neg_kg_col, tid_field)

        super().__init__(config, dataset, sampler, neg_sample_args,
                         batch_size=batch_size, dl_format=dl_format, shuffle=shuffle)

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
        hid_field = self.config['HEAD_ENTITY_ID_FIELD']
        tid_field = self.config['TAIL_ENTITY_ID_FIELD']
        hids = kg_feat[hid_field].to_list()
        neg_tids = self.sampler.sample_by_entity_ids(hids, self.neg_sample_by)
        return self._neg_sample_by_pair_wise_sampling(tid_field, neg_tids, kg_feat)

    def _neg_sample_by_pair_wise_sampling(self, tid_field, neg_tids, kg_feat):
        neg_prefix = self.config['NEG_PREFIX']
        neg_tail_entity_id = neg_prefix + tid_field
        kg_feat.insert(len(kg_feat.columns), neg_tail_entity_id, neg_tids)
        return kg_feat

    def _batch_size_adaptation(self):
        self.step = self.batch_size


class KnowledgeBasedDataLoader(AbstractDataLoader):

    def __init__(self, config, dataset, sampler, kg_sampler, neg_sample_args,
                 batch_size=1, dl_format=InputType.POINTWISE, shuffle=False):

        # using sampler
        self.general_dataloader = GeneralNegSampleDataLoader(config=config, dataset=dataset,
                                                             sampler=sampler, neg_sample_args=neg_sample_args,
                                                             batch_size=batch_size, dl_format=dl_format,
                                                             shuffle=shuffle)

        # using kg_sampler and dl_format is pairwise
        self.kg_dataloader = KGDataLoader(config, dataset, kg_sampler, neg_sample_args,
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
        if self.state == KGDataLoaderState.RSKG:
            self.general_dataloader._shuffle()
            self.kg_dataloader._shuffle()
        else:
            self.main_dataloader._shuffle()

    def __next__(self):
        if self.pr >= self.pr_end:
            if self.state == KGDataLoaderState.RSKG:
                self.general_dataloader.pr = 0
                self.kg_dataloader.pr = 0
            else:
                self.pr = 0
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
        if self.state == KGDataLoaderState.RS:
            self.main_dataloader = self.general_dataloader
        elif self.state == KGDataLoaderState.KG:
            self.main_dataloader = self.kg_dataloader
        else:   # RSKG
            kgpr = self.kg_dataloader.pr_end
            rspr = self.general_dataloader.pr_end
            self.main_dataloader = self.general_dataloader if rspr < kgpr else self.kg_dataloader
