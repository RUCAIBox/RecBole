# @Time   : 2020/7/7
# @Author : Yupeng Hou
# @Email  : houyupeng@ruc.edu.cn

# UPDATE
# @Time   : 2020/9/9, 2020/9/17
# @Author : Yupeng Hou, Yushuo Chen
# @email  : houyupeng@ruc.edu.cn, chenyushuo@ruc.edu.cn

from recbox.data.dataloader.abstract_dataloader import AbstractDataLoader
from recbox.utils import DataLoaderType, EvaluatorType, FeatureSource, FeatureType, InputType


class NegSampleMixin(AbstractDataLoader):
    dl_type = DataLoaderType.NEGSAMPLE

    def __init__(self,
                 config,
                 dataset,
                 sampler,
                 neg_sample_args,
                 batch_size=1,
                 dl_format=InputType.POINTWISE,
                 shuffle=False):
        if neg_sample_args['strategy'] not in ['by', 'full']:
            raise ValueError(
                'neg_sample strategy [{}] has not been implemented'.format(
                    neg_sample_args['strategy']))

        self.sampler = sampler
        self.neg_sample_args = neg_sample_args

        super().__init__(config,
                         dataset,
                         batch_size=batch_size,
                         dl_format=dl_format,
                         shuffle=shuffle)

    def setup(self):
        self._batch_size_adaptation()

    def data_preprocess(self):
        raise NotImplementedError(
            'Method [data_preprocess] should be implemented.')

    def _batch_size_adaptation(self):
        raise NotImplementedError(
            'Method [batch_size_adaptation] should be implemented.')

    def _neg_sampling(self, inter_feat):
        raise NotImplementedError(
            'Method [neg_sampling] should be implemented.')

    def get_pos_len_list(self):
        raise NotImplementedError(
            'Method [get_pos_len_list] should be implemented.')


class NegSampleByMixin(NegSampleMixin):
    def __init__(self,
                 config,
                 dataset,
                 sampler,
                 neg_sample_args,
                 batch_size=1,
                 dl_format=InputType.POINTWISE,
                 shuffle=False):
        if neg_sample_args['strategy'] != 'by':
            raise ValueError(
                'neg_sample strategy in GeneralInteractionBasedDataLoader() should be `by`'
            )
        if dl_format == InputType.PAIRWISE and neg_sample_args['by'] != 1:
            raise ValueError('Pairwise dataloader can only neg sample by 1')

        self.user_inter_in_one_batch = (sampler.phase != 'train') and (
            config['eval_type'] != EvaluatorType.INDIVIDUAL)
        self.neg_sample_by = neg_sample_args['by']

        # TODO self.times 改个名（有点意义不明）
        if dl_format == InputType.POINTWISE:
            self.times = 1 + self.neg_sample_by
            self.sampling_func = self._neg_sample_by_point_wise_sampling

            self.label_field = config['LABEL_FIELD']
            dataset.set_field_property(self.label_field, FeatureType.FLOAT,
                                       FeatureSource.INTERACTION, 1)
        elif dl_format == InputType.PAIRWISE:
            self.times = 1
            self.sampling_func = self._neg_sample_by_pair_wise_sampling

            neg_prefix = config['NEG_PREFIX']
            iid_field = config['ITEM_ID_FIELD']
            self.neg_item_id = neg_prefix + iid_field

            columns = [
                iid_field
            ] if dataset.item_feat is None else dataset.item_feat.columns
            for item_feat_col in columns:
                neg_item_feat_col = neg_prefix + item_feat_col
                dataset.copy_field_property(neg_item_feat_col, item_feat_col)
        else:
            raise ValueError(
                '`neg sampling by` with dl_format [{}] not been implemented'.
                format(dl_format))

        super().__init__(config,
                         dataset,
                         sampler,
                         neg_sample_args,
                         batch_size=batch_size,
                         dl_format=dl_format,
                         shuffle=shuffle)

    def _neg_sample_by_pair_wise_sampling(self, *args):
        raise NotImplementedError(
            'Method [neg_sample_by_pair_wise_sampling] should be implemented.')

    def _neg_sample_by_point_wise_sampling(self, *args):
        raise NotImplementedError(
            'Method [neg_sample_by_point_wise_sampling] should be implemented.'
        )
