# @Time   : 2020/7/7
# @Author : Yupeng Hou
# @Email  : houyupeng@ruc.edu.cn

# UPDATE
# @Time   : 2020/9/9, 2020/9/17
# @Author : Yupeng Hou, Yushuo Chen
# @email  : houyupeng@ruc.edu.cn, chenyushuo@ruc.edu.cn

"""
recbole.data.dataloader.neg_sample_mixin
################################################
"""

from recbole.data.dataloader.abstract_dataloader import AbstractDataLoader
from recbole.utils import DataLoaderType, EvaluatorType, FeatureSource, FeatureType, InputType


class NegSampleMixin(AbstractDataLoader):
    """:class:`NegSampleMixin` is a abstract class, all dataloaders that need negative sampling should inherit
    this class. This class provides some necessary parameters and method for negative sampling, such as
    :attr:`neg_sample_args` and :meth:`_neg_sampling()` and so on.

    Args:
        config (Config): The config of dataloader.
        dataset (Dataset): The dataset of dataloader.
        sampler (Sampler): The sampler of dataloader.
        neg_sample_args (dict): The neg_sample_args of dataloader.
        batch_size (int, optional): The batch_size of dataloader. Defaults to ``1``.
        dl_format (InputType, optional): The input type of dataloader. Defaults to
            :obj:`~recbole.utils.InputType.POINTWISE`.
        shuffle (bool, optional): Whether the dataloader will be shuffle after a round. Defaluts to ``False``.
    """
    dl_type = DataLoaderType.NEGSAMPLE

    def __init__(self, config, dataset, sampler, neg_sample_args,
                 batch_size=1, dl_format=InputType.POINTWISE, shuffle=False):
        if neg_sample_args['strategy'] not in ['by', 'full']:
            raise ValueError('neg_sample strategy [{}] has not been implemented'.format(neg_sample_args['strategy']))

        self.sampler = sampler
        self.neg_sample_args = neg_sample_args

        super().__init__(config, dataset,
                         batch_size=batch_size, dl_format=dl_format, shuffle=shuffle)

    def setup(self):
        """Do batch size adaptation.
        """
        self._batch_size_adaptation()

    def data_preprocess(self):
        """Do neg-sampling before training/evaluation.
        """
        raise NotImplementedError('Method [data_preprocess] should be implemented.')

    def _batch_size_adaptation(self):
        """Adjust the batch size to ensure that each positive and negative interaction can be in a batch.
        """
        raise NotImplementedError('Method [batch_size_adaptation] should be implemented.')

    def _neg_sampling(self, inter_feat):
        """
        Args:
            inter_feat: The origin user-item interaction table.

        Returns:
            The user-item interaction table with negative example.
        """
        raise NotImplementedError('Method [neg_sampling] should be implemented.')

    def get_pos_len_list(self):
        """
        Returns:
            np.ndarray or list: Number of positive item for each user in a training/evaluating epoch.
        """
        raise NotImplementedError('Method [get_pos_len_list] should be implemented.')


class NegSampleByMixin(NegSampleMixin):
    """:class:`NegSampleByMixin` is an abstract class which can sample negative examples by ratio.
    It has two neg-sampling method, the one is 1-by-1 neg-sampling (pair wise),
    and the other is 1-by-multi neg-sampling (point wise).

    Args:
        config (Config): The config of dataloader.
        dataset (Dataset): The dataset of dataloader.
        sampler (Sampler): The sampler of dataloader.
        neg_sample_args (dict): The neg_sample_args of dataloader.
        batch_size (int, optional): The batch_size of dataloader. Defaults to ``1``.
        dl_format (InputType, optional): The input type of dataloader. Defaults to
            :obj:`~recbole.utils.enum_type.InputType.POINTWISE`.
        shuffle (bool, optional): Whether the dataloader will be shuffle after a round. Defaults to ``False``.
    """
    def __init__(self, config, dataset, sampler, neg_sample_args,
                 batch_size=1, dl_format=InputType.POINTWISE, shuffle=False):
        if neg_sample_args['strategy'] != 'by':
            raise ValueError('neg_sample strategy in GeneralInteractionBasedDataLoader() should be `by`')
        if dl_format == InputType.PAIRWISE and neg_sample_args['by'] != 1:
            raise ValueError('Pairwise dataloader can only neg sample by 1')

        self.user_inter_in_one_batch = (sampler.phase != 'train') and (config['eval_type'] != EvaluatorType.INDIVIDUAL)
        self.neg_sample_by = neg_sample_args['by']

        if dl_format == InputType.POINTWISE:
            self.times = 1 + self.neg_sample_by
            self.sampling_func = self._neg_sample_by_point_wise_sampling

            self.label_field = config['LABEL_FIELD']
            dataset.set_field_property(self.label_field, FeatureType.FLOAT, FeatureSource.INTERACTION, 1)
        elif dl_format == InputType.PAIRWISE:
            self.times = 1
            self.sampling_func = self._neg_sample_by_pair_wise_sampling

            neg_prefix = config['NEG_PREFIX']
            iid_field = config['ITEM_ID_FIELD']
            self.neg_item_id = neg_prefix + iid_field

            columns = [iid_field] if dataset.item_feat is None else dataset.item_feat.columns
            for item_feat_col in columns:
                neg_item_feat_col = neg_prefix + item_feat_col
                dataset.copy_field_property(neg_item_feat_col, item_feat_col)
        else:
            raise ValueError('`neg sampling by` with dl_format [{}] not been implemented'.format(dl_format))

        super().__init__(config, dataset, sampler, neg_sample_args,
                         batch_size=batch_size, dl_format=dl_format, shuffle=shuffle)

    def _neg_sample_by_pair_wise_sampling(self, *args):
        """Pair-wise sampling.
        """
        raise NotImplementedError('Method [neg_sample_by_pair_wise_sampling] should be implemented.')

    def _neg_sample_by_point_wise_sampling(self, *args):
        """Point-wise sampling.
        """
        raise NotImplementedError('Method [neg_sample_by_point_wise_sampling] should be implemented.')
