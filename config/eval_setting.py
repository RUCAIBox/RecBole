# -*- coding: utf-8 -*-
# @Author : Yupeng Hou
# @Email  : houyupeng@ruc.edu.cn
# @File   : eval_setting.py

class EvalSetting(object):
    def __init__(self, config):
        self.config = config

        self.group_field = None
        self.ordering_args = None
        self.split_args = None
        self.neg_sample_args = None

        presetting_args = ['group_field', 'ordering_args', 'split_args', 'neg_sample_args']
        for args in presetting_args:
            if config[args] is not None:
                setattr(self, args, config[args])

    def __str__(self):
        info = 'Evaluation Setting:\n'
        info += ('\tGroup by {}\n'.format(self.group_field) if self.group_field is not None else '\tNo Grouping\n')
        info += ('\tOrdering: {}\n'.format(self.ordering_args) if (self.ordering_args is not None and self.ordering_args['strategy'] != 'none') else '\tNo Ordering\n')
        info += ('\tSplitting: {}\n'.format(self.split_args) if (self.split_args is not None and self.split_args['strategy'] != 'none') else '\tNo Splitting\n')
        info += ('\tNegative Sampling: {}'.format(self.neg_sample_args) if (self.neg_sample_args is not None and self.neg_sample_args['strategy'] != 'none') else '\tNo Negative Sampling\n')
        return info

    def __repr__(self):
        return self.__str__()

    r"""Setting about group

    Args:
        field (str): The field of dataset grouped by, default None (Not Grouping)

    Example:
        >>> es.group_by('month')
        >>> es.group_by_user()
    """
    def group_by(self, field=None):
        self.group_field = field

    def group_by_user(self):
        self.group_field = self.config['USER_ID_FIELD']

    r"""Setting about ordering

    Args:
        strategy (str): Either 'none', 'shuffle' or 'by'
        field (str or list of str): Name or list of names
        ascending (bool or list of bool): Sort ascending vs. descending. Specify list for multiple sort orders.
            If this is a list of bools, must match the length of the field

    Example:
        >>> es.set_ordering('shuffle')
        >>> es.set_ordering('by', field='timestamp')
        >>> es.set_ordering('by', field=['timestamp', 'price'], ascending=[True, False])

    or
        >>> es.random_ordering()
        >>> es.sort_by('timestamp') # ascending default
        >>> es.sort_by(field=['timestamp', 'price'], ascending=[True, False])
    """
    def set_ordering(self, strategy='none', **kwargs):
        legal_strategy = {'none', 'shuffle', 'by'}
        if strategy not in legal_strategy:
            raise ValueError('Ordering Strategy [{}] should in {}'.format(strategy, list(legal_strategy)))
        self.ordering_args = {'strategy': strategy}
        self.ordering_args.update(kwargs)

    def random_ordering(self):
        self.set_ordering('shuffle')

    def sort_by(self, field, ascending=None):
        if ascending is None:
            ascending = [True] * len(field)
            if len(ascending) == 1:
                ascending = True
        self.set_ordering('by', field=field, ascending=ascending)

    def temporal_ordering(self):
        self.sort_by(field=self.config['TIMESTAMP_FIELD'])

    r"""Setting about split method

    Args:
        strategy (str): Either 'none', 'by_ratio', 'by_value' or 'loo'.
        ratios (list of float): Dataset will be splited into `len(ratios)` parts.
        field (str): Split by values of field.
        values (list of float or float): Dataset will be splited into `len(values) + 1` parts.
            The first part will be interactions whose field value in (*, values[0]].
        ascending (bool): Order of values after splitting.

    Example:
        >>> es.leave_one_out()
        >>> es.split_by_ratio(ratios=[0.8, 0.1, 0.1])
        >>> es.split_by_value(field='month', values=[6, 7], ascending=False)    # (*, 7], (7, 6], (6, *)
    """
    def set_splitting(self, strategy='none', **kwargs):
        legal_strategy = {'none', 'by_ratio', 'by_value', 'loo'}
        if strategy not in legal_strategy:
            raise ValueError('Split Strategy [{}] should in {}'.format(strategy, list(legal_strategy)))
        if strategy == 'loo' and self.group_by is None:
            raise ValueError('Leave-One-Out request group firstly')
        self.split_args = {'strategy': strategy}
        self.split_args.update(kwargs)

    def leave_one_out(self, leave_one_num=1):
        if self.group_field is None:
            raise ValueError('Leave one out request grouped dataset, please set group field.')
        self.set_splitting(strategy='loo', leave_one_num=leave_one_num)

    def split_by_ratio(self, ratios):
        if not isinstance(ratios, list):
            raise ValueError('ratios [{}] should be list'.format(ratios))
        self.set_splitting(strategy='by_ratio', ratios=ratios)

    def split_by_value(self, field, values, ascending=True):
        if not isinstance(field, str):
            raise ValueError('field [{}] should be str'.format(field))
        if not isinstance(values, list):
            values = [values]
        values.sort(reverse=(not ascending))
        self.set_splitting(strategy='by_value', values=values, ascending=ascending)

    r"""Setting about negative sampling

    Args:
        strategy (str): Either 'none', 'to' or 'by'.
        to (int): Negative Sampling Until `pos + num == to`.
        by (int): Negative Sampling `by` neg cases for one pos case.
        real_time (bool): real time negative sampling if True, else negative cases will be pre-sampled and stored.
        distribution (str): distribution of sampler, either 'uniform' or 'popularity'.

    Example:
        >>> es.neg_sample_to(100, real_time=True)
        >>> es.neg_sample_by(1)
    """
    def set_neg_sampling(self, strategy='none', real_time=False, **kwargs):
        legal_strategy = {'none', 'to', 'by'}
        if strategy not in legal_strategy:
            raise ValueError('Negative Sampling Strategy [{}] should in {}'.format(strategy, list(legal_strategy)))
        distrib = self.neg_sample_args['distribution'] if self.neg_sample_args is not None else 'uniform'
        self.neg_sample_args = {'strategy': strategy, 'real_time': real_time, 'distribution': distrib}
        self.neg_sample_args.update(kwargs)

    def neg_sample_to(self, to, real_time=False):
        self.set_neg_sampling(strategy='to', to=to, real_time=real_time)

    def neg_sample_by(self, by, real_time=False):
        self.set_neg_sampling(strategy='by', by=by, real_time=real_time)

    r"""Setting about sampling distribution

    Args:
        distribution (str): Either 'uniform' or 'popularity'.

    Example:
        >>> es.popularity_based_sampling()
    """
    def set_sampling_distribution(self, distribution='uniform'):
        legal_distribution = {'uniform', 'popularity'}
        if distribution not in legal_distribution:
            raise ValueError('Sampling Distribution [{}] should in {}'.format(distribution, list(legal_distribution)))
        self.neg_sample_args.update({'distribution': distribution})

    def popularity_based_sampling(self):
        self.set_sampling_distribution(distribution='popularity')

    r"""Presets

    RO: Random Ordering         TO: Temporal Ordering
    RS: Ratio-based Splitting   LS: Leave-one-out Splitting     split_by 8:1:1 bu default if RS is setted.
    full: all non-ground-truth items
    uni: uniform sampling       pop: popularity sampling        neg_sample_by 100 by default.
    """
    def RO_RS(self, ratios=[0.8, 0.1, 0.1]):
        self.group_by_user()
        self.random_ordering()
        self.split_by_ratio(ratios)

    def TO_RS(self, ratios=[0.8, 0.1, 0.1]):
        self.group_by_user()
        self.temporal_ordering()
        self.split_by_ratio(ratios)

    def RO_LS(self, leave_one_num=1):
        self.group_by_user()
        self.random_ordering()
        self.leave_one_out(leave_one_num=leave_one_num)

    def TO_LS(self, leave_one_num=1):
        self.group_by_user()
        self.temporal_ordering()
        self.leave_one_out(leave_one_num=leave_one_num)

    def uni100(self, real_time=False):
        self.neg_sample_by(100, real_time=real_time)

    def pop100(self, real_time=False):
        self.neg_sample_by(100, real_time=real_time)
        self.popularity_based_sampling()

    def uni1000(self, real_time=False):
        self.neg_sample_by(1000, real_time=real_time)

    def pop1000(self, real_time=False):
        self.neg_sample_by(1000, real_time=real_time)
        self.popularity_based_sampling()

    def full(self, real_time=True):
        self.neg_sample_to(to=-1, real_time=real_time)
