# @Time   : 2020/7/20
# @Author : Yupeng Hou
# @Email  : houyupeng@ruc.edu.cn

# UPDATE:
# @Time   : 2020/10/22, 2020/8/31
# @Author : Yupeng Hou, Yushuo Chen
# @Email  : houyupeng@ruc.edu.cn, chenyushuo@ruc.edu.cn


"""
recbole.config.eval_setting
################################
"""


class EvalSetting(object):
    """Class containing settings about model evaluation.

    Evaluation setting contains four parts:
        * Group
        * Sort
        * Split
        * Negative Sample

    APIs are provided for users to set up or modify their evaluation setting easily and clearly.

    Besides, some presets are provided, which is more recommended.

    For example:
        RO: Random Ordering
        TO: Temporal Ordering

        RS: Ratio-based Splitting
        LS: Leave-one-out Splitting

        full: adopt the entire item set (excluding ground-truth items) for ranking
        uniXX: uniform sampling XX items while negative sampling
        popXX: popularity-based sampling XX items while negative sampling

    Note that records are grouped by user_id by default if you use these presets.

    Thus you can use `RO_RS, full` to represent Shuffle, Grouped by user, Ratio-based Splitting
    and Evaluate all non-ground-truth items.

    Check out *Revisiting Alternative Experimental Settings for Evaluating Top-N Item Recommendation Algorithms*
    Wayne Xin Zhao et.al. CIKM 2020 to figure out the details about presets of evaluation settings.

    Args:
        config (Config): Global configuration object.

    Attributes:
        group_field (str or None): Don't group if ``None``, else group by field before splitting.
            Usually records are grouped by user id.

        ordering_args (dict): Args about ordering.
            Usually records are sorted by timestamp, or shuffled.

        split_args (dict): Args about splitting.
            usually records are splitted by ratio (eg. 8:1:1),
            or by 'leave one out' strategy, which means the last purchase record
            of one user is used for evaluation.

        neg_sample_args (dict): Args about negative sampling.
            Negative sample is used wildly in training and evaluating.

            We provide two strategies:

            - ``neg_sample_by``:  sample several negative records for each positive records.
            - ``full_sort``:      don't negative sample, while all unused items are used for evaluation.

    """

    def __init__(self, config):
        self.config = config

        self.group_field = None
        self.ordering_args = None
        self.split_args = None
        self.neg_sample_args = {'strategy': 'none'}

        presetting_args = ['group_field', 'ordering_args', 'split_args', 'neg_sample_args']
        for args in presetting_args:
            if config[args] is not None:
                setattr(self, args, config[args])

    def __str__(self):
        info = ['Evaluation Setting:']

        if self.group_field:
            info.append('Group by {}'.format(self.group_field))
        else:
            info.append('No Grouping')

        if self.ordering_args is not None and self.ordering_args['strategy'] != 'none':
            info.append('Ordering: {}'.format(self.ordering_args))
        else:
            info.append('No Ordering')

        if self.split_args is not None and self.split_args['strategy'] != 'none':
            info.append('Splitting: {}'.format(self.split_args))
        else:
            info.append('No Splitting')

        if self.neg_sample_args is not None and self.neg_sample_args['strategy'] != 'none':
            info.append('Negative Sampling: {}'.format(self.neg_sample_args))
        else:
            info.append('No Negative Sampling')

        return '\n\t'.join(info)

    def __repr__(self):
        return self.__str__()

    def group_by(self, field=None):
        """Setting about group

        Args:
            field (str): The field of dataset grouped by, default None (Not Grouping)

        Example:
            >>> es.group_by('month')
            >>> es.group_by_user()
        """
        self.group_field = field

    def group_by_user(self):
        """Group by user

        Note:
            Requires ``USER_ID_FIELD`` in config
        """
        self.group_field = self.config['USER_ID_FIELD']

    def set_ordering(self, strategy='none', **kwargs):
        """Setting about ordering

        Args:
            strategy (str): Either ``none``, ``shuffle`` or ``by``
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
        legal_strategy = {'none', 'shuffle', 'by'}
        if strategy not in legal_strategy:
            raise ValueError('Ordering Strategy [{}] should in {}'.format(strategy, list(legal_strategy)))
        self.ordering_args = {'strategy': strategy}
        self.ordering_args.update(kwargs)

    def random_ordering(self):
        """Shuffle Setting
        """
        self.set_ordering('shuffle')

    def sort_by(self, field, ascending=None):
        """Setting about Sorting.

        Similar with pandas' sort_values_

        .. _sort_values: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.sort_values.html?highlight=sort_values#pandas.DataFrame.sort_values

        Args:
            field (str or list of str): Name or list of names
            ascending (bool or list of bool): Sort ascending vs. descending. Specify list for multiple sort orders.
                If this is a list of bools, must match the length of the field
        """
        if not isinstance(field, list):
            field = [field]
        if ascending is None:
            ascending = [True] * len(field)
            if len(ascending) == 1:
                ascending = True
        self.set_ordering('by', field=field, ascending=ascending)

    def temporal_ordering(self):
        """Setting about Sorting by timestamp.

        Note:
            Requires `TIME_FIELD` in config
        """
        self.sort_by(field=self.config['TIME_FIELD'])

    def set_splitting(self, strategy='none', **kwargs):
        """Setting about split method

        Args:
            strategy (str): Either ``none``, ``by_ratio``, ``by_value`` or ``loo``.
            ratios (list of float): Dataset will be splited into `len(ratios)` parts.
            field (str): Split by values of field.
            values (list of float or float): Dataset will be splited into `len(values) + 1` parts.
                The first part will be interactions whose field value in (\\*, values[0]].
            ascending (bool): Order of values after splitting.

        Example:
            >>> es.leave_one_out()
            >>> es.split_by_ratio(ratios=[0.8, 0.1, 0.1])
            >>> es.split_by_value(field='month', values=[6, 7], ascending=False)    # (*, 7], (7, 6], (6, *)
        """
        legal_strategy = {'none', 'by_ratio', 'by_value', 'loo'}
        if strategy not in legal_strategy:
            raise ValueError('Split Strategy [{}] should in {}'.format(strategy, list(legal_strategy)))
        if strategy == 'loo' and self.group_by is None:
            raise ValueError('Leave-One-Out request group firstly')
        self.split_args = {'strategy': strategy}
        self.split_args.update(kwargs)

    def leave_one_out(self, leave_one_num=1):
        """ Setting about Splitting by 'leave-one-out' strategy.

        Note:
            Requires setting group by.

        Args:
            leave_one_num (int): number of sub datasets for evaluation.
                E.g. ``leave_one_num = 2`` if you have one validation dataset and one test dataset.
        """
        if self.group_field is None:
            raise ValueError('Leave one out request grouped dataset, please set group field.')
        self.set_splitting(strategy='loo', leave_one_num=leave_one_num)

    def split_by_ratio(self, ratios):
        """ Setting about Ratio-based Splitting.

        Args:
            ratios (list of float): ratio of each part.
                No need to normalize. It's ok with either `[0.8, 0.1, 0.1]`, `[8, 1, 1]` or `[56, 7, 7]`
        """
        if not isinstance(ratios, list):
            raise ValueError('ratios [{}] should be list'.format(ratios))
        self.set_splitting(strategy='by_ratio', ratios=ratios)

    def _split_by_value(self, field, values, ascending=True):
        raise NotImplementedError('Split by value has not been implemented.')
        if not isinstance(field, str):
            raise ValueError('field [{}] should be str'.format(field))
        if not isinstance(values, list):
            values = [values]
        values.sort(reverse=(not ascending))
        self.set_splitting(strategy='by_value', values=values, ascending=ascending)

    def set_neg_sampling(self, strategy='none', distribution='uniform', **kwargs):
        """Setting about negative sampling

        Args:
            strategy (str): Either ``none``, ``full`` or ``by``.
            by (int): Negative Sampling `by` neg cases for one pos case.
            distribution (str): distribution of sampler, either 'uniform' or 'popularity'.

        Example:
            >>> es.neg_sample_to(100)
            >>> es.neg_sample_by(1)
        """
        legal_strategy = {'none', 'full', 'by'}
        if strategy not in legal_strategy:
            raise ValueError('Negative Sampling Strategy [{}] should in {}'.format(strategy, list(legal_strategy)))
        if strategy == 'full' and distribution != 'uniform':
            raise ValueError('Full Sort can not be sampled by distribution [{}]'.format(distribution))
        self.neg_sample_args = {'strategy': strategy, 'distribution': distribution}
        self.neg_sample_args.update(kwargs)

    def neg_sample_by(self, by, distribution='uniform'):
        """Setting about negative sampling by, which means sample several negative records for each positive records.

        Args:
            by (int): The number of neg cases for one pos case.
            distribution (str): distribution of sampler, either ``uniform`` or ``popularity``.
        """
        self.set_neg_sampling(strategy='by', by=by, distribution=distribution)

    def RO_RS(self, ratios=(0.8, 0.1, 0.1), group_by_user=True):
        """Preset about Random Ordering and Ratio-based Splitting.

        Args:
            ratios (list of float): ratio of each part.
                No need to normalize. It's ok with either ``[0.8, 0.1, 0.1]``, ``[8, 1, 1]`` or ``[56, 7, 7]``
            group_by_user (bool): set group field to user_id if True
        """
        if group_by_user:
            self.group_by_user()
        self.random_ordering()
        self.split_by_ratio(ratios)

    def TO_RS(self, ratios=(0.8, 0.1, 0.1), group_by_user=True):
        """Preset about Temporal Ordering and Ratio-based Splitting.

        Args:
            ratios (list of float): ratio of each part.
                No need to normalize. It's ok with either ``[0.8, 0.1, 0.1]``, ``[8, 1, 1]`` or ``[56, 7, 7]``
            group_by_user (bool): set group field to user_id if True
        """
        if group_by_user:
            self.group_by_user()
        self.temporal_ordering()
        self.split_by_ratio(ratios)

    def RO_LS(self, leave_one_num=1, group_by_user=True):
        """Preset about Random Ordering and Leave-one-out Splitting.

        Args:
            leave_one_num (int): number of sub datasets for evaluation.
                E.g. ``leave_one_num=2`` if you have one validation dataset and one test dataset.
            group_by_user (bool): set group field to user_id if True
        """
        if group_by_user:
            self.group_by_user()
        self.random_ordering()
        self.leave_one_out(leave_one_num=leave_one_num)

    def TO_LS(self, leave_one_num=1, group_by_user=True):
        """Preset about Temporal Ordering and Leave-one-out Splitting.

        Args:
            leave_one_num (int): number of sub datasets for evaluation.
                E.g. ``leave_one_num=2`` if you have one validation dataset and one test dataset.
            group_by_user (bool): set group field to user_id if True
        """
        if group_by_user:
            self.group_by_user()
        self.temporal_ordering()
        self.leave_one_out(leave_one_num=leave_one_num)

    def uni100(self):
        """Preset about uniform sampling 100 items for each positive records while negative sampling.
        """
        self.neg_sample_by(100)

    def pop100(self):
        """Preset about popularity-based sampling 100 items for each positive records while negative sampling.
        """
        self.neg_sample_by(100, distribution='popularity')

    def uni1000(self):
        """Preset about uniform sampling 1000 items for each positive records while negative sampling.
        """
        self.neg_sample_by(1000)

    def pop1000(self):
        """Preset about popularity-based sampling 1000 items for each positive records while negative sampling.
        """
        self.neg_sample_by(1000, distribution='popularity')

    def full(self):
        """Preset about adopt the entire item set (excluding ground-truth items) for ranking.
        """
        self.set_neg_sampling(strategy='full')
