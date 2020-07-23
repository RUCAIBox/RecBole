import abc
import numpy as np
import pandas as pd
import utils
import warnings
# from joblib import Parallel, delayed
# from pandarallel import pandarallel
# pandarallel.initialize()

# 'Precision', 'Hit', 'Recall', 'MAP', 'NDCG', 'MRR', 'AUC'
metric_name = {metric.lower() : metric for metric in ['Hit', 'Recall', 'MRR', 'AUC', 'Precision', 'NDCG']}

# These metrics are typical in topk recommendations
topk_metric = {'hit', 'recall', 'precision', 'ndcg', 'mrr'}
other_metric = {'auc'}

class AbstractEvaluator(metaclass=abc.ABCMeta):
    """The abstract class of the evaluation module, its subclasses must implement their functions

    """    
    def __init__(self):

        pass
    
    @abc.abstractmethod
    def recommend(self):
        """Recommend items for users

        """        
        raise NotImplementedError

    @abc.abstractmethod
    def evaluate(self):
        """Evaluate the dataset

        """
        raise NotImplementedError

class RankEvaluator(AbstractEvaluator):
    """Top k evaluator

    """    
    def __init__(self, config, logger, metrics, topk):
        self.logger = logger
        self.metrics = metrics
        self.topk = topk

        self.USER_FIELD = config['USER_ID_FIELD']
        self.ITEM_FIELD = config['ITEM_ID_FIELD']
        self.LABEL_FIELD = config['LABEL_FIELD']

    def recommend(self, df, k):
        """Recommend the top k items for users

        Args:
            df (pandas.core.frame.DataFrame): merged data which contains user id, item id, score, rank
            k ([type]): topk

        Returns:
            (list, list): (user_list, item_list)
        """

        df['rank'] = df.groupby(self.USER_FIELD)['score'].rank(method='first', ascending=False)
        mask = (df['rank'].values > 0) & (df['rank'].values <= k)
        topk_df = df[mask]
        return topk_df[self.USER_FIELD].values.tolist(), topk_df[self.ITEM_FIELD].values.tolist()

    def metric_info(self, df, metric, k):
        """Get the result of the metric on the data

        Args:
            df (pandas.core.frame.DataFrame): merged data which contains user id, item id, score, rank
            metric (str): one of the metrics
            k (int): top k

        Returns:
            float: metric result
        """

        fuc = getattr(utils, metric)
        metric_fuc = lambda x: fuc(x['rank'].values, x[self.LABEL_FIELD].values.astype(bool), k)

        # groups = df.groupby(self.USER_FIELD)['rank']
        # results = Parallel(n_jobs=10)(delayed(metric_fuc)(group) for _, group in groups)
        # result = np.mean(results)

        metric_result = df.groupby(self.USER_FIELD)[['rank', self.LABEL_FIELD]].apply(metric_fuc)
        return metric_result

    def evaluate(self, df):
        """Generate metrics results on the dataset

        Args:
            df (pandas.core.frame.DataFrame): merged data which contains user id, item id, score, rank

        Returns:
            dict: such as { 'Hit@5': 0.6666666666666666, 'MRR@5': 0.23796296296296293, 'Recall@5': 0.5277777777777778, 
                            'Hit@3': 0.6666666666666666, 'MRR@3': 0.22685185185185186, 'Recall@3': 0.47222222222222215, 
                            'Hit@1': 0.16666666666666666, 'MRR@1': 0.08333333333333333, 'Recall@1': 0.08333333333333333 }
        """

        metric_dict = {}
        num_users = df[self.USER_FIELD].nunique()
        for metric in self.metrics:
            if metric in topk_metric:
                for k in self.topk:
                    metric_result = self.metric_info(df, metric, k)
                    key = '{}@{}'.format(metric_name[metric], k)
                    score = metric_result.sum() / num_users
                    metric_dict[key] = score
            else:
                key = metric_name[metric]
                metric_result = self.metric_info(df, metric, None)
                score = metric_result.sum() / num_users
                metric_dict[key] = score

        return metric_dict

class LossEvaluator(AbstractEvaluator):
    """Loss evaluator

    """

    def __init__(self, config, logger, metrics):
        self.logger = logger
        self.metrics = metrics 
        self.cut_method = ['ceil', 'floor', 'around']

        self.LABEL_FIELD = config['LABEL_FIELD']
        self.USER_FIELD = config['USER_ID_FIELD']
        self.ITEM_FIELD = config['ITEM_ID_FIELD']
        self.metric_cols = ['score', self.LABEL_FIELD]

    def cutoff(self, df, col, method):
        """Cut off the col's values by using the method

        Args:
            df (pandas.core.frame.DataFrame): merged data which contains user id, item id, score, rank
            col (str): the specified column
            method (str): one of ['ceil', 'floor', 'around']

        Raises:
            AttributeError: method error

        Returns:
            (pandas.core.frame.DataFrame): the processed dataframe
        """   

        try:
            cut_method = getattr(np, method)
        except AttributeError as e:
            raise AttributeError("module 'numpy' has no attribute '{}'".format(method))
        df[col] = df[col].apply(cut_method)
        return df

    def recommend(self, df, k):

        raise NotImplementedError

    def metric_info(self, df, metric):
        """Get the result of the metric on the data

        Args:
            df (pandas.core.frame.DataFrame): merged data which contains user id, item id, score, rank
            metric (str): one of the metrics

        Returns:
            float: metric result
        """

        metric_fuc = getattr(utils, metric)
        metric_result = df[self.metric_cols].apply(metric_fuc)
        return metric_result

    def evaluate(self, df):
        """Generate metrics results on the dataset

        Args:
            df (pandas.core.frame.DataFrame): merged data which contains user id, item id, score, rank

        Returns:
            dict: such as { 'Hit@5': 0.6666666666666666, 'MRR@5': 0.23796296296296293, 'Recall@5': 0.5277777777777778, 
                            'Hit@3': 0.6666666666666666, 'MRR@3': 0.22685185185185186, 'Recall@3': 0.47222222222222215, 
                            'Hit@1': 0.16666666666666666, 'MRR@1': 0.08333333333333333, 'Recall@1': 0.08333333333333333 }
        """
        metric_dict = {}
        for metric in self.metrics:
            score = self.metric_info(df, metric)
            key, value = metric_name[metric], score
            metric_dict[key] = value
        return metric_dict
            
class Evaluator(AbstractEvaluator):

    def __init__(self, config, logger):

        super(Evaluator, self).__init__()

        self.group_view = config['group_view']
        self.eval_metric = config['eval_metric']
        self.topk = config['topk']
        self.logger = logger
        self.verbose = True

        self.USER_FIELD = config['USER_ID_FIELD']
        self.ITEM_FIELD = config['ITEM_ID_FIELD']
        self.LABEL_FIELD = config['LABEL_FIELD']

        self._check_args()

        if (topk_metric | other_metric) & set(self.eval_metric):
            self.evaluator = RankEvaluator(config, self.logger, self.eval_metric, self.topk)
        else:
            self.evaluator = LossEvaluator(config, self.logger, self.eval_metric)

    def collect(self, result_list, batch_size_list):
        """when using minibatch in training phase, you need to call this function to summarize the results

        Args:
            result_list (list): a list of metric dict
            batch_size_list (list): a list of integers

        Returns:
            dict: such as { 'Hit@5': 0.6666666666666666, 'MRR@5': 0.23796296296296293, 'Recall@5': 0.5277777777777778, 
                            'Hit@3': 0.6666666666666666, 'MRR@3': 0.22685185185185186, 'Recall@3': 0.47222222222222215, 
                            'Hit@1': 0.16666666666666666, 'MRR@1': 0.08333333333333333, 'Recall@1': 0.08333333333333333 }
        """    

        tmp_result_list = []
        keys = list(result_list[0].keys())
        for result in result_list:
            tmp_result_list.append(list(result.values()))

        result_matrix = np.array(tmp_result_list)
        batch_size_matrix = np.array(batch_size_list).reshape(-1, 1)
        assert result_matrix.shape[0] == batch_size_matrix.shape[0]

        weighted_matrix = result_matrix * batch_size_matrix
        
        metric_list =  (np.sum(weighted_matrix, axis=0) / np.sum(batch_size_matrix)).tolist()
        metric_dict = {}
        for method, score in zip(keys, metric_list):
            metric_dict[method] = score
        return metric_dict

    def _check_args(self):

        # check group_view
        if isinstance(self.group_view, (int, list, None.__class__)):
            if isinstance(self.group_view, int):
                assert self.group_view > 0, 'group_view must be a pistive integer or a list of postive integers'
                self.group_view = [self.group_view]
        else:
            raise TypeError('The group_view muse be int or list')
        if self.group_view is not None:
            for number in self.group_view:
                assert isinstance(number, int) and number > 0, '{} in group_view is not a postive integer'.format(number)
            self.group_view = sorted(self.group_view)

        # check eval_metric
        if isinstance(self.eval_metric, (str, list)):
            if isinstance(self.eval_metric, str):
                self.eval_metric = [self.eval_metric]
        else:
            raise TypeError('eval_metric must be str or list')
        
        for m in self.eval_metric:
            if m.lower() not in metric_name:
                raise ValueError("There is not the metric named {}!".format(m))
        self.eval_metric = [metric.lower() for metric in self.eval_metric]

        # check topk:
        if isinstance(self.topk, (int, list, None.__class__)):
            if isinstance(self.topk, int):
                assert self.topk > 0, 'topk must be a pistive integer or a list of postive integers'
                self.topk = list(range(1, self.topk + 1))
            elif self.topk is None:
                for metric in self.eval_metric:
                    if metric in topk_metric:
                        warn_str = 'The {} is not calculated in topk evaluator. please confirm your purpose of using this metric.'.format(metric)
                        self.logger.warning(warn_str)

        else:
            raise TypeError('The topk muse be int or list')
        if self.topk is not None:
            for number in self.topk:
                assert isinstance(number, int) and number > 0, '{} in topk is not a posttive integer'.format(number) 
            self.topk = sorted(self.topk)[::-1]
            for metric in self.eval_metric:
                if metric in other_metric:
                    warn_str = 'The {} is calculated in topk evaluator. please confirm your purpose of using this metric.'.format(metric)
                    self.logger.warning(warn_str)

    def get_grouped_data(self, df):
        """ a interface which can split the users into groups.

        Args:
            df (pandas.core.frame.DataFrame): merged data which contains user id, item id, score, rank

        Returns:
            (zip): a zip object which consists of (name, (idx, group))
        """

        group_view = [0] + self.group_view + ['-']
        group_names = []
        for begin, end in zip(group_view[:-1], group_view[1:]):
            group_names.append('({},{}]'.format(begin, end))
        group_data = df.groupby('group_id', sort=True)
        return zip(group_names, group_data)

    def common_evaluate(self, df):
        """evaluate the dataset by no grouping

        Args:
            df (pandas.core.frame.DataFrame): merged data which contains user id, item id, score, rank

        Returns:
            dict: such as { 'Hit@5': 0.6666666666666666, 'MRR@5': 0.23796296296296293, 'Recall@5': 0.5277777777777778, 
                            'Hit@3': 0.6666666666666666, 'MRR@3': 0.22685185185185186, 'Recall@3': 0.47222222222222215, 
                            'Hit@1': 0.16666666666666666, 'MRR@1': 0.08333333333333333, 'Recall@1': 0.08333333333333333 }
        """        
        return self.evaluator.evaluate(df)

    def group_evaluate(self, df):
        """evaluate the dataset by grouping

        Args:
            df (pandas.core.frame.DataFrame): merged data which contains user id, item id, score, rank

        Returns:
            {'(0,1]': 
                    {'Hit@5': 0.3333333333333333, 'MRR@5': 0.05925925925925926, 'Recall@5': 0.2222222222222222,
                    'Hit@3': 0.3333333333333333, 'MRR@3': 0.037037037037037035, 'Recall@3': 0.1111111111111111,
                    'Hit@1': 0.0, 'MRR@1': 0.0, 'Recall@1': 0.0},
            '(1,-]':
                    {'Hit@5': 1.0, 'MRR@5': 0.4166666666666667, 'Recall@5': 0.8333333333333334,
                    'Hit@3': 1.0, 'MRR@3': 0.4166666666666667, 'Recall@3': 0.8333333333333334,
                    'Hit@1': 0.3333333333333333, 'MRR@1': 0.16666666666666666, 'Recall@1': 0.16666666666666666}}
        """ 

        result_dict = {}
        for name, (_, group) in self.get_grouped_data(df):
            result_dict[name] = self.common_evaluate(group.copy())
        return result_dict

    def _print(self, message):
        if self.verbose:
            self.logger.info(message)

    def build_evaluate_df(self, rdata, result):

        df = pd.DataFrame({
            self.USER_FIELD: rdata[self.USER_FIELD],
            self.ITEM_FIELD: rdata[self.ITEM_FIELD],
            'score': result,
            self.LABEL_FIELD: rdata[self.LABEL_FIELD]
        })
        return df

    def build_recommend_df(self, rdata, result):
        df = pd.DataFrame({
            self.USER_FIELD: rdata[self.USER_FIELD],
            self.ITEM_FIELD: rdata[self.ITEM_FIELD],
            'score': result
        })
        return df
        
    def recommend(self, rdata, result, k):
        """Recommend the top k items for users

        Args:
            df (pandas.core.frame.DataFrame): merged data which contains user id, item id, score, rank
            k ([type]): topk

        Returns:
            (list, list): (user_list, item_list)
        """

        df = self.build_recommend_df(rdata, result)
        return self.evaluator.recommend(df, k)

    def evaluate(self, result, rdata):
        """Generate metrics results on the dataset

        Args:
            df (pandas.core.frame.DataFrame): merged data which contains user id, item id, score, rank

        Returns:
            dict: a dict
        """

        df = self.build_evaluate_df(rdata, result)
        if (topk_metric | other_metric) & set(self.eval_metric):
            df['rank'] = df.groupby(self.USER_FIELD)['score'].rank(method='first', ascending=False)
        if self.group_view is not None:
            return self.group_evaluate(df)
        return self.common_evaluate(df)

    def __str__(self):
        return 'The evaluator will evaluate test_data on {} at {}'.format(', '.join(self.eval_metric), ', '.join(map(str, self.topk)))

    def __repr__(self):
        return self.__str__()

    def __enter__(self):
        self.logger.info('Evaluate Start...')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.info('Evaluate End...')

