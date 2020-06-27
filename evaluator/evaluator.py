import pandas as pd
import utils

metric_name = {metric.lower() : metric for metric in ['Precision', 'Hit', 'Recall', 'MAP', 'NDCG', 'MRR']}

class BaseEvaluator(object):

    def __init__(self):
        pass

    def evaluate(self, result, test_data):
        """Evaluate the model and get the results of the model on the specified data

        Args:
            result : TODO 
            test_data : TODO

        Returns:
            A string which consists of all information about the result on the test_data 
        """        
        return NotImplementedError

class _UnionEvaluator(BaseEvaluator):

    def __init__(self, eval_metric, topk, workers):
        super(_UnionEvaluator, self).__init__()
        self.topk = topk
        self.eval_metric = eval_metric
        self.workers = workers

    def get_ground_truth(self, users, test_data):
        # TODO 对接
        users, items = test_data
        return users, items
    
    def get_result_pairs(self, result):
        # TODO 对接
        users, items, scores = result
        return users, items, scores

    # @profile
    def evaluate(self, result, test_data):
        users, items, scores = self.get_result_pairs(result)
        result_df = pd.DataFrame({'user_id':users, 'item_id':items, 'score':scores})
        result_df['rank'] = result_df.groupby(['user_id'])['score'].rank(method='first', ascending=False)

        users, items = self.get_ground_truth(users, test_data)
        truth_df = pd.DataFrame({'user_id':users, 'item_id':items})

        truth_df['count'] = truth_df.groupby('user_id')['item_id'].transform('count')

        eval_df = truth_df.merge(result_df, on=['user_id', 'item_id'], how='left')
        eval_df['rank'].fillna(-1, inplace=True)

        print(eval_df)
        metric_info = []
        for k in self.topk:
            for method in self.eval_metric:
                eval_fuc = getattr(utils, method)
                topk_df = eval_df[eval_df['rank'] <= k]
                score = eval_fuc(topk_df)
                metric_info.append('{:>5}@{} : {:5f}'.format(metric_name[method], k, score))
        return '\t'.join(metric_info)


class _GroupEvaluator(_UnionEvaluator):

    def __init__(self, group_view, eval_metric, topk, workers):

        super(_GroupEvaluator, self).__init__(eval_metric, topk, workers)
        self.group_view = group_view
        self.groups = self.get_groups()

    def split_data(self, result, test_data):
        group_list = []
        return group_list

    def get_groups(self):
        group_view = [0] + self.group_view + ['-']
        groups = []
        for begin, end in zip(group_view[:-1], group_view[1:]):
            groups.append('({},{}]'.format(begin, end))
        return groups

    def evaluate_groups(self, result, test_data):
        group_list = self.split_data(result, test_data)
        info_list = []
        for index, group in enumerate(group_list):
            info_str = self.evaluate(group, test_data)
            info_list.append(self.groups[index] + info_str)
        
class Evaluator(BaseEvaluator):

    def __init__(self, config):
        super(Evaluator, self).__init__()

        self.group_view = config['group_view']
        self.eval_metric = config['metric']
        self.topk = config['topk']
        self.workers = config['workers']  # TODO 多进程，但是windows可能有点难搞, 貌似要在__main__里
 
        # XXX 这种类型检查应该放到哪呢?放在config部分一次判断，还是分散在各模块中呢？
        self._check_args()

        if self.group_view is not None:
            self.evaluator = _GroupEvaluator(self.group_view, self.eval_metric, self.topk, self.workers)
        else:
            self.evaluator = _UnionEvaluator(self.eval_metric, self.topk, self.workers)

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
                assert isinstance(number, int) and number > 0, 'The number {} of group_view is not a number of postive integer'.format(number)
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
        if isinstance(self.topk, (int, list)):
            if isinstance(self.topk, int):
                assert self.topk > 0, 'topk must be a pistive integer or a list of postive integers'
                self.topk = list(range(1, self.topk + 1))
        else:
            raise TypeError('The topk muse be int or list')
        for number in self.topk:
            assert isinstance(number, int) and number > 0, 'The number {} of topk is not a number of posttive integer'.format(number) 
        self.topk = sorted(self.topk)

    def evaluate(self, result, test_data):
        info_str = self.evaluator.evaluate(result, test_data)
        print(info_str)

    def __str__(self):
        return 'The evaluator will evaluate test_data on metirc {} at {}'.format(', '.join(self.eval_metric), ', '.join(map(str, self.topk)))

    def __repr__(self):
        return self.__str__()

    def __enter__(self):
        print('Evaluate Start...')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print('Evaluate End...')