from config.abstract_configurator import AbstractConfig
import os
from evaluator import loss_metrics, topk_metrics
from utils import EvaluatorType


class RunningConfig(AbstractConfig):
    def __init__(self, config_file_name):

        super().__init__()
        self.must_args = ['model', 'dataset']
        self.args = self._read_config_file(config_file_name, 'default')

        self._check_args()
        #self._replace_args(cmd_args)
        if 'gpu_id' in self:
            self['use_gpu'] = True
        else:
            self['use_gpu'] = False

        if 'data_path' not in self:
            data_path = os.path.join('dataset', self['dataset'])
            self['data_path'] = data_path

        if 'checkpoint_dir' not in self:
            self['checkpoint_dir'] = 'saved'

        eval_type = None
        for metric in self['metrics']:
            if metric.lower() in loss_metrics:
                if eval_type is not None and eval_type == EvaluatorType.RANKING:
                    raise RuntimeError('Ranking metrics and other metrics can not be used at the same time!')
                else:
                    eval_type = EvaluatorType.INDIVIDUAL
            if metric.lower() in topk_metrics:
                if eval_type is not None and eval_type == EvaluatorType.INDIVIDUAL:
                    raise RuntimeError('Ranking metrics and other metrics can not be used at the same time!')
                else:
                    eval_type = EvaluatorType.RANKING
        self['eval_type'] = eval_type

        smaller_metric = ['rmse','mae', 'logloss']

        if 'valid_metric' not in self:
            valid_metric = self['metric'][0]
            if 'topk' in self:
                valid_metric += '@' + self['topk'][0]
            self['valid_metric'] = valid_metric

        if 'valid_metric_bigger' not in self:
            valid_metric = self['valid_metric'].split('@')[0]
            if valid_metric in smaller_metric:
                self['valid_metric_bigger'] = False
            else:
                self['valid_metric_bigger'] = True




