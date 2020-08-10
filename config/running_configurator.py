from config.abstract_configurator import AbstractConfig
import os
from evaluator import loss_metrics, topk_metrics


class RunningConfig(AbstractConfig):
    def __init__(self, config_file_name, cmd_args):

        super().__init__()
        self.must_args = ['model', 'dataset']
        self.args = self._read_config_file(config_file_name, 'default')

        self._check_args()
        self._replace_args(cmd_args)
        if 'data_path' not in self:
            data_path = os.path.join('dataset', self['dataset'])
            self['data_path'] = data_path

        eval_type = None
        for metric in self['metrics']:
            if metric.lower() in loss_metrics:
                if eval_type is not None and eval_type == 'topk':
                    raise RuntimeError('Ranking metrics and other metrics can not be used at the same time!')
                else:
                    eval_type = 'loss'
            if metric.lower() in topk_metrics:
                if eval_type is not None and eval_type == 'loss':
                    raise RuntimeError('Ranking metrics and other metrics can not be used at the same time!')
                else:
                    eval_type = 'topk'
        self['eval_type'] = eval_type

