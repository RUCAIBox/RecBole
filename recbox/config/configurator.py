# @Time   : 2020/6/28
# @Author : Zihan Lin
# @Email  : linzihan.super@foxmail.com

# UPDATE
# @Time   : 2020/10/04, 2020/10/9
# @Author : Shanlei Mu, Yupeng Hou
# @Email  : slmu@ruc.edu.cn, houyupeng@ruc.edu.cn

"""
recbox.config.configurator
################################
"""

import re
import os
import sys
import yaml
import torch
from logging import getLogger

from recbox.evaluator import loss_metrics, topk_metrics
from recbox.utils import get_model, Enum, EvaluatorType


class Config(object):
    """ Configurator module that load the defined parameters.

    Configurator module will first load the default parameters from the fixed properties in UniRec and then
    load parameters from the external input.

    External input supports three kind of forms: config file, command line and parameter dictionaries.

    - config file: It's a file that record the parameters to be modified or added. It should be in ``yaml`` format,
      e.g. a config file is 'example.yaml', the content is:
                learning_rate: 0.001

                train_batch_size: 2048

    - command line: It should be in the format as '---learning_rate=0.001'

    - parameter dictionaries: It should be a dict, where the key is parameter name and the value is parameter value,
      e.g. config_dict = {'learning_rate': 0.001}

    Configuration module allows the above three kind of external input format to be used together,
    the priority order is as following:

    command line > parameter dictionaries > config file

    e.g. If we set learning_rate=0.01 in config file, learning_rate=0.02 in command line,
    learning_rate=0.03 in parameter dictionaries.

    Finally the learning_rate is equal to 0.02.
    """

    def __init__(self, model=None, dataset=None, config_file_list=None, config_dict=None):
        """
        Args:
            model (str): the model name, default is None, if it is None, config will search the parameter 'model'
            from the external input as the model name.
            dataset (str): the dataset name, default is None, if it is None, config will search the parameter 'dataset'
            from the external input as the dataset name.
            config_file_list (list of str): the external config file, it allows multiple config files, default is None.
            config_dict (dict): the external parameter dictionaries, default is None.
        """
        self._init_parameters_category()
        self.yaml_loader = self._build_yaml_loader()
        self._load_config_files(config_file_list)
        self._load_variable_config_dict(config_dict)
        self._load_cmd_line()
        self._merge_external_config_dict()
        self.model, self.dataset = self._get_model_and_dataset(model, dataset)
        self._load_internal_config_dict(self.model, self.dataset)
        self.final_config_dict = self._get_final_config_dict()
        self._set_default_parameters()
        self._init_device()

    def _init_parameters_category(self):
        self.parameters = dict()
        self.parameters['General'] = ['gpu_id', 'use_gpu', 'seed', 'data_path']
        self.parameters['Training'] = ['epochs', 'train_batch_size', 'learner', 'learning_rate',
                                       'training_neg_sample_num', 'eval_step', 'valid_metric',
                                       'stopping_step', 'checkpoint_dir']
        self.parameters['Evaluation'] = ['eval_setting', 'group_by_user', 'split_ratio', 'leave_one_num',
                                         'real_time_process', 'metrics', 'topk', 'eval_batch_size']
        self.parameters['Dataset'] = []

    def _build_yaml_loader(self):
        loader = yaml.FullLoader
        loader.add_implicit_resolver(
            u'tag:yaml.org,2002:float',
            re.compile(u'''^(?:
             [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
            |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
            |\\.[0-9_]+(?:[eE][-+][0-9]+)?
            |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
            |[-+]?\\.(?:inf|Inf|INF)
            |\\.(?:nan|NaN|NAN))$''', re.X),
            list(u'-+0123456789.'))
        return loader

    def _load_config_files(self, file_list):
        self.file_config_dict = dict()
        if file_list:
            for file in file_list:
                if os.path.isfile(file):
                    with open(file, 'r', encoding='utf-8') as f:
                        self.file_config_dict.update(yaml.load(f.read(), Loader=self.yaml_loader))

    def _load_variable_config_dict(self, config_dict):
        self.variable_config_dict = config_dict if config_dict else dict()

    def _load_cmd_line(self):
        r""" Read parameters from command line and convert it to str.

        """

        def convert_cmd_args():
            r"""This function convert the str parameters to their original type.

            """
            for key in self.cmd_config_dict:
                param = self.cmd_config_dict[key]
                if not isinstance(param, str):
                    continue
                try:
                    value = eval(param)
                    if not isinstance(value, (str, int, float, list, tuple, dict, bool, Enum)):
                        value = param
                except (NameError, SyntaxError, TypeError):
                    if isinstance(param, str):
                        if param.lower() == "true":
                            value = True
                        elif param.lower() == "false":
                            value = False
                        else:
                            value = param
                    else:
                        value = param
                self.cmd_config_dict[key] = value

        self.cmd_config_dict = dict()
        unrecognized_args = []
        if "ipykernel_launcher" not in sys.argv[0]:
            for arg in sys.argv[1:]:
                if not arg.startswith("--") or len(arg[2:].split("=")) != 2:
                    unrecognized_args.append(arg)
                    continue
                cmd_arg_name, cmd_arg_value = arg[2:].split("=")
                if cmd_arg_name in self.cmd_config_dict and cmd_arg_value != self.cmd_config_dict[cmd_arg_name]:
                    raise SyntaxError("There are duplicate commend arg '%s' with different value!" % arg)
                else:
                    self.cmd_config_dict[cmd_arg_name] = cmd_arg_value
        if len(unrecognized_args) > 0:
            logger = getLogger()
            logger.warning('command line args [{}] will not be used in RecBox'.format(' '.join(unrecognized_args)))

        convert_cmd_args()

    def _merge_external_config_dict(self):
        external_config_dict = dict()
        external_config_dict.update(self.file_config_dict)
        external_config_dict.update(self.variable_config_dict)
        external_config_dict.update(self.cmd_config_dict)
        self.external_config_dict = external_config_dict

    def _get_model_and_dataset(self, model, dataset):
        if model is None:
            try:
                final_model = self.external_config_dict['model']
            except KeyError:
                raise KeyError('model need to be specified in at least one of the these ways: '
                               '[model variable, config file, config dict, command line] ')
        else:
            final_model = model

        if dataset is None:
            try:
                final_dataset = self.external_config_dict['dataset']
            except KeyError:
                raise KeyError('dataset need to be specified in at least one of the these ways: '
                               '[dataset variable, config file, config dict, command line] ')
        else:
            final_dataset = dataset

        return final_model, final_dataset

    def _load_internal_config_dict(self, model, dataset):
        current_path = os.path.dirname(os.path.realpath(__file__))
        overall_init_file = os.path.join(current_path, '../properties/overall.yaml')
        model_init_file = os.path.join(current_path, '../properties/model/' + model + '.yaml')
        dataset_init_file = os.path.join(current_path, '../properties/dataset/' + dataset + '.yaml')

        self.internal_config_dict = dict()
        for file in [overall_init_file, model_init_file, dataset_init_file]:
            if os.path.isfile(file):
                with open(file, 'r', encoding='utf-8') as f:
                    config_dict = yaml.load(f.read(), Loader=self.yaml_loader)
                    if file == dataset_init_file:
                        self.parameters['Dataset'] += [key for key in config_dict.keys() if
                                                       key not in self.parameters['Dataset']]
                    if config_dict is not None:
                        self.internal_config_dict.update(config_dict)

    def _get_final_config_dict(self):
        final_config_dict = dict()
        final_config_dict.update(self.internal_config_dict)
        final_config_dict.update(self.external_config_dict)
        return final_config_dict

    def _set_default_parameters(self):

        self.final_config_dict['dataset'] = self.dataset
        self.final_config_dict['model'] = self.model
        if self.dataset == 'ml-100k':
            current_path = os.path.dirname(os.path.realpath(__file__))
            self.final_config_dict['data_path'] = os.path.join(current_path, '../dataset_example/' + self.dataset)
        else:
            self.final_config_dict['data_path'] = os.path.join(self.final_config_dict['data_path'], self.dataset)

        eval_type = None
        for metric in self.final_config_dict['metrics']:
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
        self.final_config_dict['eval_type'] = eval_type

        smaller_metric = ['rmse', 'mae', 'logloss']
        valid_metric = self.final_config_dict['valid_metric'].split('@')[0]
        self.final_config_dict['valid_metric_bigger'] = False if valid_metric in smaller_metric else True

        model = get_model(self.model)
        self.final_config_dict['MODEL_TYPE'] = model.type
        self.final_config_dict['MODEL_INPUT_TYPE'] = model.input_type

        ad_suf = self.final_config_dict['additional_feat_suffix']
        if ad_suf is not None and isinstance(ad_suf, str):
            self.final_config_dict['additional_feat_suffix'] = [ad_suf]

    def _init_device(self):
        use_gpu = self.final_config_dict['use_gpu']
        if use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.final_config_dict['gpu_id'])
        self.final_config_dict['device'] = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")

    def __setitem__(self, key, value):
        if not isinstance(key, str):
            raise TypeError("index must be a str")
        self.final_config_dict[key] = value

    def __getitem__(self, item):
        if item in self.final_config_dict:
            return self.final_config_dict[item]
        else:
            return None

    def __contains__(self, key):
        if not isinstance(key, str):
            raise TypeError("index must be a str!")
        return key in self.final_config_dict

    def __str__(self):
        args_info = ''
        for category in self.parameters:
            args_info += category + ' Hyper Parameters: \n'
            args_info += '\n'.join(
                ["{}={}".format(arg, value) for arg, value in self.final_config_dict.items() if arg in self.parameters[category]])
            args_info += '\n\n'
        return args_info

    def __repr__(self):
        return self.__str__()
