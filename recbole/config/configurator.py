# @Time   : 2020/6/28
# @Author : Zihan Lin
# @Email  : linzihan.super@foxmail.com

# UPDATE
# @Time   : 2020/10/04, 2020/10/9
# @Author : Shanlei Mu, Yupeng Hou
# @Email  : slmu@ruc.edu.cn, houyupeng@ruc.edu.cn

"""
recbole.config.configurator
################################
"""

import re
import os
import sys
import yaml
import torch
from logging import getLogger

from recbole.evaluator import loss_metrics, topk_metrics
from recbole.utils import get_model, Enum, EvaluatorType, ModelType, InputType, \
    general_arguments, training_arguments, evaluation_arguments, dataset_arguments


class Config(object):
    """ Configurator module that load the defined parameters.

    Configurator module will first load the default parameters from the fixed properties in RecBole and then
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
            model (str/AbstractRecommender): the model name or the model class, default is None, if it is None, config
            will search the parameter 'model' from the external input as the model name or model class.
            dataset (str): the dataset name, default is None, if it is None, config will search the parameter 'dataset'
            from the external input as the dataset name.
            config_file_list (list of str): the external config file, it allows multiple config files, default is None.
            config_dict (dict): the external parameter dictionaries, default is None.
        """
        self._init_parameters_category()
        self.yaml_loader = self._build_yaml_loader()
        self.file_config_dict = self._load_config_files(config_file_list)
        self.variable_config_dict = self._load_variable_config_dict(config_dict)
        self.cmd_config_dict = self._load_cmd_line()
        self._merge_external_config_dict()

        self.model, self.model_class, self.dataset = self._get_model_and_dataset(model, dataset)
        self._load_internal_config_dict(self.model, self.model_class, self.dataset)
        self.final_config_dict = self._get_final_config_dict()
        self._set_default_parameters()
        self._init_device()

    def _init_parameters_category(self):
        self.parameters = dict()
        self.parameters['General'] = general_arguments
        self.parameters['Training'] = training_arguments
        self.parameters['Evaluation'] = evaluation_arguments
        self.parameters['Dataset'] = dataset_arguments

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

    def _convert_config_dict(self, config_dict):
        r"""This function convert the str parameters to their original type.

        """
        for key in config_dict:
            param = config_dict[key]
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
            config_dict[key] = value
        return config_dict

    def _load_config_files(self, file_list):
        file_config_dict = dict()
        if file_list:
            for file in file_list:
                with open(file, 'r', encoding='utf-8') as f:
                    file_config_dict.update(yaml.load(f.read(), Loader=self.yaml_loader))
        return file_config_dict

    def _load_variable_config_dict(self, config_dict):
        # HyperTuning may set the parameters such as mlp_hidden_size in NeuMF in the format of ['[]', '[]']
        # then config_dict will receive a str '[]', but indeed it's a list []
        # temporarily use _convert_config_dict to solve this problem
        return self._convert_config_dict(config_dict) if config_dict else dict()

    def _load_cmd_line(self):
        r""" Read parameters from command line and convert it to str.

        """
        cmd_config_dict = dict()
        unrecognized_args = []
        if "ipykernel_launcher" not in sys.argv[0]:
            for arg in sys.argv[1:]:
                if not arg.startswith("--") or len(arg[2:].split("=")) != 2:
                    unrecognized_args.append(arg)
                    continue
                cmd_arg_name, cmd_arg_value = arg[2:].split("=")
                if cmd_arg_name in cmd_config_dict and cmd_arg_value != cmd_config_dict[cmd_arg_name]:
                    raise SyntaxError("There are duplicate commend arg '%s' with different value." % arg)
                else:
                    cmd_config_dict[cmd_arg_name] = cmd_arg_value
        if len(unrecognized_args) > 0:
            logger = getLogger()
            logger.warning('command line args [{}] will not be used in RecBole'.format(' '.join(unrecognized_args)))
        cmd_config_dict = self._convert_config_dict(cmd_config_dict)
        return cmd_config_dict

    def _merge_external_config_dict(self):
        external_config_dict = dict()
        external_config_dict.update(self.file_config_dict)
        external_config_dict.update(self.variable_config_dict)
        external_config_dict.update(self.cmd_config_dict)
        self.external_config_dict = external_config_dict

    def _get_model_and_dataset(self, model, dataset):

        if model is None:
            try:
                model = self.external_config_dict['model']
            except KeyError:
                raise KeyError(
                    'model need to be specified in at least one of the these ways: '
                    '[model variable, config file, config dict, command line] ')
        if not isinstance(model, str):
            final_model_class = model
            final_model = model.__name__
        else:
            final_model = model
            final_model_class = get_model(final_model)

        if dataset is None:
            try:
                final_dataset = self.external_config_dict['dataset']
            except KeyError:
                raise KeyError('dataset need to be specified in at least one of the these ways: '
                               '[dataset variable, config file, config dict, command line] ')
        else:
            final_dataset = dataset

        return final_model, final_model_class, final_dataset

    def _load_internal_config_dict(self, model, model_class, dataset):
        current_path = os.path.dirname(os.path.realpath(__file__))
        overall_init_file = os.path.join(current_path, '../properties/overall.yaml')
        model_init_file = os.path.join(current_path, '../properties/model/' + model + '.yaml')
        sample_init_file = os.path.join(current_path, '../properties/dataset/sample.yaml')
        dataset_init_file = os.path.join(current_path, '../properties/dataset/' + dataset + '.yaml')

        self.internal_config_dict = dict()
        for file in [overall_init_file, model_init_file, sample_init_file, dataset_init_file]:
            if os.path.isfile(file):
                with open(file, 'r', encoding='utf-8') as f:
                    config_dict = yaml.load(f.read(), Loader=self.yaml_loader)
                    if file == dataset_init_file:
                        self.parameters['Dataset'] += [key for key in config_dict.keys() if
                                                       key not in self.parameters['Dataset']]
                    if config_dict is not None:
                        self.internal_config_dict.update(config_dict)
        self.internal_config_dict['MODEL_TYPE'] = model_class.type
        if self.internal_config_dict['MODEL_TYPE'] == ModelType.GENERAL:
            pass
        elif self.internal_config_dict['MODEL_TYPE'] == ModelType.CONTEXT:
            self.internal_config_dict.update({
                'eval_setting': 'RO_RS',
                'group_by_user': False,
                'training_neg_sample_num': 0,
                'metrics': ['AUC', 'LogLoss'],
                'valid_metric': 'AUC',
            })
            if dataset == 'ml-100k':
                self.internal_config_dict.update({
                    'threshold': {'rating': 4},
                    'load_col': {'inter': ['user_id', 'item_id', 'rating', 'timestamp'],
                                 'user': ['user_id', 'age', 'gender', 'occupation'],
                                 'item': ['item_id', 'release_year', 'class']},
                })

        elif self.internal_config_dict['MODEL_TYPE'] == ModelType.SEQUENTIAL:
            if model == 'DIN':
                self.internal_config_dict.update({
                    'eval_setting': 'TO_LS, uni100',
                    'metrics': ['AUC', 'LogLoss'],
                    'valid_metric': 'AUC',
                })
                if dataset == 'ml-100k':
                    self.internal_config_dict.update({
                        'load_col': {'inter': ['user_id', 'item_id', 'rating', 'timestamp'],
                                     'user': ['user_id', 'age', 'gender', 'occupation'],
                                     'item': ['item_id', 'release_year']},
                    })

            else:
                self.internal_config_dict.update({
                    'eval_setting': 'TO_LS,full',
                })
                if dataset == 'ml-100k' and model in ['GRU4RecF', 'SASRecF', 'FDSA', 'S3Rec']:
                    self.internal_config_dict.update({
                        'load_col': {'inter': ['user_id', 'item_id', 'rating', 'timestamp'],
                                     'item': ['item_id', 'release_year', 'class']},
                    })

        elif self.internal_config_dict['MODEL_TYPE'] == ModelType.KNOWLEDGE:
            self.internal_config_dict.update({
                'load_col': {'inter': ['user_id', 'item_id', 'rating', 'timestamp'],
                             'kg': ['head_id', 'relation_id', 'tail_id'],
                             'link': ['item_id', 'entity_id']}
            })

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

        if hasattr(self.model_class, 'input_type'):
            self.final_config_dict['MODEL_INPUT_TYPE'] = self.model_class.input_type
        elif 'loss_type' in self.final_config_dict:
            if self.final_config_dict['loss_type'] in ['CE']:
                self.final_config_dict['MODEL_INPUT_TYPE'] = InputType.POINTWISE
            elif self.final_config_dict['loss_type'] in ['BPR']:
                self.final_config_dict['MODEL_INPUT_TYPE'] = InputType.PAIRWISE
        else:
            raise ValueError('Either Model has attr \'input_type\','
                             'or arg \'loss_type\' should exist in config.')

        eval_type = None
        for metric in self.final_config_dict['metrics']:
            if metric.lower() in loss_metrics:
                if eval_type is not None and eval_type == EvaluatorType.RANKING:
                    raise RuntimeError('Ranking metrics and other metrics can not be used at the same time.')
                else:
                    eval_type = EvaluatorType.INDIVIDUAL
            if metric.lower() in topk_metrics:
                if eval_type is not None and eval_type == EvaluatorType.INDIVIDUAL:
                    raise RuntimeError('Ranking metrics and other metrics can not be used at the same time.')
                else:
                    eval_type = EvaluatorType.RANKING
        self.final_config_dict['eval_type'] = eval_type

        smaller_metric = ['rmse', 'mae', 'logloss']
        valid_metric = self.final_config_dict['valid_metric'].split('@')[0]
        self.final_config_dict['valid_metric_bigger'] = False if valid_metric in smaller_metric else True

        if 'additional_feat_suffix' in self.final_config_dict:
            ad_suf = self.final_config_dict['additional_feat_suffix']
            if isinstance(ad_suf, str):
                self.final_config_dict['additional_feat_suffix'] = [ad_suf]

    def _init_device(self):
        use_gpu = self.final_config_dict['use_gpu']
        if use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.final_config_dict['gpu_id'])
        self.final_config_dict['device'] = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")

    def __setitem__(self, key, value):
        if not isinstance(key, str):
            raise TypeError("index must be a str.")
        self.final_config_dict[key] = value

    def __getitem__(self, item):
        if item in self.final_config_dict:
            return self.final_config_dict[item]
        else:
            return None

    def __contains__(self, key):
        if not isinstance(key, str):
            raise TypeError("index must be a str.")
        return key in self.final_config_dict

    def __str__(self):
        args_info = ''
        for category in self.parameters:
            args_info += category + ' Hyper Parameters: \n'
            args_info += '\n'.join(
                ["{}={}".format(arg, value)
                 for arg, value in self.final_config_dict.items()
                 if arg in self.parameters[category]])
            args_info += '\n\n'
        return args_info

    def __repr__(self):
        return self.__str__()
