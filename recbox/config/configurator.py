# -*- coding: utf-8 -*-
# @Time   : 2020/6/28
# @Author : Zihan Lin
# @Email  : linzihan.super@foxmail.com
import os
import sys
from logging import getLogger
from recbox.evaluator import loss_metrics, topk_metrics

import torch
from recbox.utils import *
from recbox.config.config_file_reader import ConfigFileReader


class Config(object):
    r"""A configuration class that load predefined hyper parameters.

    This class can read arguments from ini-style configuration file. There are two type
    of config file can be defined: running config file and model config file. Each file should
    be named as XXX.config, and model config file MUST be named as the model name.
    In the ini-style config file, only one section is cared. For running config file, the section
    MUST be [default] . For model config file, it MUST be [model]

    There are three parameter MUST be included in config file: model, data.name, data.path

    After initialization successful, the objective of this class can be used as
    a dictionary:
        config = Configurator("./overall.config")
        ratio = config["ratio"]
        metric = config["metric"]
    All the parameter key MUST be str, but returned value is exactly the corresponding type

    support parameter type: str, int, float, list, tuple, bool, None
    """

    def __init__(self, config_file_name, config_dict=None):
        """
        Args:
            config_file_name(str): The path of ini-style configuration file.
            config_dict(dict) : The parameters dict if you want to transmit a dict to get a `Config` object.

        Raises:
            FileNotFoundError: If `config_file` is not existing.
            ValueError: If `config_file` is not in correct format or
                        MUST parameter are not defined
        """
        self.config_dict = config_dict
        self.cmd_args_dict = {}
        self._read_cmd_line()
        if self.config_dict:
            self._read_config_dict()
        self.convert_cmd_args()

        self.run_args = ConfigFileReader(config_file_name, must_args=['model', 'dataset'])

        model_name = self['model']
        model_dir = os.path.join(os.path.dirname(config_file_name), 'model')
        model_arg_file_name = os.path.join(model_dir, model_name + '.yaml')
        self.model_args = ConfigFileReader(model_arg_file_name)

        dataset_name = self['dataset']
        dataset_dir = os.path.join(os.path.dirname(config_file_name), 'dataset')
        dataset_arg_file_name = os.path.join(dataset_dir, dataset_name + '.yaml')
        self.dataset_args = ConfigFileReader(dataset_arg_file_name)

        self._set_default_parameters()

        self.device = None
        self._init_device()

    def convert_cmd_args(self):
        r"""This function convert the str parameters to their original type.

        """
        for key in self.cmd_args_dict:
            param = self.cmd_args_dict[key]
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
            self.cmd_args_dict[key] = value

    def _init_device(self):
        r"""This function is a global initialization function that fix random seed and gpu device.

        """
        use_gpu = self.cmd_args_dict['use_gpu']
        if use_gpu:
            gpu_id = self['gpu_id']
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        # Get the device that run on.
        self.device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")

    def _read_cmd_line(self):
        r""" Read parameters from command line and convert it to str.

        """
        unrecognized_args = []
        if "ipykernel_launcher" not in sys.argv[0]:
            for arg in sys.argv[1:]:
                if not arg.startswith("--") or len(arg[2:].split("=")) != 2:
                    unrecognized_args.append(arg)
                    continue
                cmd_arg_name, cmd_arg_value = arg[2:].split("=")
                if cmd_arg_name in self.cmd_args_dict and cmd_arg_value != self.cmd_args_dict[cmd_arg_name]:
                    raise SyntaxError("There are duplicate commend arg '%s' with different value!" % arg)
                else:
                    self.cmd_args_dict[cmd_arg_name] = cmd_arg_value
        if len(unrecognized_args) > 0:
            logger = getLogger()
            logger.warning('command line args [{}] will not be used in RecBox'.format(' '.join(unrecognized_args)))

    def _read_config_dict(self):
        r"""Convert parameters in dict into cmd parameters to keep the priority.

        """
        for dict_arg_name in self.config_dict:
            if dict_arg_name not in self.cmd_args_dict:
                if isinstance(self.config_dict[dict_arg_name], str):
                    self.cmd_args_dict[dict_arg_name] = self.config_dict[dict_arg_name]
                else:
                    self.cmd_args_dict[dict_arg_name] = str(self.config_dict[dict_arg_name])

    def _set_default_parameters(self):
        r"""This function can automatically set some parameters that don't need be set by user.
        """
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

        smaller_metric = ['rmse', 'mae', 'logloss']

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

        model = get_model(self['model'])
        self['MODEL_TYPE'] = model.type
        self['MODEL_INPUT_TYPE'] = model.input_type

    def __setitem__(self, key, value):

        if not isinstance(key, str):
            raise TypeError("index must be a str")

        self.cmd_args_dict[key] = value

    def __getitem__(self, item):
        if item == "device":
            return self.device
        elif item in self.cmd_args_dict:
            return self.cmd_args_dict[item]
        elif item in self.run_args:
            return self.run_args[item]
        elif item in self.model_args:
            return self.model_args[item]
        elif item in self.dataset_args:
            return self.dataset_args[item]
        else:
            return None

    def __contains__(self, o):
        if not isinstance(o, str):
            raise TypeError("index must be a str!")
        return o in self.run_args or o in self.model_args or o in self.dataset_args

    def __str__(self):

        run_args_info = str(self.run_args)
        model_args_info = str(self.model_args)
        dataset_args_info = str(self.dataset_args)
        info = "\nRunning Hyper Parameters:\n%s\n\nRunning Model:%s\n\nDataset Hyper Parameters:%s\n\n" \
               "Model Hyper Parameters:\n%s\n" % \
               (run_args_info, self.run_args['model'], dataset_args_info, model_args_info)
        return info

    def __repr__(self):

        return self.__str__()
