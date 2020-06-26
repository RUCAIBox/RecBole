import os
import torch
import random
import numpy as np
from configparser import ConfigParser


class Config(object):
    """
    A configuration class that load predefined hyper parameters.

    This class can read arguments from ini-style configuration file. There are two type
    of config file can be defined: running config file and model config file. Each file should
    be named as XXX.config, and model config file MUST be named as the model name.
    In the ini-style config file, only one section is cared. For running config file, the section
    MUST be [default] . For model config file, it MUST be [model]

    There are three parameter MUST be included in config file: model, data.name, data.path

    After initialization successful, the objective of this class can be used as
    a dictionary:
        config = Configurator("./overall.config")
        ratio = config["process.ratio"]
        metric = config["eval.metric"]
    All the parameter key MUST be str, but returned value is exactly the corresponding type

    support parameter type: str, int, float, list, tuple, bool, None
    """
    def __init__(self, config_file_name):
        """

        :param config_file_name(str): The path of ini-style configuration file.

        :raises :
                FileNotFoundError: If `config_file` is not existing.
                ValueError: If `config_file` is not in correct format or
                        MUST parameter are not defined
        """
        self.must_args = ['model', 'data.name', 'data.path']

        self.run_args = self._read_config_file(config_file_name, 'default')
        self._check_args()
        model_name = self.run_args['model']
        model_arg_file_name = os.path.join(os.path.dirname(config_file_name), model_name + '.config')
        self.model_args = self._read_config_file(model_arg_file_name, 'model')

    def _read_config_file(self, file_name, arg_section):
        """
        This function is a protected function that read the config file and convert it to a dict
        :param file_name(str):  The path of ini-style configuration file.
        :param arg_section(str):  section name that distinguish running config file and model config file
        :return: A dict whose key and value are both str.
        """
        if not os.path.isfile(file_name):
            raise FileNotFoundError("There is no config file named '%s'!" % file_name)
        config = ConfigParser()
        config.optionxform = str
        config.read(file_name, encoding="utf-8")
        sections = config.sections()

        if len(sections) == 0:
            raise ValueError("'%s' is not in correct format!" % file_name)
        elif arg_section not in sections:
            raise ValueError("'%s' is not in correct format!" % file_name)
        else:
            config_arg = dict(config[arg_section].items())

        return config_arg

    def _check_args(self):
        """
        This function is a protected function that check MUST parameters
        """
        for parameter in self.must_args:
            if parameter not in self.run_args:
                raise ValueError("'%s' must be specified in configuration file!" % parameter)

    def init(self):
        """
        This function is a global initialization function that fix random seed and gpu device.
        """
        init_seed = self['seed']
        gpu_id = self['gpu_id']
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        random.seed(init_seed)
        np.random.seed(init_seed)
        torch.manual_seed(init_seed)
        torch.cuda.manual_seed(init_seed)
        torch.cuda.manual_seed_all(init_seed)
        torch.backends.cudnn.deterministic = True

    def __getitem__(self, item):

        if not isinstance(item, str):
            raise TypeError("index must be a str")

        if item in self.run_args:
            param = self.run_args[item]
        elif item in self.model_args:
            param = self.model_args[item]
        else:
            raise KeyError("There are no parameter named '%s'" % item)

        # convert param from str to value, i.e. int, float or list etc.
        try:
            value = eval(param)
            if not isinstance(value, (str, int, float, list, tuple, bool, None.__class__)):
                value = param
        except NameError:
            if param.lower() == "true":
                value = True
            elif param.lower() == "false":
                value = False
            else:
                value = param
        return value

    def __setitem__(self, key, value):

        if not isinstance(key, str):
            raise TypeError("index must be a str")
        if key in self.run_args:
            raise KeyError("Running parameter can't be changed")
        elif key not in self.model_args:
            raise KeyError("There are no model parameter named '%s'" % key)

        self.model_args[key] = str(value)

    def __contains__(self, o):
        if not isinstance(o, str):
            raise TypeError("index must be a str")
        return o in self.run_args or o in self.model_args

    def __str__(self):

        run_args_info = '\n'.join(
            ["{}={}".format(arg, value) for arg, value in self.run_args.items() if arg != 'model'])
        model_args_info = '\n'.join(["{}={}".format(arg, value) for arg, value in self.model_args.items()])
        info = "\nRunning Hyper Parameters:\n%s\n\nRunning Model:%s\n\nModel Hyper Parameters:\n%s\n" % \
               (run_args_info, self.run_args['model'], model_args_info)
        return info

    def __repr__(self):

        return self.__str__

    def dump_model_param(self, model_param_file):
        """
        This function can dump the model's hyper parameters to a new config file
        :param model_param_file: file name that write to.
        """
        model_config = ConfigParser()
        model_config['model'] = self.model_args
        with open(model_param_file, 'w') as configfile:
            model_config.write(configfile)


if __name__ == '__main__':
    config = Config('../properties/overall.config')
    config.init()
    print(config)
    print(config['process.ratio'])
    print(config['eval.metric'])
    print(config['learner'])
    print(config['eval.group_view'])
    print(config['process.by_time'])
    print(config['data.convert.separator'])
    print(config['reg_mf'])
    config['reg_mf'] = 0.6
    print(config['reg_mf'])
    config.dump_model_param('../properties/mf_new.config')
