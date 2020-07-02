import os
import torch
import random
import numpy as np
from config.running_configurator import RunningConfig
from config.model_configurator import ModelConfig


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

        self.run_args = RunningConfig(config_file_name)

        model_name = self.run_args['model']
        model_arg_file_name = os.path.join(os.path.dirname(config_file_name), model_name + '.config')
        self.model_args = ModelConfig(model_arg_file_name)

        self.device = None

    def init(self):
        """
        This function is a global initialization function that fix random seed and gpu device.
        """
        init_seed = self.run_args['seed']
        gpu_id = self.run_args['gpu_id']
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Get the device that run on.

        random.seed(init_seed)
        np.random.seed(init_seed)
        torch.manual_seed(init_seed)
        torch.cuda.manual_seed(init_seed)
        torch.cuda.manual_seed_all(init_seed)
        torch.backends.cudnn.deterministic = True

    def dump_config_file(self, config_file):
        """
        This function can dump the model's hyper parameters to a new config file
        :param config_file: file name that write to.
        """
        self.model_args.dump_config_file(config_file)

    def __getitem__(self, item):
        if item == "device":
            if self.device is None:
                raise SyntaxError("device only can be get after init() !")
            else:
                return self.device
        elif item in self.run_args:
            return self.run_args[item]
        elif item in self.model_args:
            return self.model_args[item]
        else:
            raise KeyError("There are no parameter named '%s'" % item)

    def __setitem__(self, key, value):
        if not isinstance(key, str):
            raise TypeError("index must be a str")
        if key in self.run_args:
            raise SyntaxError("running args can not be changed when running!")
        self.model_args[key] = value

    def __contains__(self, o):
        if not isinstance(o, str):
            raise TypeError("index must be a str!")
        return o in self.run_args or o in self.model_args

    def __str__(self):

        run_args_info = str(self.run_args)
        model_args_info = str(self.model_args)
        info = "\nRunning Hyper Parameters:\n%s\n\nRunning Model:%s\n\nModel Hyper Parameters:\n%s\n" % \
               (run_args_info, self.run_args['model'], model_args_info)
        return info

    def __repr__(self):

        return self.__str__()


if __name__ == '__main__':
    config = Config('../properties/overall.config')
    config.init()
    # print(config)
    print(config['process.split_by_ratio.train_ratio'])
    print(config['eval.metric'])
    print(config['eval.topk'])
    print(config['model.learning_rate'])
    print(config['eval.group_view'])
    print(config['data.separator'])
    print(config['model.reg_mf'])
    print(config['device'])
    config['model.reg_mf'] = 0.6
    print(config['model.reg_mf'])
    #config.dump_config_file('../properties/mf_new.config')
