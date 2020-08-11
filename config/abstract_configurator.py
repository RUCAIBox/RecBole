import os
import sys
from configparser import ConfigParser


class AbstractConfig(object):
    def __init__(self):
        self.cmd_args = dict()
        self.args = dict()
        self.must_args = []

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
            if parameter not in self.args:
                raise ValueError("'%s' must be specified !" % parameter)

    def _replace_args(self, cmd_args):
        for arg_name in self.args.keys():
            if arg_name in cmd_args:
                self.args[arg_name] = cmd_args[arg_name]

    def __getitem__(self, item):

        if not isinstance(item, str):
            raise TypeError("index must be a str")
        # Get device or other parameters

        if item in self.args:
            param = self.args[item]
        else:
            raise KeyError("There are no parameter named '%s'" % item)

        # convert param from str to value, i.e. int, float or list etc.
        try:
            value = eval(param)
            if not isinstance(value, (str, int, float, list, tuple, dict, bool, None.__class__)):
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
        return value

    def __setitem__(self, key, value):

        if not isinstance(key, str):
            raise TypeError("index must be a str")
        if isinstance(value, str):
            self.args[key] = value
        else:
            self.args[key] = str(value)

    def __contains__(self, o):
        if not isinstance(o, str):
            raise TypeError("index must be a str!")
        return o in self.args

    def __str__(self):
        args_info = '\n'.join(
            ["{}={}".format(arg, value) for arg, value in self.args.items()])
        return args_info

    def __repr__(self):

        return self.__str__()