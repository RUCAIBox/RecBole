# -*- coding: utf-8 -*-
# @Time   : 2020/10/1
# @Author : Zihan Lin
# @Email  : zhlin@ruc.edu.cn
import os
import yaml


class ConfigFileReader(object):
    r"""This class is a fundamental class for loading config file and wrap it as a dict.

    """
    def __init__(self, file_name, must_args=None):
        r"""This class can ONLY be initialized by the interface class `configurator`.

        Args:
            file_name(str):  The path of ini-style configuration file.
            must_args(list): The dict that contains parameters which MUST be set.
        """
        self.file_name = file_name
        self.args = dict()
        self.must_args = must_args
        self._read_config_file()

    def _read_config_file(self):
        r"""This function is a protected function that read the config file and convert it to a dict

        """
        if not os.path.isfile(self.file_name):
            raise FileNotFoundError("There is no config file named '%s'!" % self.file_name)

        f = open(self.file_name, 'r', encoding='utf-8')

        self.args = yaml.load(f.read(), Loader=yaml.FullLoader)

    def _check_args(self):
        r"""This function is a protected function that check MUST parameters

        """
        for parameter in self.must_args:
            if parameter not in self.args:
                raise ValueError("'%s' must be specified !" % parameter)

    def __getitem__(self, item):

        if not isinstance(item, str):
            raise TypeError("index must be a str")
        # Get device or other parameters

        if item in self.args:
            param = self.args[item]
        else:
            raise KeyError("There are no parameter named '%s'" % item)

        # convert param from str to value, i.e. int, float or list etc.

        value = param

        return value

    def __setitem__(self, key, value):

        if not isinstance(key, str):
            raise TypeError("index must be a str")
        if not isinstance(value, str):
            raise TypeError("value must be a str")

        self.args[key] = value

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
