# -*- coding: utf-8 -*-
# @Time   : 2020/6/28
# @Author : Zihan Lin
# @Email  : linzihan.super@foxmail.com
from recbox.config.abstract_configurator import AbstractConfig


class DataConfig(AbstractConfig):
    def __init__(self, config_file_name, cmd_args):
        super().__init__()
        self.must_args = []
        self.args = self._read_config_file(config_file_name, 'data')
        self._check_args()

        self._replace_args(cmd_args)
