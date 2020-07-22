from config.abstract_configurator import AbstractConfig
import os


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

