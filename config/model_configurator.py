from config.abstract_configurator import AbstractConfig


class ModelConfig(AbstractConfig):
    def __init__(self, config_file_name, cmd_args):
        super().__init__()
        self.must_args = []
        self.args = self._read_config_file(config_file_name, 'model')

        self._check_args()
        self._replace_args(cmd_args)

