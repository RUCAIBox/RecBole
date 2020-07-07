from configparser import ConfigParser
from config.abstract_configurator import AbstractConfig


class ModelConfig(AbstractConfig):
    def __init__(self, config_file_name):
        super().__init__()
        self.must_args = []
        self.args = self._read_config_file(config_file_name, 'model')
        for cmd_arg_name, cmd_arg_value in self.cmd_args.items():
            if cmd_arg_name.startswith('model.'):
                self.args[cmd_arg_name] = cmd_arg_value

        self.default_args = self.default_parser.getargs()
        for default_args in self.default_args.keys():

            if str(default_args) not in self.args and str(default_args).startswith('model.'):
                self.args[str(default_args)] = str(self.default_args[default_args])

        self._check_args()

    def dump_config_file(self, config_file):
        """
        This function can dump the model's hyper parameters to a new config file
        :param config_file: file name that write to.
        """
        model_config = ConfigParser()
        model_config['model'] = self.args
        with open(config_file, 'w') as configfile:
            model_config.write(configfile)