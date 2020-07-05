from config.abstract_configurator import AbstractConfig


class DataConfig(AbstractConfig):
    def __init__(self, config_file_name):
        super().__init__()
        self.must_args = []
        self.args = self._read_config_file(config_file_name, 'data')
        for cmd_arg_name, cmd_arg_value in self.cmd_args.items():
            if cmd_arg_name.startswith('data.'):
                self.args[cmd_arg_name] = cmd_arg_value

        self.default_args = self.default_parser.getargs()
        for default_args in self.default_args.keys():

            if str(default_args) not in self.args and (str(default_args).startswith('data.')):
                self.args[str(default_args)] = str(self.default_args[default_args])

        self._check_args()
