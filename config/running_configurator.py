from config.abstract_configurator import AbstractConfig


class RunningConfig(AbstractConfig):
    def __init__(self, config_file_name):

        super().__init__()
        self.must_args = ['model', 'dataset', 'dataset.path']
        self.args = self._read_config_file(config_file_name, 'default')
        for cmd_arg_name, cmd_arg_value in self.cmd_args.items():
            if not cmd_arg_name.startswith('model.') or not cmd_arg_name.startswith('data.'):
                self.args[cmd_arg_name] = cmd_arg_value

        self.default_args = self.default_parser.getargs()
        for default_args in self.default_args.keys():
            if str(default_args) not in self.args and not str(default_args).startswith('model.')  \
                                                  and not str(default_args).startswith('data.'):
                self.args[str(default_args)] = str(self.default_args[default_args])

        self._check_args()