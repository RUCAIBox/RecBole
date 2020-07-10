from config.abstract_configurator import AbstractConfig


class ModelConfig(AbstractConfig):
    def __init__(self, config_file_name, cmd_args):
        super().__init__()
        self.must_args = ['train_batch_size']
        self.args = self._read_config_file(config_file_name, 'model')

        self._check_args()
        self._replace_args(cmd_args)

    def dump_config_file(self, config_file):
        """
        This function can dump the model's hyper parameters to a new config file
        :param config_file: file name that write to.
        """
        # model_config = ConfigParser()
        # model_config['model'] = self.args
        # with open(config_file, 'w') as configfile:
        #     model_config.write(configfile)
        raise NotImplementedError
