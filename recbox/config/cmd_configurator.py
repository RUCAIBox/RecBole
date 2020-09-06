from .abstract_configurator import AbstractConfig


class CmdConfig(AbstractConfig):
    def __init__(self, cmd_args):
        super().__init__()
        self.must_args = []
        self.args = cmd_args

        self._check_args()