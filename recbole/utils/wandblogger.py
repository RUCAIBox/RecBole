# -*- coding: utf-8 -*-
# @Time   : 2022/8/2
# @Author : Ayush Thakur
# @Email  : ayusht@wandb.com

r"""
recbole.utils.wandblogger
################################
"""

class WandbLogger(object):
    """WandbLogger to log metrics to Weights and Biases.

    """
    def __init__(self, config):
        """
        Args:
            config (dict): A dictionary of parameters used by RecBole.
        """
        self.config = config
        self.log_wandb = config.log_wandb
        self.setup()
        
    def setup(self):
        if self.log_wandb:
            try:
                import wandb
                self._wandb = wandb
            except ImportError:
                raise ImportError(
                    "To use the Weights and Biases Logger please install wandb."
                    "Run `pip install wandb` to install it."
                )

            # Initialize a W&B run
            if self._wandb.run is None:
                self._wandb.init(
                    project='recbole-test',
                    config=self.config
                )

    def log_metrics(self, metrics, head='train'):
        if self.log_wandb:
            if head:
                metrics = self.add_head_to_metrics(metrics, head)
                self._wandb.log(metrics)
            else:
                self._wandb.log(metrics)

    def add_head_to_metrics(self, metrics, head):
        head_metrics = dict()
        for k, v in metrics.items():
            head_metrics[f'{head}/{k}'] = v

        return head_metrics
        