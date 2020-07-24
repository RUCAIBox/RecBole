# -*- coding: utf-8 -*-
# @Time   : 2020/7/20 20:32
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn
# @File   : run_test.py

import importlib
from trainer import Trainer
from utils import Logger
from config import Config
from data import Dataset, data_preparation


class ModelTest(object):

    def __init__(self, config, model):
        self.logger = Logger(config)
        self.trainer = Trainer(config, model, self.logger)

    def run(self, train_data, valid_data, test_data):
        best_valid_score, best_valid_result = self.trainer.fit(train_data, valid_data)
        test_result = self.trainer.evaluate(test_data)
        return best_valid_score, best_valid_result, test_result


def run_test():

    """
    初始化 config
    """
    config = Config('properties/overall.config')
    config.init()

    """
    初始化 dataset
    """
    dataset = Dataset(config)
    print(dataset)

    """
    初始化 model
    """
    model_name = config['model']
    model_file_name = model_name.lower()
    if importlib.util.find_spec("model.general_recommender." + model_file_name) is not None:
        model_module = importlib.import_module("model.general_recommender." + model_file_name)
    elif importlib.util.find_spec("model.context_aware_recommender." + model_file_name) is not None:

        model_module = importlib.import_module("model.context_aware_recommender." + model_file_name)
    else:
        model_module = importlib.import_module("model.sequential_recommender." + model_file_name)

    model_class = getattr(model_module, model_name)
    model = model_class(config, dataset).to(config['device'])
    print(model)

    """
    生成 训练/验证/测试 数据
    """
    train_data, test_data, valid_data = data_preparation(config, model, dataset)

    """
    Model Test
    """
    mt = ModelTest(config, model)
    valid_score, _, _ = mt.run(train_data, test_data, valid_data)
    print(valid_score)


if __name__ == '__main__':
    run_test()
