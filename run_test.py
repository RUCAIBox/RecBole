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


def whole_process(config_file='properties/overall.config', config_dict=None):
    """
    初始化 config
    """
    config = Config(config_file, config_dict)
    config.init()

    """
    初始化 logger
    """
    logger = Logger(config)

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
    初始化 trainer
    """
    trainer = Trainer(config, model, logger)

    """
    训练
    """
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data)

    """
    测试
    """
    test_result = trainer.evaluate(test_data)
    print('test result: ', test_result)


def run_test():
    whole_process()


if __name__ == '__main__':
    run_test()
