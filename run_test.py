# -*- coding: utf-8 -*-
# @Time   : 2020/7/20 20:32
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn
# @File   : run_test.py

import importlib
from trainer import Trainer
from utils import Logger
from config import Config
from data import Dataset


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
    这部分可能随着开发会有所改变，可能需要自行进行改变
    """
    config = Config('properties/overall.config')
    config.init()
    dataset = Dataset(config)
    train_data, test_data, valid_data = dataset.build(
        inter_filter_lowest_val=config['lowest_val'],
        inter_filter_highest_val=config['highest_val'],
        split_by_ratio=[config['train_split_ratio'], config['valid_split_ratio'], config['test_split_ratio']],
        train_batch_size=config['train_batch_size'],
        test_batch_size=config['test_batch_size'],
        valid_batch_size=config['valid_batch_size'],
        pairwise=True,
        neg_sample_by=1,
        neg_sample_to=config['test_neg_sample_num']
    )

    """
    调用模型并进行测试
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
    mt = ModelTest(config, model)
    valid_score, _, _ = mt.run(train_data, test_data, valid_data)
    print(valid_score)


if __name__ == '__main__':
    run_test()
