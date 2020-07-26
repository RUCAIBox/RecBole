# -*- coding: utf-8 -*-
# @Time   : 2020/7/24 15:57
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn
# @File   : run_hyper.py

import importlib
from config import Config
from data import Dataset, data_preparation
from trainer import Trainer, HyperTuning
from utils import Logger


def objective_function(config_dict=None):
    config = Config('properties/overall.config', config_dict)
    config.init()

    logger = Logger(config)

    dataset = Dataset(config)
    print(dataset)

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

    train_data, test_data, valid_data = data_preparation(config, model, dataset)

    trainer = Trainer(config, model, logger)
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data)

    return best_valid_score, config['valid_metric_bigger']


def main():
    hp = HyperTuning(objective_function, params_file='hyper.test')
    hp.run()
    print(hp.best_params)


if __name__ == '__main__':
    main()
