# -*- coding: utf-8 -*-
# @Time   : 2020/10/6 下午9:45
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn
# @File   : scipts.py

from logging import getLogger
from recbox.utils import init_logger, get_model, get_trainer, init_seed
from recbox.config import Config
from recbox.data import create_dataset, data_preparation


def run_unirec(model=None, dataset=None, config_file_list=None, config_dict=None, saved=True):
    """
    初始化 config
    """
    config = Config(model=model, dataset=dataset, config_file_list=config_file_list, config_dict=config_dict)
    init_seed(config['seed'])

    """
    初始化 logger
    """
    init_logger(config)
    logger = getLogger()

    logger.info(config)

    """
    初始化 dataset
    """
    dataset = create_dataset(config)
    logger.info(dataset)

    """
    生成 训练/验证/测试 数据
    """
    train_data, valid_data, test_data = data_preparation(config, dataset)

    """
    初始化 model
    """
    model = get_model(config['model'])(config, train_data).to(config['device'])
    logger.info(model)

    """
    初始化 trainer
    """
    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)

    """
    训练
    """
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data, saved=saved)

    """
    测试
    """
    test_result = trainer.evaluate(test_data, load_best_model=saved)

    logger.info('best valid result: {}'.format(best_valid_result))
    logger.info('test result: {}'.format(test_result))

    return {
        'best_valid_score': best_valid_score,
        'valid_score_bigger': config['valid_metric_bigger'],
        'best_valid_result': best_valid_result,
        'test_result': test_result
    }


def objective_function(config_dict=None):

    config = Config(config_dict=config_dict)
    init_seed(config['seed'])
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    model = get_model(config['model'])(config, train_data).to(config['device'])
    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data, verbose=False)
    test_result = trainer.evaluate(test_data)

    return {
        'best_valid_score': best_valid_score,
        'valid_score_bigger': config['valid_metric_bigger'],
        'best_valid_result': best_valid_result,
        'test_result': test_result
    }
