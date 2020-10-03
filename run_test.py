# -*- coding: utf-8 -*-
# @Time   : 2020/7/20 20:32
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn
# @File   : run_test.py

# UPDATE
# @Time   : 2020/10/3, 2020/10/1
# @Author : Yupeng Hou, Zihan Lin
# @Email  : houyupeng@ruc.edu.cn, zhlin@ruc.edu.cn

from logging import getLogger
from recbox.utils import init_logger, get_model, get_trainer, init_seed
from recbox.config import Config
from recbox.data import create_dataset, data_preparation


def whole_process(config_file='properties/overall.yaml', config_dict=None, saved=True):
    """
    初始化 config
    """
    config = Config(config_file, config_dict)
    init_seed(config['seed'])

    """
    初始化 logger
    """
    init_logger(config)
    logger = getLogger()

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
    #model.dump_parameters()


def run_test():
    whole_process()


if __name__ == '__main__':
    run_test()
