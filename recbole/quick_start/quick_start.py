# @Time   : 2020/10/6
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn

"""
recbole.quick_start
########################
"""

from logging import getLogger
from recbole.utils import init_logger, get_model, get_trainer, init_seed
from recbole.config import Config
from recbole.data import create_dataset, data_preparation


def run_recbole(model=None, dataset=None, config_file_list=None, config_dict=None, saved=True):

    # configurations initialization
    config = Config(model=model, dataset=dataset, config_file_list=config_file_list, config_dict=config_dict)
    init_seed(config['seed'])

    # logger initialization
    init_logger(config)
    logger = getLogger()

    logger.info(config)

    # dataset filtering
    dataset = create_dataset(config)
    logger.info(dataset)

    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # model loading and initialization
    model = get_model(config['model'])(config, train_data).to(config['device'])
    logger.info(model)

    # trainer loading and initialization
    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)

    # model training
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data, saved=saved)

    # model evaluation
    test_result = trainer.evaluate(test_data, load_best_model=saved)

    logger.info('best valid result: {}'.format(best_valid_result))
    logger.info('test result: {}'.format(test_result))

    return {
        'best_valid_score': best_valid_score,
        'valid_score_bigger': config['valid_metric_bigger'],
        'best_valid_result': best_valid_result,
        'test_result': test_result
    }


def objective_function(config_dict=None, config_file_list=None):

    config = Config(config_dict=config_dict, config_file_list=config_file_list)
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
