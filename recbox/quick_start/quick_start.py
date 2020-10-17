# @Time   : 2020/10/6
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn

"""
recbox.quick_start
########################
"""

import os
import yaml

from logging import getLogger
from recbox.utils import init_logger, get_model, get_trainer, init_seed
from recbox.config import Config
from recbox.data import create_dataset, data_preparation


def load_presets():
    current_path = os.path.dirname(os.path.realpath(__file__))
    presets_file = os.path.join(current_path, '../properties/presets.yaml')
    with open(presets_file, 'r', encoding='utf-8') as fp:
        presets_dict = yaml.load(fp.read(), Loader=yaml.FullLoader)
    return presets_dict


def run_unirec(model=None, dataset=None, config_file_list=None, config_dict=None, saved=True, presets=False):

    # configurations initialization
    if presets:
        assert model is not None and dataset is not None, \
            "Quick-start required positional arguments: 'model' and 'dataset'"

        presets_dict = load_presets()
        try:
            config_dict = presets_dict['-'.join([model, dataset])]
        except KeyError:
            raise KeyError('UniRec only support quick-start of the following combinations:  '
                           + str(list(presets_dict.keys())))
        config = Config(model=model, dataset=dataset, config_dict=config_dict)
    else:
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
