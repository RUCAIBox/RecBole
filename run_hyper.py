# -*- coding: utf-8 -*-
# @Time   : 2020/7/24 15:57
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn
# @File   : run_hyper.py


from config import Config
from data import Dataset, data_preparation
from trainer import Trainer, HyperTuning
from utils import get_logger, get_model


def data_preparation_function():
    config = Config('properties/overall.config')
    config.init()

    logger = get_logger(config)

    dataset = Dataset(config)
    print(dataset)

    model = get_model(config)(config, dataset).to(config['device'])
    print(model)

    train_data, test_data, valid_data = data_preparation(config, model, dataset)

    dataloader = {
        'train_data': train_data,
        'test_data': test_data,
        'valid_data': valid_data
    }
    return dataset, dataloader


def objective_function(dataset, dataloader, config_dict=None):
    config = Config('properties/overall.config', config_dict)
    config.init()

    assert(config['dataset'] == dataset.dataset_name)
    # todo: 判断eval_setting是否一致

    logger = get_logger()

    model = get_model(config)(config, dataset).to(config['device'])

    train_data, test_data, valid_data = dataloader['train_data'], dataloader['test_data'], dataloader['valid_data']

    trainer = Trainer(config, model)
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data, verbose=False)
    test_result = trainer.evaluate(test_data)

    return {
        'best_valid_score': best_valid_score,
        'valid_score_bigger': config['valid_metric_bigger'],
        'best_valid_result': best_valid_result,
        'test_result': test_result
    }


def main():
    # plz set algo='exhaustive' to use exhaustive search, in this case, max_evals is auto set
    hp = HyperTuning(data_preparation_function, objective_function, algo='exhaustive', params_file='hyper.test', max_evals=5)
    hp.run()
    print('best params: ', hp.best_params)
    print('best result: ')
    print(hp.params2result[hp.params2str(hp.best_params)])


if __name__ == '__main__':
    main()
