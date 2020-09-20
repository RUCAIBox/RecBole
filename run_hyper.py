# -*- coding: utf-8 -*-
# @Time   : 2020/7/24 15:57
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn
# @File   : run_hyper.py
# UPDATE:
# @Time   : 2020/8/20 21:17, 2020/8/29
# @Author : Zihan Lin, Yupeng Hou
# @Email  : linzihan.super@foxmail.com, houyupeng@ruc.edu.cn

from recbox.config import Config
from recbox.data import create_dataset, data_preparation
from recbox.trainer import HyperTuning
from recbox.utils import get_model, get_trainer


def objective_function(config_dict=None):

    config = Config('properties/overall.config', config_dict)
    config.init()
    dataset = create_dataset(config)

    train_data, test_data, valid_data = data_preparation(config, dataset)

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


def main():
    # plz set algo='exhaustive' to use exhaustive search, in this case, max_evals is auto set
    hp = HyperTuning(objective_function, algo='exhaustive', params_file='hyper.test')
    hp.run()
    hp.export_result(output_file='hyper_example.result')
    print('best params: ', hp.best_params)
    print('best result: ')
    print(hp.params2result[hp.params2str(hp.best_params)])


if __name__ == '__main__':
    main()
