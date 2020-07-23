# -*- coding: utf-8 -*-
# @Time   : 2020/7/23 20:34
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn
# @File   : run_test_example.py

import traceback
import importlib
from config import Config
from data import Dataset, data_preparation
from run_test import ModelTest


"""
乞丐版代码测试程序，防止bug越写越多，尤其是后期model多起来，一不小心就会使某些model run不起来
 
代码提交前，请运行一下这个程序，保证无误后再提交

有必要加入测试例子的，请尽量添加！按照格式添加到 `test_examples` 中

"""


test_examples = {
    'Test Eval Metric': {
        'model': 'BPRMF',
        'dataset': 'ml-100k',
        'epochs': 1,
        'valid_metric': 'Recall@10',
        'eval_metric:': ['Recall', 'MRR', 'NDCG'],
        'topk': [5, 10, 20],
    },
    'Test Full Sort': {
        'model': 'BPRMF',
        'dataset': 'ml-100k',
        'epochs': 1,
        'valid_metric': 'Recall@10',
        'eval_metric:': ['Recall'],
        'topk': [10],
        'test_neg_sample_num': -1
    },
    'Test BPRMF': {
        'model': 'BPRMF',
        'dataset': 'ml-100k',
        'epochs': 1,
        'valid_metric': 'Recall@10',
        'eval_metric:': ['Recall'],
        'topk': [10]
    },
    'Test NeuMF': {
        'model': 'NeuMF',
        'input_format': 'pointwise',
        'dataset': 'ml-100k',
        'epochs': 1,
        'valid_metric': 'Recall@10',
        'eval_metric:': ['Recall'],
        'topk': [10]
    },
    'Test POP': {
        'model': 'Pop',
        'dataset': 'ml-100k',
        'epochs': 1,
        'valid_metric': 'Recall@10',
        'eval_metric:': ['Recall'],
        'topk': [10]
    }

}


def run_test_examples():

    success_examples, fail_examples = [], []
    n_examples = len(test_examples.keys())
    for idx, example in enumerate(test_examples.keys()):
        print('\n\n Begin to run %d / %d example: %s \n\n' % (idx + 1, n_examples, example))
        config = Config('properties/overall.config', test_examples[example])
        config.init()

        try:
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

            mt = ModelTest(config, model)
            valid_score, _, _ = mt.run(train_data, test_data, valid_data)
            print('\n\n Running %d / %d example successfully: %s \n\n' % (idx + 1, n_examples, example))
            success_examples.append(example)
        except Exception:
            print(traceback.format_exc())
            fail_examples.append(example)
    print('success examples: ', success_examples)
    print('fail examples: ', fail_examples)
    print('\n')


if __name__ == '__main__':
    run_test_examples()
