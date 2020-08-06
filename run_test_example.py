# -*- coding: utf-8 -*-
# @Time   : 2020/7/23 20:34
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn
# @File   : run_test_example.py

import traceback
from run_test import whole_process


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
        'metrics:': ['Recall', 'MRR', 'NDCG'],
        'topk': [5, 10, 20],
    },
    'Test Real Time Full Sort': {
        'model': 'BPRMF',
        'dataset': 'ml-100k',
        'epochs': 1,
        'valid_metric': 'Recall@10',
        'metrics:': ['Recall'],
        'topk': [10],
        'eval_setting': 'RO_RS, full',
        'real_time_neg_sampling': True
    },
    'Test Pre Full Sort': {
        'model': 'BPRMF',
        'dataset': 'ml-100k',
        'epochs': 1,
        'valid_metric': 'Recall@10',
        'metrics:': ['Recall'],
        'topk': [10],
        'eval_setting': 'RO_RS, full',
        'real_time_neg_sampling': False
    },
    'Test Real Time Neg Sample By': {
        'model': 'BPRMF',
        'dataset': 'ml-100k',
        'epochs': 1,
        'valid_metric': 'Recall@10',
        'metrics:': ['Recall'],
        'topk': [10],
        'eval_setting': 'RO_RS, uni100',
        'real_time_neg_sampling': True
    },
    'Test Pre Neg Sample By': {
        'model': 'BPRMF',
        'dataset': 'ml-100k',
        'epochs': 1,
        'valid_metric': 'Recall@10',
        'metrics:': ['Recall'],
        'topk': [10],
        'eval_setting': 'RO_RS, uni100',
        'real_time_neg_sampling': False
    },
    'Test Leave One Out': {
        'model': 'BPRMF',
        'dataset': 'ml-100k',
        'epochs': 1,
        'valid_metric': 'Recall@10',
        'metrics:': ['Recall'],
        'topk': [10],
        'eval_setting': 'RO_LS, full',
        'real_time_neg_sampling': True
    },
    'Test BPRMF': {
        'model': 'BPRMF',
        'dataset': 'ml-100k',
        'epochs': 1,
        'valid_metric': 'Recall@10',
        'metrics:': ['Recall'],
        'topk': [10]
    },
    'Test NeuMF': {
        'model': 'NeuMF',
        'input_format': 'pointwise',
        'dataset': 'ml-100k',
        'epochs': 1,
        'valid_metric': 'Recall@10',
        'metrics:': ['Recall'],
        'topk': [10]
    },
    'Test POP': {
        'model': 'Pop',
        'dataset': 'ml-100k',
        'epochs': 1,
        'valid_metric': 'Recall@10',
        'metrics:': ['Recall'],
        'topk': [10]
    }

}


def run_test_examples():

    success_examples, fail_examples = [], []
    n_examples = len(test_examples.keys())
    for idx, example in enumerate(test_examples.keys()):
        print('\n\n Begin to run %d / %d example: %s \n\n' % (idx + 1, n_examples, example))
        try:
            whole_process(config_file='properties/overall.config', config_dict=test_examples[example])
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
