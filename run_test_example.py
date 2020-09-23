# @Time   : 2020/7/23 20:34
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn

# UPDATE:
# @Time   : 2020/9/16, 2020/9/10
# @Author : Yupeng Hou, Yushuo Chen
# @Email  : houyupeng@ruc.edu.cn, chenyushuo@ruc.edu.cn

import traceback
from run_test import whole_process
from time import time


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
        'metrics': ['Recall', 'MRR', 'NDCG'],
        'topk': [5, 10, 20],
    },
    'Test Real Time Full Sort': {
        'model': 'BPRMF',
        'dataset': 'ml-100k',
        'epochs': 1,
        'valid_metric': 'Recall@10',
        'metrics': ['Recall'],
        'topk': [10],
        'eval_setting': 'RO_RS, full',
        'real_time_process': True
    },
    'Test Pre Full Sort': {
        'model': 'BPRMF',
        'dataset': 'ml-100k',
        'epochs': 1,
        'valid_metric': 'Recall@10',
        'metrics': ['Recall'],
        'topk': [10],
        'eval_setting': 'RO_RS, full',
        'real_time_process': False
    },
    'Test Real Time Neg Sample By': {
        'model': 'BPRMF',
        'dataset': 'ml-100k',
        'epochs': 1,
        'valid_metric': 'Recall@10',
        'metrics': ['Recall'],
        'topk': [10],
        'eval_setting': 'RO_RS, uni100',
        'real_time_process': True
    },
    'Test Pre Neg Sample By': {
        'model': 'BPRMF',
        'dataset': 'ml-100k',
        'epochs': 1,
        'valid_metric': 'Recall@10',
        'metrics': ['Recall'],
        'topk': [10],
        'eval_setting': 'RO_RS, uni100',
        'real_time_process': False
    },
    'Test Leave One Out': {
        'model': 'BPRMF',
        'dataset': 'ml-100k',
        'epochs': 1,
        'valid_metric': 'Recall@10',
        'metrics': ['Recall'],
        'topk': [10],
        'eval_setting': 'RO_LS, full',
        'leave_one_num': 2,
        'real_time_process': True
    },
    'Test BPRMF': {
        'model': 'BPRMF',
        'dataset': 'ml-100k',
        'epochs': 1,
        'valid_metric': 'Recall@10',
        'metrics': ['Recall'],
        'topk': [10]
    },
    'Test NeuMF': {
        'model': 'NeuMF',
        'dataset': 'ml-100k',
        'epochs': 1,
        'valid_metric': 'Recall@10',
        'metrics': ['Recall'],
        'topk': [10]
    },
    'Test DMF': {
        'model': 'DMF',
        'dataset': 'ml-100k',
        'epochs': 1,
        'valid_metric': 'Recall@10',
        'metrics': ['Recall'],
        'topk': [10]
    },
    'Test NAIS': {
        'model': 'NAIS',
        'dataset': 'ml-100k',
        'epochs': 1,
        'eval_setting': 'RO_LS, uni100',
        'valid_metric': 'Recall@10',
        'leave_one_num': 2,
        'metrics': ["Recall"],
        'topk': [10]
    },
    'Test GCMC': {
        'model': 'GCMC',
        'dataset': 'ml-100k',
        'epochs': 1,
        'valid_metric': 'Recall@10',
        'metrics': ['Recall'],
        'topk': [10]
    },
    'Test NGCF': {
        'model': 'NGCF',
        'dataset': 'ml-100k',
        'epochs': 1,
        'valid_metric': 'Recall@10',
        'metrics': ['Recall'],
        'topk': [10]
    },
    'Test LightGCN': {
        'model': 'LightGCN',
        'dataset': 'ml-100k',
        'epochs': 1,
        'valid_metric': 'Recall@10',
        'metrics': ['Recall'],
        'topk': [10]
    },
    'Test DGCF': {
        'model': 'DGCF',
        'dataset': 'ml-100k',
        'epochs': 1,
        'valid_metric': 'Recall@10',
        'metrics': ['Recall'],
        'topk': [10]
    },
    'Test POP': {
        'model': 'Pop',
        'dataset': 'ml-100k',
        'epochs': 1,
        'valid_metric': 'Recall@10',
        'metrics': ['Recall'],
        'topk': [10]
    },
    'Test ItemKNN': {
        'model': 'ItemKNN',
        'dataset': 'ml-100k',
        'epochs': 1,
        'valid_metric': 'Recall@10',
        'metrics': ['Recall'],
        'topk': [10]
    },
    'Test FM': {
        'model': 'FM',
        'dataset': 'ml-100k',
        'lowest_val': None,
        'threshold': {'rating': 3},
        'group_by_user': False,
        'epochs': 1,
        'training_neg_sample_num': 0,
        'eval_setting': 'RO_RS',
        'valid_metric': 'AUC',
        'metrics': ['AUC']
    },
    'Test Criteo': {
        'model': 'FM',
        'dataset': 'criteo',
        'normalize_all': True,
        'group_by_user': False,
        'epochs': 1,
        'training_neg_sample_num': 0,
        'eval_setting': 'RO_RS',
        'valid_metric': 'AUC',
        'metrics': ['AUC']
    },
    'Test GRU4Rec': {
        'model': 'GRU4Rec',
        'dataset': 'ml-100k',
        'epochs': 1,
        'training_neg_sample_num': 0,
        'eval_setting': 'TO_LS, full',
        'split_ratio': None,
        'leave_one_num': 2,
        'real_time_process': True,
        'NEG_PREFIX': None,
        'LABEL_FIELD': None,
        'TIME_FIELD': 'timestamp',
        'load_col': {'inter': ['user_id', 'item_id', 'rating', 'timestamp']},
        'min_user_inter_num': 5
    },
    'Test FPMC': {
        'model': 'FPMC',
        'dataset': 'ml-100k',
        'epochs': 1,
        'training_neg_sample_num': 1,
        'eval_setting': 'TO_LS, uni100',
        'split_ratio': None,
        'leave_one_num': 2,
        'real_time_process': True,
        'TIME_FIELD': 'timestamp',
        'load_col': {'inter': ['user_id', 'item_id', 'rating', 'timestamp']},
        'min_user_inter_num': 5
    },
    'Test DIN': {
        'model': 'DIN',
        'dataset': 'ml-100k',
        'epochs': 1,
        'training_neg_sample_num': 1,
        'eval_setting': 'TO_LS, uni100',
        'split_ratio': None,
        'leave_one_num': 2,
        'real_time_process': True,
        'TIME_FIELD': 'timestamp',
        'LABEL_FIELD': 'label',
        'load_col': {'inter': ['user_id', 'item_id', 'rating', 'timestamp'],
                     'user': ['user_id', 'age', 'gender', 'occupation'],
                     'item': ['item_id', 'release_year']},
        'threshold': {'rating': 4},
        'valid_metric': 'AUC',
        'metrics': ['AUC'],
        'eval_batch_size': 10000
    },
    'Test CKE': {
        'model': 'CKE',
        'dataset': 'kgdata_example',
        'epochs': 1,
        'train_kg_step': 0,
        'valid_metric': 'Recall@10',
        'metrics': ['Recall'],
        'topk': [10]
    },
    'Test KTUP': {
        'model': 'KTUP',
        'dataset': 'kgdata_example',
        'epochs': 1,
        'train_kg_step': 1,
        'valid_metric': 'Recall@10',
        'metrics': ['Recall'],
        'topk': [10]
    },
    'Test CFKG': {
        'model': 'CFKG',
        'dataset': 'kgdata_example',
        'epochs': 1,
        'train_kg_step': 0,
        'valid_metric': 'Recall@10',
        'metrics': ['Recall'],
        'topk': [10]
    },
    'Test Caser': {
        'model': 'Caser',
        'dataset': 'ml-100k',
        'epochs': 1,
        'training_neg_sample_num': 0,
        'eval_setting': 'TO_LS, full',
        'split_ratio': None,
        'leave_one_num': 2,
        'real_time_process': True,
        'NEG_PREFIX': None,
        'LABEL_FIELD': None,
        'TIME_FIELD': 'timestamp',
        'load_col': {'inter': ['user_id', 'item_id', 'rating', 'timestamp']},
        'min_user_inter_num': 5
    },

}


def run_test_examples():

    test_start_time = time()
    success_examples, fail_examples = [], []
    n_examples = len(test_examples.keys())
    for idx, example in enumerate(test_examples.keys()):
        print('\n\n Begin to run %d / %d example: %s \n\n' % (idx + 1, n_examples, example))
        try:
            whole_process(config_file='properties/overall.config', config_dict=test_examples[example], saved=False)
            print('\n\n Running %d / %d example successfully: %s \n\n' % (idx + 1, n_examples, example))
            success_examples.append(example)
        except Exception:
            print(traceback.format_exc())
            fail_examples.append(example)
    test_end_time = time()
    print('total test time: ', test_end_time - test_start_time)
    print('success examples: ', success_examples)
    print('fail examples: ', fail_examples)
    print('\n')


if __name__ == '__main__':
    run_test_examples()
