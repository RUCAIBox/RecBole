# @Time   : 2020/7/23
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn

# UPDATE:
# @Time   : 2020/9/16, 2020/9/10
# @Author : Yupeng Hou, Yushuo Chen
# @Email  : houyupeng@ruc.edu.cn, chenyushuo@ruc.edu.cn

import traceback
from time import time
from recbole.quick_start import run_recbole


closed_examples = ['Test GRU4RecKG', 'Test S3Rec', 'Test DIN']

test_examples = {
    'Test Eval Metric': {
        'model': 'BPR',
        'dataset': 'ml-100k',
        'epochs': 1,
        'valid_metric': 'Recall@10',
        'eval_setting': 'RO_RS, full',
        'training_neg_sample_num': 1,
        'metrics': ['Precision', 'Hit', 'Recall', 'MRR', 'NDCG'],
        'topk': [5, 10, 20],
    },
    'Test Real Time Full Sort': {
        'model': 'BPR',
        'dataset': 'ml-100k',
        'epochs': 1,
        'valid_metric': 'Recall@10',
        'metrics': ['Recall'],
        'topk': [10],
        'eval_setting': 'RO_RS, full',
        'real_time_process': True
    },
    'Test Pre Full Sort': {
        'model': 'BPR',
        'dataset': 'ml-100k',
        'epochs': 1,
        'valid_metric': 'Recall@10',
        'metrics': ['Recall'],
        'topk': [10],
        'eval_setting': 'RO_RS, full',
        'real_time_process': False
    },
    'Test Real Time Neg Sample By': {
        'model': 'BPR',
        'dataset': 'ml-100k',
        'epochs': 1,
        'valid_metric': 'Recall@10',
        'metrics': ['Recall'],
        'topk': [10],
        'eval_setting': 'RO_RS, uni100',
        'real_time_process': True
    },
    'Test Pre Neg Sample By': {
        'model': 'BPR',
        'dataset': 'ml-100k',
        'epochs': 1,
        'valid_metric': 'Recall@10',
        'metrics': ['Recall'],
        'topk': [10],
        'eval_setting': 'RO_RS, uni100',
        'real_time_process': False
    },
    'Test Leave One Out': {
        'model': 'BPR',
        'dataset': 'ml-100k',
        'epochs': 1,
        'valid_metric': 'Recall@10',
        'metrics': ['Recall'],
        'topk': [10],
        'eval_setting': 'RO_LS, full',
        'leave_one_num': 2,
        'real_time_process': True
    },

    # General Recommendation
    'Test BPR': {
        'model': 'BPR',
        'dataset': 'ml-100k',
    },
    'Test NeuMF': {
        'model': 'NeuMF',
        'dataset': 'ml-100k',
    },
    'Test DMF': {
        'model': 'DMF',
        'dataset': 'ml-100k',
    },
    'Test NAIS': {
        'model': 'NAIS',
        'dataset': 'ml-100k',
    },
    'Test GCMC': {
        'model': 'GCMC',
        'dataset': 'ml-100k',
    },
    'Test NGCF': {
        'model': 'NGCF',
        'dataset': 'ml-100k',
    },
    'Test LightGCN': {
        'model': 'LightGCN',
        'dataset': 'ml-100k',
    },
    'Test DGCF': {
        'model': 'DGCF',
        'dataset': 'ml-100k',
    },
    'Test FISM': {
        'model': 'FISM',
        'dataset': 'ml-100k'
    },
    'Test SpectralCF': {
        'model': 'SpectralCF',
        'dataset': 'ml-100k'
    },
    'Test POP': {
        'model': 'Pop',
        'dataset': 'ml-100k',
    },
    'Test ItemKNN': {
        'model': 'ItemKNN',
        'dataset': 'ml-100k',
    },
    'Test ConvNCF': {
        'model': 'ConvNCF',
        'dataset': 'ml-100k',
    },

    # Context-aware Recommendation
    'Test FM': {
        'model': 'FM',
        'dataset': 'ml-100k',
    },
    'Test DCN': {
        'model': 'DCN',
        'dataset': 'ml-100k',
    },
    'Test xDeepFM': {
        'model': 'xDeepFM',
        'dataset': 'ml-100k',
    },
    'Test AFM': {
        'model': 'AFM',
        'dataset': 'ml-100k',
    },
    'Test AUTOINT': {
        'model': 'AutoInt',
        'dataset': 'ml-100k',
    },
    'Test DeepFM': {
        'model': 'DeepFM',
        'dataset': 'ml-100k',
    },
    'Test DSSM': {
        'model': 'DSSM',
        'dataset': 'ml-100k',
    },
    'Test FFM': {
        'model': 'FFM',
        'dataset': 'ml-100k',
    },
    'Test FNN': {
        'model': 'FNN',
        'dataset': 'ml-100k',
    },
    'Test FwFM': {
        'model': 'FwFM',
        'dataset': 'ml-100k',
    },
    'Test LR': {
        'model': 'LR',
        'dataset': 'ml-100k',
    },
    'Test NFM': {
        'model': 'NFM',
        'dataset': 'ml-100k',
    },
    'Test PNN': {
        'model': 'PNN',
        'dataset': 'ml-100k',
    },
    'Test WideDeep': {
        'model': 'WideDeep',
        'dataset': 'ml-100k',
    },

    # Sequential Recommendation
    'Test GRU4Rec': {
        'model': 'GRU4Rec',
        'dataset': 'ml-100k',
    },
    'Test FPMC': {
        'model': 'FPMC',
        'dataset': 'ml-100k',
    },
    'Test Caser': {
        'model': 'Caser',
        'dataset': 'ml-100k',
        'reproducibility': False,
    },
    'Test TransRec': {
        'model': 'TransRec',
        'dataset': 'ml-100k',
    },
    'Test SASRec': {
        'model': 'SASRec',
        'dataset': 'ml-100k',
    },
    'Test BERT4Rec': {
        'model': 'BERT4Rec',
        'dataset': 'ml-100k',
    },
    'Test STAMP': {
        'model': 'STAMP',
        'dataset': 'ml-100k',
    },
    'Test NARM': {
        'model': 'NARM',
        'dataset': 'ml-100k',
    },
    'Test NextItNet': {
        'model': 'NextItNet',
        'dataset': 'ml-100k',
        'reproducibility': False,
    },
    'Test SRGNN': {
        'model': 'SRGNN',
        'dataset': 'ml-100k',
        'MAX_ITEM_LIST_LENGTH': 3,
    },
    'Test GCSAN': {
        'model': 'GCSAN',
        'dataset': 'ml-100k',
        'MAX_ITEM_LIST_LENGTH': 3,
    },
    'Test GRU4RecF': {
        'model': 'GRU4RecF',
        'dataset': 'ml-100k',
    },
    'Test SASRecF': {
        'model': 'SASRecF',
        'dataset': 'ml-100k',
    },
    'Test FDSA': {
        'model': 'FDSA',
        'dataset': 'ml-100k',
    },
    'Test S3Rec': {

    },
    'Test GRU4RecKG': {
        'model': 'GRU4RecKG',
        'dataset': 'ml-1m',
        'TIME_FIELD': 'timestamp',
        'HEAD_ENTITY_ID_FIELD': 'head_id',
        'TAIL_ENTITY_ID_FIELD': 'tail_id',
        'RELATION_ID_FIELD': 'relation_id',
        'ENTITY_ID_FIELD': 'entity_id',
        'MAX_ITEM_LIST_LENGTH': 50,
        'LIST_SUFFIX': '_list',
        'ITEM_LIST_LENGTH_FIELD': 'item_length',
        'load_col': {
            'inter': ['user_id', 'item_id', 'rating', 'timestamp'],
            'feature': ['ent_id', 'ent_feature']
        },
        'additional_feat_suffix': ['feature'],
        'fields_in_same_space': [['entity_id', 'ent_id']],
        'preload_weight': {
            'ent_id': 'ent_feature'
        }
    },
    'Test DIN': {
        'model': 'DIN',
        'dataset': 'ml-100k',
        'training_neg_sample_num': 1,
        'eval_setting': 'TO_LS, uni100',
        'load_col': {'inter': ['user_id', 'item_id', 'rating', 'timestamp'],
                     'user': ['user_id', 'age', 'gender', 'occupation'],
                     'item': ['item_id', 'release_year']},
        'threshold': {'rating': 4},
        'valid_metric': 'AUC',
        'metrics': ['AUC'],
        'eval_batch_size': 10000,
    },

    # Knowledge-based Recommendation
    'Test CKE': {
        'model': 'CKE',
        'dataset': 'ml-100k',
    },
    'Test KTUP': {
        'model': 'KTUP',
        'dataset': 'ml-100k',
        'train_rec_step': 1,
        'train_kg_step': 1,
        'epochs': 2,
    },
    'Test CFKG': {
        'model': 'CFKG',
        'dataset': 'ml-100k',
    },
    'Test KGAT': {
        'model': 'KGAT',
        'dataset': 'ml-100k',
    },
    'Test RippleNet': {
        'model': 'RippleNet',
        'dataset': 'ml-100k',
    },
    'Test MKR': {
        'model': 'MKR',
        'dataset': 'ml-100k',
    },
    'Test KGCN': {
        'model': 'KGCN',
        'dataset': 'ml-100k',
    },
    'Test KGNNLS': {
        'model': 'KGNNLS',
        'dataset': 'ml-100k',
    },
}


def run_test_examples():

    test_start_time = time()
    success_examples, fail_examples = [], []
    n_examples = len(test_examples.keys())
    for idx, example in enumerate(test_examples.keys()):
        if example in closed_examples:
            continue
        print('\n\n Begin to run %d / %d example: %s \n\n' % (idx + 1, n_examples, example))
        try:
            config_dict = test_examples[example]
            if 'epochs' not in config_dict:
                config_dict['epochs'] = 1
            run_recbole(config_dict=config_dict, saved=False)
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
