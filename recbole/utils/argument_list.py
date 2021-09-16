# @Time   : 2020/10/19
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn

# yapf: disable

general_arguments = [
    'gpu_id', 'use_gpu',
    'seed',
    'reproducibility',
    'state',
    'data_path',
    'benchmark_filename',
    'show_progress',
    'config_file',
    'save_dataset',
    'save_dataloaders',
]

training_arguments = [
    'epochs', 'train_batch_size',
    'learner', 'learning_rate',
    'training_neg_sample_num',
    'training_neg_sample_distribution',
    'eval_step', 'stopping_step',
    'checkpoint_dir',
    'clip_grad_norm',
    'loss_decimal_place',
    'weight_decay'
]

evaluation_arguments = [
    'eval_args',
    'metrics', 'topk', 'valid_metric', 'valid_metric_bigger',
    'eval_batch_size',
    'metric_decimal_place'
]

dataset_arguments = [
    'field_separator', 'seq_separator',
    'USER_ID_FIELD', 'ITEM_ID_FIELD', 'RATING_FIELD', 'TIME_FIELD',
    'seq_len',
    'LABEL_FIELD', 'threshold',
    'NEG_PREFIX',
    'ITEM_LIST_LENGTH_FIELD', 'LIST_SUFFIX', 'MAX_ITEM_LIST_LENGTH', 'POSITION_FIELD',
    'HEAD_ENTITY_ID_FIELD', 'TAIL_ENTITY_ID_FIELD', 'RELATION_ID_FIELD', 'ENTITY_ID_FIELD',
    'load_col', 'unload_col', 'unused_col', 'additional_feat_suffix',
    'filter_inter_by_user_or_item', 'rm_dup_inter',
    'val_interval', 'user_inter_num_interval', 'item_inter_num_interval',
    'alias_of_user_id', 'alias_of_item_id', 'alias_of_entity_id', 'alias_of_relation_id',
    'preload_weight',
    'normalize_field', 'normalize_all'
]
