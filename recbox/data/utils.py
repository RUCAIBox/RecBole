# @Time   : 2020/7/21
# @Author : Yupeng Hou
# @Email  : houyupeng@ruc.edu.cn

# UPDATE:
# @Time   : 2020/8/25, 2020/8/14
# @Author : Yupeng Hou, Yushuo Chen
# @Email  : houyupeng@ruc.edu.cn, chenyushuo@ruc.edu.cn

import copy
import os
from logging import getLogger

from ..config import EvalSetting
from ..sampler import KGSampler, Sampler
from ..utils import EvaluatorType, InputType, ModelType
from .dataloader import *


def data_preparation(config, dataset, save=False):
    model_type = config['MODEL_TYPE']

    es_str = [_.strip() for _ in config['eval_setting'].split(',')]
    es = EvalSetting(config)

    kwargs = {}
    if 'RS' in es_str[0]:
        kwargs['ratios'] = config['split_ratio']
    if 'LS' in es_str[0]:
        kwargs['leave_one_num'] = config['leave_one_num']
    kwargs['group_by_user'] = config['group_by_user']
    getattr(es, es_str[0])(**kwargs)

    if es.split_args['strategy'] != 'loo' and model_type == ModelType.SEQUENTIAL:
        raise ValueError('Sequential models require "loo" split strategy.')

    builded_datasets = dataset.build(es, model_type)
    train_dataset, valid_dataset, test_dataset = builded_datasets
    phases = ['train', 'valid', 'test']

    if save:
        save_datasets(config['checkpoint_dir'], name=phases, dataset=builded_datasets)

    kwargs = {}
    # TODO 为什么这里type不包含context？
    if model.type in [ModelType.GENERAL, ModelType.KNOWLEDGE]:
        es.neg_sample_by(1, real_time=True)
        sampler = Sampler(config, phases, builded_datasets, es.neg_sample_args['distribution'])
        # TODO 如果model.type是kg, 可能还要设置一个kg的sampler
        kwargs['sampler'] = sampler
        kwargs['phase'] = 'train'
        kwargs['neg_sample_args'] = copy.deepcopy(es.neg_sample_args)
        if model.type == ModelType.KNOWLEDGE:
            kg_sampler = KGSampler(config, phases, builded_datasets, es.neg_sample_args['distribution'])
            kwargs['kg_sampler'] = kg_sampler
    train_data = dataloader_construct(
        name='train',
        config=config,
        eval_setting=es,
        dataset=train_dataset,
        model_type=config['MODEL_TYPE'],
        dl_format=config['MODEL_INPUT_TYPE'],
        batch_size=config['train_batch_size'],
        shuffle=True,
        **kwargs
    )

    if model.type in [ModelType.GENERAL, ModelType.KNOWLEDGE]:
        getattr(es, es_str[1])(real_time=config['real_time_neg_sampling'])
        kwargs['phase'] = ['valid', 'test']
        kwargs['neg_sample_args'] = copy.deepcopy(es.neg_sample_args)
    valid_data, test_data = dataloader_construct(
        name='evaluation',
        config=config,
        eval_setting=es,
        dataset=[valid_dataset, test_dataset],
        model_type=config['MODEL_TYPE'],
        batch_size=config['eval_batch_size'],
        **kwargs
    )

    return train_data, valid_data, test_data


def dataloader_construct(name, config, eval_setting, dataset,
                         model_type=ModelType.GENERAL, dl_format=InputType.POINTWISE,
                         batch_size=1, shuffle=False, **kwargs):
    if not isinstance(dataset, list):
        dataset = [dataset]

    if not isinstance(batch_size, list):
        batch_size = [batch_size] * len(dataset)

    if len(dataset) != len(batch_size):
        raise ValueError('dataset {} and batch_size {} should have the same length'.format(dataset, batch_size))

    kwargs_list = [{} for i in range(len(dataset))]
    for key, value in kwargs.items():
        key = [key] * len(dataset)
        if not isinstance(value, list):
            value = [value] * len(dataset)
        if len(dataset) != len(value):
            raise ValueError('dataset {} and {} {} should have the same length'.format(dataset, key, value))
        for kw, k, w in zip(kwargs_list, key, value):
            kw[k] = w

    logger = getLogger()
    logger.info('Build [{}] DataLoader for [{}] with format [{}]'.format(model_type, name, dl_format))
    logger.info(eval_setting)
    logger.info('batch_size = [{}], shuffle = [{}]\n'.format(batch_size, shuffle))

    DataLoader = get_data_loader(name, config, eval_setting, model_type)

    ret = [
        DataLoader(
            config=config,
            dataset=ds,
            batch_size=bs,
            dl_format=dl_format,
            shuffle=shuffle,
            **kw
        ) for ds, bs, kw in zip(dataset, batch_size, kwargs_list)
    ]

    if len(ret) == 1:
        return ret[0]
    else:
        return ret


def save_datasets(save_path, name, dataset):
    if (not isinstance(name, list)) and (not isinstance(dataset, list)):
        name = [name]
        dataset = [dataset]
    if len(name) != len(dataset):
        raise ValueError('len of name {} should equal to len of dataset'.format(name, dataset))
    for i, d in enumerate(dataset):
        cur_path = os.path.join(save_path, name[i])
        if not os.path.isdir(cur_path):
            os.makedirs(cur_path)
        d.save(cur_path)


def get_data_loader(name, config, eval_setting, model_type):
    if model_type == ModelType.GENERAL:
        neg_sample_strategy = eval_setting.neg_sample_args['strategy']
        if neg_sample_strategy == 'none':
            return GeneralDataLoader
        elif neg_sample_strategy == 'by':
            if name == 'train' or config['eval_type'] == EvaluatorType.INDIVIDUAL:
                return GeneralIndividualDataLoader
            else:
                return GeneralGroupedDataLoader
        elif neg_sample_strategy == 'full':
            return GeneralFullDataLoader
    elif model_type == ModelType.CONTEXT:
        neg_sample_strategy = eval_setting.neg_sample_args['strategy']
        if neg_sample_strategy == 'none':
            return ContextDataLoader
        elif neg_sample_strategy == 'by':
            if name == 'train' or config['eval_type'] == EvaluatorType.INDIVIDUAL:
                return ContextIndividualDataLoader
            else:
                return ContextGroupedDataLoader
        elif neg_sample_strategy == 'full':
            raise NotImplementedError('context model\'s full_sort has not been implemented')
    elif model_type == ModelType.SEQUENTIAL:
        if name == 'train' or config['eval_type'] == EvaluatorType.INDIVIDUAL:
            return SequentialDataLoader
        else:
            return SequentialFullDataLoader
    elif model_type == ModelType.KNOWLEDGE:
        neg_sample_strategy = eval_setting.neg_sample_args['strategy']
        if neg_sample_strategy == 'by':
            if name == 'train':
                return KnowledgeBasedDataLoader
            elif config['eval_type'] == EvaluatorType.INDIVIDUAL:
                return GeneralIndividualDataLoader
            else:
                return GeneralGroupedDataLoader
        elif neg_sample_strategy == 'full':
            return GeneralFullDataLoader
        elif neg_sample_strategy == 'none':
            # return GeneralDataLoader
            # TODO 训练也可以为none? 看general的逻辑似乎是都可以为None
            raise NotImplementedError('The use of external negative sampling for knowledge model has not been implemented')
    else:
        raise NotImplementedError('model_type [{}] has not been implemented'.format(model_type))
