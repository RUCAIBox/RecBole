# @Time   : 2020/7/21
# @Author : Yupeng Hou
# @Email  : houyupeng@ruc.edu.cn

# UPDATE:
# @Time   : 2021/7/9, 2020/9/17, 2020/8/31, 2021/2/20, 2021/3/1
# @Author : Yupeng Hou, Yushuo Chen, Kaiyuan Li, Haoran Cheng, Jiawei Guan
# @Email  : houyupeng@ruc.edu.cn, chenyushuo@ruc.edu.cn, tsotfsk@outlook.com, chenghaoran29@foxmail.com, guanjw@ruc.edu.cn

"""
recbole.data.utils
########################
"""

import copy
import importlib
import os
import pickle

from recbole.data.dataloader import *
from recbole.sampler import KGSampler, Sampler, RepeatableSampler
from recbole.utils import ModelType, ensure_dir, get_local_time, set_color


def create_dataset(config):
    """Create dataset according to :attr:`config['model']` and :attr:`config['MODEL_TYPE']`.

    Args:
        config (Config): An instance object of Config, used to record parameter information.

    Returns:
        Dataset: Constructed dataset.
    """
    dataset_module = importlib.import_module('recbole.data.dataset')
    if hasattr(dataset_module, config['model'] + 'Dataset'):
        return getattr(dataset_module, config['model'] + 'Dataset')(config)
    else:
        model_type = config['MODEL_TYPE']
        if model_type == ModelType.SEQUENTIAL:
            from .dataset import SequentialDataset
            return SequentialDataset(config)
        elif model_type == ModelType.KNOWLEDGE:
            from .dataset import KnowledgeBasedDataset
            return KnowledgeBasedDataset(config)
        elif model_type == ModelType.DECISIONTREE:
            from .dataset import DecisionTreeDataset
            return DecisionTreeDataset(config)
        else:
            from .dataset import Dataset
            return Dataset(config)


def save_split_dataloaders(config, dataloaders):
    """Save split dataloaders.

    Args:
        config (Config): An instance object of Config, used to record parameter information.
        dataloaders (tuple of AbstractDataLoader): The split dataloaders.
    """
    save_path = config['checkpoint_dir']
    saved_dataloaders_file = f'{config["dataset"]}-for-{config["model"]}-dataloader.pth'
    file_path = os.path.join(save_path, saved_dataloaders_file)
    logger = getLogger()
    logger.info(set_color('Saved split dataloaders', 'blue') + f': {file_path}')
    with open(file_path, 'wb') as f:
        pickle.dump(dataloaders, f)


def load_split_dataloaders(saved_dataloaders_file):
    """Load split dataloaders.

    Args:
        saved_dataloaders_file (str): The path of split dataloaders.

    Returns:
        dataloaders (tuple of AbstractDataLoader): The split dataloaders.
    """
    with open(saved_dataloaders_file, 'rb') as f:
        dataloaders = pickle.load(f)
    return dataloaders


def data_preparation(config, dataset, save=False):
    """Split the dataset by :attr:`config['eval_args']` and create training, validation and test dataloader.

    Args:
        config (Config): An instance object of Config, used to record parameter information.
        dataset (Dataset): An instance object of Dataset, which contains all interaction records.
        save (bool, optional): If ``True``, it will call :func:`save_datasets` to save split dataset.
            Defaults to ``False``.

    Returns:
        tuple:
            - train_data (AbstractDataLoader): The dataloader for training.
            - valid_data (AbstractDataLoader): The dataloader for validation.
            - test_data (AbstractDataLoader): The dataloader for testing.
    """
    model_type = config['MODEL_TYPE']
    built_datasets = dataset.build()
    logger = getLogger()

    train_dataset, valid_dataset, test_dataset = built_datasets
    train_sampler, valid_sampler, test_sampler = create_samplers(config, dataset, built_datasets)

    if model_type != ModelType.KNOWLEDGE:
        train_data = get_dataloader(config, 'train')(config, train_dataset, train_sampler, shuffle=True)
    else:
        kg_sampler = KGSampler(dataset, config['train_neg_sample_args']['distribution'])
        train_data = get_dataloader(config, 'train')(config, train_dataset, train_sampler, kg_sampler, shuffle=True)

    valid_data = get_dataloader(config, 'evaluation')(config, valid_dataset, valid_sampler, shuffle=False)
    test_data = get_dataloader(config, 'evaluation')(config, test_dataset, test_sampler, shuffle=False)
    logger.info(
        set_color('[Training]: ', 'pink') + set_color('train_batch_size', 'cyan') + ' = ' +
        set_color(f'[{config["train_batch_size"]}]', 'yellow') + set_color(' negative sampling', 'cyan') + ': ' +
        set_color(f'[{config["neg_sampling"]}]', 'yellow')
    )
    logger.info(
        set_color('[Evaluation]: ', 'pink') + set_color('eval_batch_size', 'cyan') + ' = ' +
        set_color(f'[{config["eval_batch_size"]}]', 'yellow') + set_color(' eval_args', 'cyan') + ': ' +
        set_color(f'[{config["eval_args"]}]', 'yellow')
    )
    if save:
        save_split_dataloaders(config, dataloaders=(train_data, valid_data, test_data))

    return train_data, valid_data, test_data


def get_dataloader(config, phase):
    """Return a dataloader class according to :attr:`config` and :attr:`phase`.

    Args:
        config (Config): An instance object of Config, used to record parameter information.
        phase (str): The stage of dataloader. It can only take two values: 'train' or 'evaluation'.

    Returns:
        type: The dataloader class that meets the requirements in :attr:`config` and :attr:`phase`.
    """
    register_table = {
        "MultiDAE": _get_AE_dataloader,
        "MultiVAE": _get_AE_dataloader,
        'MacridVAE': _get_AE_dataloader,
        'CDAE': _get_AE_dataloader,
        'ENMF': _get_AE_dataloader,
        'RaCT': _get_AE_dataloader,
        'RecVAE': _get_AE_dataloader,
    }

    if config['model'] in register_table:
        return register_table[config['model']](config, phase)

    model_type = config['MODEL_TYPE']
    if phase == 'train':
        if model_type != ModelType.KNOWLEDGE:
            return TrainDataLoader
        else:
            return KnowledgeBasedDataLoader
    else:
        eval_strategy = config['eval_neg_sample_args']['strategy']
        if eval_strategy in {'none', 'by'}:
            return NegSampleEvalDataLoader
        elif eval_strategy == 'full':
            return FullSortEvalDataLoader


def _get_AE_dataloader(config, phase):
    """Customized function for VAE models to get correct dataloader class.

    Args:
        config (Config): An instance object of Config, used to record parameter information.
        phase (str): The stage of dataloader. It can only take two values: 'train' or 'evaluation'.

    Returns:
        type: The dataloader class that meets the requirements in :attr:`config` and :attr:`phase`.
    """
    if phase == 'train':
        return UserDataLoader
    else:
        eval_strategy = config['eval_neg_sample_args']['strategy']
        if eval_strategy in {'none', 'by'}:
            return NegSampleEvalDataLoader
        elif eval_strategy == 'full':
            return FullSortEvalDataLoader


def create_samplers(config, dataset, built_datasets):
    """Create sampler for training, validation and testing.

    Args:
        config (Config): An instance object of Config, used to record parameter information.
        dataset (Dataset): An instance object of Dataset, which contains all interaction records.
        built_datasets (list of Dataset): A list of split Dataset, which contains dataset for
            training, validation and testing.

    Returns:
        tuple:
            - train_sampler (AbstractSampler): The sampler for training.
            - valid_sampler (AbstractSampler): The sampler for validation.
            - test_sampler (AbstractSampler): The sampler for testing.
    """
    phases = ['train', 'valid', 'test']
    train_neg_sample_args = config['train_neg_sample_args']
    eval_neg_sample_args = config['eval_neg_sample_args']
    sampler = None
    train_sampler, valid_sampler, test_sampler = None, None, None

    if train_neg_sample_args['strategy'] != 'none':
        if not config['repeatable']:
            sampler = Sampler(phases, built_datasets, train_neg_sample_args['distribution'])
        else:
            sampler = RepeatableSampler(phases, dataset, train_neg_sample_args['distribution'])
        train_sampler = sampler.set_phase('train')

    if eval_neg_sample_args['strategy'] != 'none':
        if sampler is None:
            if not config['repeatable']:
                sampler = Sampler(phases, built_datasets, eval_neg_sample_args['distribution'])
            else:
                sampler = RepeatableSampler(phases, dataset, eval_neg_sample_args['distribution'])
        else:
            sampler.set_distribution(eval_neg_sample_args['distribution'])
        valid_sampler = sampler.set_phase('valid')
        test_sampler = sampler.set_phase('test')

    return train_sampler, valid_sampler, test_sampler
