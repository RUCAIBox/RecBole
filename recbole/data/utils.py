# @Time   : 2020/7/21
# @Author : Yupeng Hou
# @Email  : houyupeng@ruc.edu.cn

# UPDATE:
# @Time   : 2020/10/19, 2020/9/17, 2020/8/31, 2021/2/20, 2021/3/1
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
from recbole.utils import ModelType, ensure_dir, get_local_time
from recbole.utils.utils import set_color


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
        elif model_type == ModelType.SOCIAL:
            from .dataset import SocialDataset
            return SocialDataset(config)
        elif model_type == ModelType.DECISIONTREE:
            from .dataset import DecisionTreeDataset
            return DecisionTreeDataset(config)
        else:
            from .dataset import Dataset
            return Dataset(config)


def data_preparation(config, dataset, save=False):
    """Split the dataset by :attr:`config['eval_setting']` and call :func:`dataloader_construct` to create
    corresponding dataloader.

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
    if model_type != ModelType.SEQUENTIAL and model_type != ModelType.KNOWLEDGE:
        return general_data_preparation(config, dataset, save)

    built_datasets = dataset.build()
    
    train_dataset, valid_dataset, test_dataset = built_datasets
    phases = ['train', 'valid', 'test']
    sampler = None
    logger = getLogger()
    train_neg_sample_args = config['train_neg_sample_args']
    eval_neg_sample_args = config['eval_neg_sample_args']

    # Training
    train_kwargs = {
        'config': config,
        'dataset': train_dataset,
        'batch_size': config['train_batch_size'],
        'dl_format': config['MODEL_INPUT_TYPE'],
        'shuffle': True,
    }
    if train_neg_sample_args['strategy'] != 'none':
        if dataset.label_field in dataset.inter_feat:
            raise ValueError(
                f'`training_neg_sample_num` should be 0 '
                f'if inter_feat have label_field [{dataset.label_field}].'
            )
        if model_type != ModelType.SEQUENTIAL:
            sampler = Sampler(phases, built_datasets, train_neg_sample_args['distribution'])
        else:
            sampler = RepeatableSampler(phases, dataset, train_neg_sample_args['distribution'])
        train_kwargs['sampler'] = sampler.set_phase('train')
        train_kwargs['neg_sample_args'] = train_neg_sample_args
        if model_type == ModelType.KNOWLEDGE:
            kg_sampler = KGSampler(dataset, train_neg_sample_args['distribution'])
            train_kwargs['kg_sampler'] = kg_sampler

    dataloader = get_data_loader('train', config, train_neg_sample_args)
    logger.info(
        set_color('Build', 'pink') + set_color(f' [{dataloader.__name__}]', 'yellow') + ' for ' +
        set_color('[train]', 'yellow') + ' with format ' + set_color(f'[{train_kwargs["dl_format"]}]', 'yellow')
    )
    if train_neg_sample_args['strategy'] != 'none':
        logger.info(
            set_color('[train]', 'pink') + set_color(' Negative Sampling', 'blue') + f': {train_neg_sample_args}'
        )
    else:
        logger.info(set_color('[train]', 'pink') + set_color(' No Negative Sampling', 'yellow'))
    logger.info(
        set_color('[train]', 'pink') + set_color(' batch_size', 'cyan') + ' = ' +
        set_color(f'[{train_kwargs["batch_size"]}]', 'yellow') + ', ' + set_color('shuffle', 'cyan') + ' = ' +
        set_color(f'[{train_kwargs["shuffle"]}]\n', 'yellow')
    )
    train_data = dataloader(**train_kwargs)

    # Evaluation
    eval_kwargs = {
        'config': config,
        'batch_size': config['eval_batch_size'],
        'dl_format': InputType.POINTWISE,
        'shuffle': False,
    }
    valid_kwargs = {'dataset': valid_dataset}
    test_kwargs = {'dataset': test_dataset}
    if eval_neg_sample_args['strategy'] != 'none':
        if dataset.label_field in dataset.inter_feat:
            raise ValueError(
                f'It can not validate with `{config["eval_args"]["mode"]}` '
                f'when inter_feat have label_field [{dataset.label_field}].'
            )
        if sampler is None:
            if model_type != ModelType.SEQUENTIAL:
                sampler = Sampler(phases, built_datasets, eval_neg_sample_args['distribution'])
            else:
                sampler = RepeatableSampler(phases, dataset, eval_neg_sample_args['distribution'])
        else:
            sampler.set_distribution(eval_neg_sample_args['distribution'])
        eval_kwargs['neg_sample_args'] = eval_neg_sample_args
        valid_kwargs['sampler'] = sampler.set_phase('valid')
        test_kwargs['sampler'] = sampler.set_phase('test')
    valid_kwargs.update(eval_kwargs)
    test_kwargs.update(eval_kwargs)

    dataloader = get_data_loader('evaluation', config, eval_neg_sample_args)
    logger.info(
        set_color('Build', 'pink') + set_color(f' [{dataloader.__name__}]', 'yellow') + ' for ' +
        set_color('[evaluation]', 'yellow') + ' with format ' + set_color(f'[{eval_kwargs["dl_format"]}]', 'yellow')
    )
    logger.info(
        set_color('[evaluation]', 'pink') + set_color(' batch_size', 'cyan') + ' = ' +
        set_color(f'[{eval_kwargs["batch_size"]}]', 'yellow') + ', ' + set_color('shuffle', 'cyan') + ' = ' +
        set_color(f'[{eval_kwargs["shuffle"]}]\n', 'yellow')
    )

    valid_data = dataloader(**valid_kwargs)
    test_data = dataloader(**test_kwargs)

    if save:
        save_split_dataloaders(config, dataloaders=(train_data, valid_data, test_data))

    return train_data, valid_data, test_data


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


def get_data_loader(name, config, neg_sample_args):
    """Return a dataloader class according to :attr:`config` and :attr:`eval_setting`.

    Args:
        name (str): The stage of dataloader. It can only take two values: 'train' or 'evaluation'.
        config (Config): An instance object of Config, used to record parameter information.
        neg_sample_args (dict) : Settings of negative sampling.

    Returns:
        type: The dataloader class that meets the requirements in :attr:`config` and :attr:`eval_setting`.
    """
    register_table = {
        'DIN': _get_DIN_data_loader,
        'DIEN': _get_DIEN_data_loader,
        "MultiDAE": _get_AE_data_loader,
        "MultiVAE": _get_AE_data_loader,
        'MacridVAE': _get_AE_data_loader,
        'CDAE': _get_AE_data_loader,
        'ENMF': _get_AE_data_loader,
        'RaCT': _get_AE_data_loader,
        'RecVAE': _get_AE_data_loader
    }

    if config['model'] in register_table:
        return register_table[config['model']](name, config, neg_sample_args)

    model_type_table = {
        ModelType.GENERAL: 'General',
        ModelType.TRADITIONAL: 'General',
        ModelType.CONTEXT: 'Context',
        ModelType.SEQUENTIAL: 'Sequential',
        ModelType.DECISIONTREE: 'DecisionTree',
    }
    neg_sample_strategy_table = {
        'none': 'DataLoader',
        'by': 'NegSampleDataLoader',
        'full': 'FullDataLoader',
    }
    model_type = config['MODEL_TYPE']
    neg_sample_strategy = neg_sample_args['strategy']
    dataloader_module = importlib.import_module('recbole.data.dataloader')

    if model_type in model_type_table and neg_sample_strategy in neg_sample_strategy_table:
        dataloader_name = model_type_table[model_type] + neg_sample_strategy_table[neg_sample_strategy]
        return getattr(dataloader_module, dataloader_name)
    elif model_type == ModelType.KNOWLEDGE:
        if neg_sample_strategy == 'by':
            if name == 'train':
                return KnowledgeBasedDataLoader
            else:
                return GeneralNegSampleDataLoader
        elif neg_sample_strategy == 'full':
            return GeneralFullDataLoader
        elif neg_sample_strategy == 'none':
            raise NotImplementedError(
                'The use of external negative sampling for knowledge model has not been implemented'
            )
    else:
        raise NotImplementedError(f'Model_type [{model_type}] has not been implemented.')


def _get_DIN_data_loader(name, config, neg_sample_args):
    """Customized function for DIN to get correct dataloader class.

    Args:
        name (str): The stage of dataloader. It can only take two values: 'train' or 'evaluation'.
        config (Config): An instance object of Config, used to record parameter information.
        neg_sample_args : Settings of negative sampling.

    Returns:
        type: The dataloader class that meets the requirements in :attr:`config` and :attr:`eval_setting`.
    """
    neg_sample_strategy = neg_sample_args['strategy']
    if neg_sample_strategy == 'none':
        return SequentialDataLoader
    elif neg_sample_strategy == 'by':
        return SequentialNegSampleDataLoader
    elif neg_sample_strategy == 'full':
        return SequentialFullDataLoader


def _get_DIEN_data_loader(name, config, neg_sample_args):
    """Customized function for DIEN to get correct dataloader class.

    Args:
        name (str): The stage of dataloader. It can only take two values: 'train' or 'evaluation'.
        config (Config): An instance object of Config, used to record parameter information.
        neg_sample_args : Settings of negative sampling.

    Returns:
        type: The dataloader class that meets the requirements in :attr:`config` and :attr:`eval_setting`.
    """
    neg_sample_strategy = neg_sample_args['strategy']
    if neg_sample_strategy == 'none':
        return DIENDataLoader
    elif neg_sample_strategy == 'by':
        return DIENNegSampleDataLoader
    elif neg_sample_strategy == 'full':
        return DIENFullDataLoader


def _get_AE_data_loader(name, config, neg_sample_args):
    """Customized function for Multi-DAE and Multi-VAE to get correct dataloader class.

    Args:
        name (str): The stage of dataloader. It can only take two values: 'train' or 'evaluation'.
        config (Config): An instance object of Config, used to record parameter information.
        neg_sample_args (dict): Settings of negative sampling.

    Returns:
        type: The dataloader class that meets the requirements in :attr:`config` and :attr:`eval_setting`.
    """
    neg_sample_strategy = neg_sample_args['strategy']
    if name == "train":
        return UserDataLoader
    else:
        if neg_sample_strategy == 'none':
            return GeneralDataLoader
        elif neg_sample_strategy == 'by':
            return GeneralNegSampleDataLoader
        elif neg_sample_strategy == 'full':
            return GeneralFullDataLoader


def general_data_preparation(config, dataset, save=False):
    built_datasets = dataset.build()

    train_dataset, valid_dataset, test_dataset = built_datasets
    train_sampler, valid_sampler, test_sampler = create_samplers(config, dataset, built_datasets)

    train_data = get_dataloader(config, 'train')(config, train_dataset, train_sampler, shuffle=True)
    valid_data = get_dataloader(config, 'valid')(config, valid_dataset, valid_sampler, shuffle=False)
    test_data = get_dataloader(config, 'test')(config, test_dataset, test_sampler, shuffle=False)

    if save:
        save_split_dataloaders(config, dataloaders=(train_data, valid_data, test_data))

    return train_data, valid_data, test_data


def get_dataloader(config, phase):
    model_type = config['MODEL_TYPE']
    eval_strategy = config['eval_neg_sample_args']['strategy']
    if model_type != ModelType.KNOWLEDGE:
        if phase == 'train':
            return NewGeneralTrainDataLoader
        elif eval_strategy in {'none', 'by'}:
            return NewGeneralNegSampleEvalDataLoader
        elif eval_strategy == 'full':
            return NewGeneralFullEvalDataLoader


def create_samplers(config, dataset, built_datasets):
    model_type = config['MODEL_TYPE']
    phases = ['train', 'valid', 'test']
    train_neg_sample_args = config['train_neg_sample_args']
    eval_neg_sample_args = config['eval_neg_sample_args']
    sampler = None
    train_sampler, valid_sampler, test_sampler = None, None, None

    if train_neg_sample_args['strategy'] != 'none':
        if model_type != ModelType.SEQUENTIAL:
            sampler = Sampler(phases, built_datasets, train_neg_sample_args['distribution'])
        else:
            sampler = RepeatableSampler(phases, dataset, train_neg_sample_args['distribution'])
        train_sampler = sampler.set_phase('train')

    if eval_neg_sample_args['strategy'] != 'none':
        if sampler is None:
            if model_type != ModelType.SEQUENTIAL:
                sampler = Sampler(phases, built_datasets, eval_neg_sample_args['distribution'])
            else:
                sampler = RepeatableSampler(phases, dataset, eval_neg_sample_args['distribution'])
        else:
            sampler.set_distribution(eval_neg_sample_args['distribution'])
        valid_sampler = sampler.set_phase('valid')
        test_sampler = sampler.set_phase('test')

    return train_sampler, valid_sampler, test_sampler
