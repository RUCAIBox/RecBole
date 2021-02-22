# @Time   : 2020/7/21
# @Author : Yupeng Hou
# @Email  : houyupeng@ruc.edu.cn

# UPDATE:
# @Time   : 2020/10/19, 2020/9/17, 2020/8/31, 2021/2/20
# @Author : Yupeng Hou, Yushuo Chen, Kaiyuan Li, Haoran Cheng
# @Email  : houyupeng@ruc.edu.cn, chenyushuo@ruc.edu.cn, tsotfsk@outlook.com, chenghaoran29@foxmail.com

"""
recbole.data.utils
########################
"""

import copy
import importlib
import os

from recbole.config import EvalSetting
from recbole.data.dataloader import *
from recbole.sampler import KGSampler, Sampler, RepeatableSampler
from recbole.utils import ModelType, ensure_dir


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

    es_str = [_.strip() for _ in config['eval_setting'].split(',')]
    es = EvalSetting(config)
    es.set_ordering_and_splitting(es_str[0])

    built_datasets = dataset.build(es)
    train_dataset, valid_dataset, test_dataset = built_datasets
    phases = ['train', 'valid', 'test']
    sampler = None

    if save:
        save_datasets(config['checkpoint_dir'], name=phases, dataset=built_datasets)

    kwargs = {}
    if config['training_neg_sample_num']:
        if dataset.label_field in dataset.inter_feat:
            raise ValueError(
                f'`training_neg_sample_num` should be 0 '
                f'if inter_feat have label_field [{dataset.label_field}].'
            )
        train_distribution = config['training_neg_sample_distribution'] or 'uniform'
        es.neg_sample_by(by=config['training_neg_sample_num'], distribution=train_distribution)
        if model_type != ModelType.SEQUENTIAL:
            sampler = Sampler(phases, built_datasets, es.neg_sample_args['distribution'])
        else:
            sampler = RepeatableSampler(phases, dataset, es.neg_sample_args['distribution'])
        kwargs['sampler'] = sampler.set_phase('train')
        kwargs['neg_sample_args'] = copy.deepcopy(es.neg_sample_args)
        if model_type == ModelType.KNOWLEDGE:
            kg_sampler = KGSampler(dataset, es.neg_sample_args['distribution'])
            kwargs['kg_sampler'] = kg_sampler
    train_data = dataloader_construct(
        name='train',
        config=config,
        eval_setting=es,
        dataset=train_dataset,
        dl_format=config['MODEL_INPUT_TYPE'],
        batch_size=config['train_batch_size'],
        shuffle=True,
        **kwargs
    )

    kwargs = {}
    if len(es_str) > 1 and getattr(es, es_str[1], None):
        if dataset.label_field in dataset.inter_feat:
            raise ValueError(
                f'It can not validate with `{es_str[1]}` '
                f'when inter_feat have label_field [{dataset.label_field}].'
            )
        getattr(es, es_str[1])()
        if sampler is None:
            if model_type != ModelType.SEQUENTIAL:
                sampler = Sampler(phases, built_datasets, es.neg_sample_args['distribution'])
            else:
                sampler = RepeatableSampler(phases, dataset, es.neg_sample_args['distribution'])
        sampler.set_distribution(es.neg_sample_args['distribution'])
        kwargs['sampler'] = [sampler.set_phase('valid'), sampler.set_phase('test')]
        kwargs['neg_sample_args'] = copy.deepcopy(es.neg_sample_args)
    valid_data, test_data = dataloader_construct(
        name='evaluation',
        config=config,
        eval_setting=es,
        dataset=[valid_dataset, test_dataset],
        batch_size=config['eval_batch_size'],
        **kwargs
    )

    return train_data, valid_data, test_data


def dataloader_construct(
    name, config, eval_setting, dataset, dl_format=InputType.POINTWISE, batch_size=1, shuffle=False, **kwargs
):
    """Get a correct dataloader class by calling :func:`get_data_loader` to construct dataloader.

    Args:
        name (str): The stage of dataloader. It can only take two values: 'train' or 'evaluation'.
        config (Config): An instance object of Config, used to record parameter information.
        eval_setting (EvalSetting): An instance object of EvalSetting, used to record evaluation settings.
        dataset (Dataset or list of Dataset): The split dataset for constructing dataloader.
        dl_format (InputType, optional): The input type of dataloader. Defaults to
            :obj:`~recbole.utils.enum_type.InputType.POINTWISE`.
        batch_size (int, optional): The batch_size of dataloader. Defaults to ``1``.
        shuffle (bool, optional): Whether the dataloader will be shuffle after a round. Defaults to ``False``.
        **kwargs: Other input args of dataloader, such as :attr:`sampler`, :attr:`kg_sampler`
            and :attr:`neg_sample_args`. The meaning of these args is the same as these args in some dataloaders.

    Returns:
        AbstractDataLoader or list of AbstractDataLoader: Constructed dataloader in split dataset.
    """
    if not isinstance(dataset, list):
        dataset = [dataset]

    if not isinstance(batch_size, list):
        batch_size = [batch_size] * len(dataset)

    if len(dataset) != len(batch_size):
        raise ValueError(f'Dataset {dataset} and batch_size {batch_size} should have the same length.')

    kwargs_list = [{} for _ in range(len(dataset))]
    for key, value in kwargs.items():
        key = [key] * len(dataset)
        if not isinstance(value, list):
            value = [value] * len(dataset)
        if len(dataset) != len(value):
            raise ValueError(f'Dataset {dataset} and {key} {value} should have the same length.')
        for kw, k, w in zip(kwargs_list, key, value):
            kw[k] = w

    model_type = config['MODEL_TYPE']
    logger = getLogger()
    logger.info(f'Build [{model_type}] DataLoader for [{name}] with format [{dl_format}]')
    logger.info(eval_setting)
    logger.info(f'batch_size = [{batch_size}], shuffle = [{shuffle}]\n')

    dataloader = get_data_loader(name, config, eval_setting)

    try:
        ret = [
            dataloader(config=config, dataset=ds, batch_size=bs, dl_format=dl_format, shuffle=shuffle, **kw)
            for ds, bs, kw in zip(dataset, batch_size, kwargs_list)
        ]
    except TypeError:
        raise ValueError('training_neg_sample_num should be 0')

    if len(ret) == 1:
        return ret[0]
    else:
        return ret


def save_datasets(save_path, name, dataset):
    """Save split datasets.

    Args:
        save_path (str): The path of directory for saving.
        name (str or list of str): The stage of dataloader. It can only take two values: 'train' or 'evaluation'.
        dataset (Dataset or list of Dataset): The split datasets.
    """
    if (not isinstance(name, list)) and (not isinstance(dataset, list)):
        name = [name]
        dataset = [dataset]
    if len(name) != len(dataset):
        raise ValueError(f'Length of name {name} should equal to length of dataset {dataset}.')
    for i, d in enumerate(dataset):
        cur_path = os.path.join(save_path, name[i])
        ensure_dir(cur_path)
        d.save(cur_path)


def get_data_loader(name, config, eval_setting):
    """Return a dataloader class according to :attr:`config` and :attr:`eval_setting`.

    Args:
        name (str): The stage of dataloader. It can only take two values: 'train' or 'evaluation'.
        config (Config): An instance object of Config, used to record parameter information.
        eval_setting (EvalSetting): An instance object of EvalSetting, used to record evaluation settings.

    Returns:
        type: The dataloader class that meets the requirements in :attr:`config` and :attr:`eval_setting`.
    """
    register_table = {
        'DIN': _get_DIN_data_loader,
        "MultiDAE": _get_AE_data_loader,
        "MultiVAE": _get_AE_data_loader,
        'MacridVAE': _get_AE_data_loader,
        'CDAE': _get_AE_data_loader,
        'ENMF': _get_AE_data_loader,
        'RaCT': _get_AE_data_loader
    }

    if config['model'] in register_table:
        return register_table[config['model']](name, config, eval_setting)

    model_type = config['MODEL_TYPE']
    neg_sample_strategy = eval_setting.neg_sample_args['strategy']
    if model_type == ModelType.GENERAL or model_type == ModelType.TRADITIONAL:
        if neg_sample_strategy == 'none':
            return GeneralDataLoader
        elif neg_sample_strategy == 'by':
            return GeneralNegSampleDataLoader
        elif neg_sample_strategy == 'full':
            return GeneralFullDataLoader
    elif model_type == ModelType.CONTEXT:
        if neg_sample_strategy == 'none':
            return ContextDataLoader
        elif neg_sample_strategy == 'by':
            return ContextNegSampleDataLoader
        elif neg_sample_strategy == 'full':
            return ContextFullDataLoader
    elif model_type == ModelType.SEQUENTIAL:
        if neg_sample_strategy == 'none':
            return SequentialDataLoader
        elif neg_sample_strategy == 'by':
            return SequentialNegSampleDataLoader
        elif neg_sample_strategy == 'full':
            return SequentialFullDataLoader
    elif model_type == ModelType.DECISIONTREE:
        if neg_sample_strategy == 'none':
            return DecisionTreeDataLoader
        elif neg_sample_strategy == 'by':
            return DecisionTreeNegSampleDataLoader
        elif neg_sample_strategy == 'full':
            return DecisionTreeFullDataLoader
    elif model_type == ModelType.KNOWLEDGE:
        if neg_sample_strategy == 'by':
            if name == 'train':
                return KnowledgeBasedDataLoader
            else:
                return GeneralNegSampleDataLoader
        elif neg_sample_strategy == 'full':
            return GeneralFullDataLoader
        elif neg_sample_strategy == 'none':
            # return GeneralDataLoader
            # TODO 训练也可以为none? 看general的逻辑似乎是都可以为None
            raise NotImplementedError(
                'The use of external negative sampling for knowledge model has not been implemented'
            )
    else:
        raise NotImplementedError(f'Model_type [{model_type}] has not been implemented.')


def _get_DIN_data_loader(name, config, eval_setting):
    """Customized function for DIN to get correct dataloader class.

    Args:
        name (str): The stage of dataloader. It can only take two values: 'train' or 'evaluation'.
        config (Config): An instance object of Config, used to record parameter information.
        eval_setting (EvalSetting): An instance object of EvalSetting, used to record evaluation settings.

    Returns:
        type: The dataloader class that meets the requirements in :attr:`config` and :attr:`eval_setting`.
    """
    neg_sample_strategy = eval_setting.neg_sample_args['strategy']
    if neg_sample_strategy == 'none':
        return SequentialDataLoader
    elif neg_sample_strategy == 'by':
        return SequentialNegSampleDataLoader
    elif neg_sample_strategy == 'full':
        return SequentialFullDataLoader


def _get_AE_data_loader(name, config, eval_setting):
    """Customized function for Multi-DAE and Multi-VAE to get correct dataloader class.

    Args:
        name (str): The stage of dataloader. It can only take two values: 'train' or 'evaluation'.
        config (Config): An instance object of Config, used to record parameter information.
        eval_setting (EvalSetting): An instance object of EvalSetting, used to record evaluation settings.

    Returns:
        type: The dataloader class that meets the requirements in :attr:`config` and :attr:`eval_setting`.
    """
    neg_sample_strategy = eval_setting.neg_sample_args['strategy']
    if name == "train":
        return UserDataLoader
    else:
        if neg_sample_strategy == 'none':
            return GeneralDataLoader
        elif neg_sample_strategy == 'by':
            return GeneralNegSampleDataLoader
        elif neg_sample_strategy == 'full':
            return GeneralFullDataLoader


class DLFriendlyAPI(object):
    """A Decorator class, which helps copying :class:`Dataset` methods to :class:`DataLoader`.

    These methods are called *DataLoader Friendly APIs*.

    E.g. if ``train_data`` is an object of :class:`DataLoader`,
    and :meth:`~recbole.data.dataset.dataset.Dataset.num` is a method of :class:`~recbole.data.dataset.dataset.Dataset`,
    Cause it has been decorated, :meth:`~recbole.data.dataset.dataset.Dataset.num` can be called directly by
    ``train_data``.

    See the example of :meth:`set` for details.

    Attributes:
        dataloader_apis (set): Register table that saves all the method names of DataLoader Friendly APIs.
    """

    def __init__(self):
        self.dataloader_apis = set()

    def __iter__(self):
        return self.dataloader_apis

    def set(self):
        """
        Example:
            .. code:: python

                from recbole.data.utils import dlapi

                @dlapi.set()
                def dataset_meth():
                    ...
        """

        def decorator(f):
            self.dataloader_apis.add(f.__name__)
            return f

        return decorator


dlapi = DLFriendlyAPI()
