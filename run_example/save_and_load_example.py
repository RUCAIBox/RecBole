# @Time   : 2021/03/20
# @Author : Yushuo Chen
# @Email  : chenyushuo@ruc.edu.cn

"""
save and load example
========================
Here is the sample code for the save and load in RecBole.

The path to saved data or model can be found in the output of RecBole.
"""

import pickle
from logging import getLogger

import torch

from recbole.config import Config
from recbole.data import create_dataset, data_preparation, save_split_dataloaders, load_split_dataloaders
from recbole.utils import init_seed, init_logger, get_model, get_trainer


def save_example():
    # configurations initialization
    config_dict = {
        'checkpoint_dir': '../saved'
    }
    config = Config(model='BPR', dataset='ml-100k', config_dict=config_dict)
    init_seed(config['seed'], config['reproducibility'])
    init_logger(config)

    # dataset filtering
    dataset = create_dataset(config)
    dataset.save('../saved/')

    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)
    save_split_dataloaders(config, dataloaders=(train_data, valid_data, test_data))

    model = get_model(config['model'])(config, train_data).to(config['device'])

    # trainer loading and initialization
    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)

    # model training
    # the best model will be saved in here
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=True, show_progress=config['show_progress']
    )


def load_example():
    # configurations initialization
    config_dict = {
        'checkpoint_dir': '../saved'
    }
    config = Config(model='BPR', dataset='ml-100k', config_dict=config_dict)
    init_seed(config['seed'], config['reproducibility'])
    init_logger(config)
    logger = getLogger()

    with open('../saved/ml-100k-dataset.pth', 'rb') as f:  # You can use your filtered data path here.
        dataset = pickle.load(f)

    train_data, valid_data, test_data = load_split_dataloaders('../saved/ml-100k-for-BPR-dataloader.pth')
    # You can use your split data path here.

    model = get_model(config['model'])(config, train_data).to(config['device'])
    checkpoint = torch.load('../saved/BPR-Mar-20-2021_17-11-05.pth')  # Here you can replace it by your model path.
    model.load_state_dict(checkpoint['state_dict'])
    model.load_other_parameter(checkpoint.get('other_parameter'))
    logger.info(model)
    logger.info(train_data.dataset)
    logger.info(valid_data.dataset)
    logger.info(test_data.dataset)


if __name__ == '__main__':
    save_example()
