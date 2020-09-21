# -*- coding: utf-8 -*-
# @Time   : 2020/7/17
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn

import os
import datetime
import importlib

from recbox.utils.enum_type import ModelType


def get_local_time():
    cur = datetime.datetime.now()
    cur = cur.strftime('%b-%d-%Y_%H-%M-%S')

    return cur


def ensure_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def get_model(model_name):
    model_submodule = [
        'general_recommender',
        'context_aware_recommender',
        'sequential_recommender',
        'knowledge_aware_recommender'
    ]

    model_file_name = model_name.lower()
    for submodule in model_submodule:
        module_path = '.'.join(['...model', submodule, model_file_name])
        if importlib.util.find_spec(module_path, __name__):
            model_module = importlib.import_module(module_path, __name__)

    model_class = getattr(model_module, model_name)
    return model_class


def get_trainer(model_type, model_name):
    try:
        return getattr(importlib.import_module('recbox.trainer'), model_name + 'Trainer')
    except AttributeError:
        if model_type == ModelType.KNOWLEDGE:
            return getattr(importlib.import_module('recbox.trainer'), 'KGTrainer')
        else:
            return getattr(importlib.import_module('recbox.trainer'), 'Trainer')
