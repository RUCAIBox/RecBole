import os
import datetime
import importlib


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
