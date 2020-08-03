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


def get_model(config):

    model_name = config['model']
    model_file_name = model_name.lower()
    if importlib.util.find_spec("model.general_recommender." + model_file_name) is not None:
        model_module = importlib.import_module("model.general_recommender." + model_file_name)
    elif importlib.util.find_spec("model.context_aware_recommender." + model_file_name) is not None:

        model_module = importlib.import_module("model.context_aware_recommender." + model_file_name)
    else:
        model_module = importlib.import_module("model.sequential_recommender." + model_file_name)

    model_class = getattr(model_module, model_name)

    return model_class
