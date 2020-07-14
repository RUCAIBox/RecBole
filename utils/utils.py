import os
import datetime


def get_local_time():
    cur = datetime.datetime.now()
    cur = cur.strftime('%b-%d-%Y_%H-%M-%S')

    return cur


def ensure_dir(dir_path):
    dir_path = os.path.dirname(dir_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
