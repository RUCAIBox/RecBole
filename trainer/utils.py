# -*- coding: utf-8 -*-
# @Time   : 2020/6/26 22:05
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn
# @File   : utils.py


def early_stopping(value, best, cur_step, max_step, order='asc'):
    """
    early stopping
    :param value: current value
    :param best: best value
    :param cur_step: current step
    :param max_step: threshold
    :param order: asc or desc
    """
    stop_flag = False
    update_flag = False
    if order == 'asc':
        if value > best:
            cur_step = 0
            best = value
            update_flag = True
        else:
            cur_step += 1
            if cur_step > max_step:
                stop_flag = True
    else:
        if value < best:
            cur_step = 0
            best = value
            update_flag = True
        else:
            cur_step += 1
            if cur_step > max_step:
                stop_flag = True
    return best, cur_step, stop_flag, update_flag


def calculate_valid_score(valid_result):
    """
    todo:  Add some rules
    :param valid_result:
    :return: score(float)
    """
    return valid_result['Recall@10']


# todo: define this function
def dict2str():
    pass
