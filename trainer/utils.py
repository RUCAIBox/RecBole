# -*- coding: utf-8 -*-
# @Time   : 2020/6/26 22:05
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn
# @File   : utils.py


def early_stopping(value, best, cur_step, max_step, order='asc'):
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


def calculate_valid_score(valid_result, valid_metric=None):
    if valid_metric:
        return valid_result[valid_metric]
    else:
        return valid_result['Recall@10']


# todo: format adjustment
def dict2str(result_dict):
    result_str = ''
    for metric, value in result_dict.items():
        result_str += str(metric) + ' : ' + '%.04f' % value + '    '
    return result_str
