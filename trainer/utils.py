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
