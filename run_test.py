# -*- coding: utf-8 -*-
# @Time   : 2020/7/20 20:32
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn
# @File   : run_test.py

# UPDATE
# @Time   : 2020/10/3, 2020/10/1
# @Author : Yupeng Hou, Zihan Lin
# @Email  : houyupeng@ruc.edu.cn, zhlin@ruc.edu.cn


import argparse

from recbox.quick_start import run_unirec


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='BPRMF', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='ml-100k', help='name of datasets')
    parser.add_argument('--quick_start', action='store_true', help='whether to use quick-start')

    args, _ = parser.parse_known_args()

    run_unirec(model=args.model, dataset=args.dataset, quick_start=args.quick_start)
