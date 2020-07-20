# -*- coding: utf-8 -*-
# @Time   : 2020/7/17 16:13
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn
# @File   : run_hyper.py

import argparse
from trainer import HyperTuning


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_evals', type=int, default=100)
    return parser.parse_args()


def main():
    args = parser_args()
    hp = HyperTuning('run_test.py', params_file='hyper.test', max_evals=args.max_evals)
    hp.run()
    print(hp.best_params)


if __name__ == '__main__':
    main()
