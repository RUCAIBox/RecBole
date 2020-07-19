# -*- coding: utf-8 -*-
# @Time   : 2020/7/17 16:13
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn
# @File   : run_hyper.py

from trainer import HyperTuning


def main():
    hp = HyperTuning('main.py', params_file='trainer/hyper.example', max_evals=5)
    hp.run()
    # print(hp.best_params)


if __name__ == '__main__':
    main()
