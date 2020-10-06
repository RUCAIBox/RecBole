# -*- coding: utf-8 -*-
# @Time   : 2020/7/24 15:57
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn
# @File   : run_hyper.py
# UPDATE:
# @Time   : 2020/8/20 21:17, 2020/8/29
# @Author : Zihan Lin, Yupeng Hou
# @Email  : linzihan.super@foxmail.com, houyupeng@ruc.edu.cn


from recbox.trainer import HyperTuning
from recbox.quick_start import objective_function


def main():
    # plz set algo='exhaustive' to use exhaustive search, in this case, max_evals is auto set
    hp = HyperTuning(objective_function, algo='exhaustive', params_file='hyper2.test')
    hp.run()
    hp.export_result(output_file='hyper_example.result')
    print('best params: ', hp.best_params)
    print('best result: ')
    print(hp.params2result[hp.params2str(hp.best_params)])


if __name__ == '__main__':
    main()
