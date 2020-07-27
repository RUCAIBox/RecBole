# -*- coding: utf-8 -*-
# @Time   : 2020/7/19 19:06
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn
# @File   : hyper_tuning.py


import hyperopt
from hyperopt import fmin, tpe, hp


class HyperTuning(object):
    def __init__(self, objective_function, space=None, params_file=None, algo=tpe.suggest, max_evals=100):
        self.objective_function = objective_function
        self.algo = algo
        self.max_evals = max_evals
        self.best_score = None
        self.best_params = None
        self.best_test_result = None
        if space:
            self.space = space
        elif params_file:
            self.space = self._build_space_from_file(params_file)
        else:
            raise ValueError('at least one of `space` and `params_file` is provided')

    @staticmethod
    def _build_space_from_file(file):
        space = {}
        with open(file, 'r') as fp:
            for line in fp:
                para_name, para_type, para_value = line.strip().split(' ')
                if para_type == 'choice':
                    para_value = eval(para_value)
                    space[para_name] = hp.choice(para_name, para_value)
                elif para_type == 'uniform':
                    low, high = para_value.strip().split(',')
                    space[para_name] = hp.uniform(para_name, float(low), float(high))
                elif para_type == 'quniform':
                    low, high, q = para_value.strip().split(',')
                    space[para_name] = hp.quniform(para_name, float(low), float(high), float(q))
                elif para_type == 'loguniform':
                    low, high = para_value.strip().split(',')
                    space[para_name] = hp.loguniform(para_name, float(low), float(high))
                else:
                    raise ValueError('Illegal para type [{}]'.format(para_type))
        return space

    @staticmethod
    def params2config(params):
        config_dict = {}
        for param_name in params:
            config_dict[param_name] = params[param_name]
        return config_dict

    def trial(self, params):
        config_dict = self.params2config(params)
        best_valid_score, bigger = self.objective_function(config_dict)
        score = best_valid_score

        if not self.best_score:
            self.best_score = score
            self.best_params = params
        else:
            if bigger:
                if score > self.best_score:
                    self.best_score = score
                    self.best_params = params
            else:
                if score < self.best_score:
                    self.best_score = score
                    self.best_params = params

        if bigger:
            score = - score
        return {'loss': score, 'status': hyperopt.STATUS_OK}

    def run(self):
        fmin(self.trial, self.space, algo=self.algo, max_evals=self.max_evals)
