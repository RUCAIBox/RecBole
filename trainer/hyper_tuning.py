# -*- coding: utf-8 -*-
# @Time   : 2020/7/19 19:06
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn
# @File   : hyper_tuning.py

import sys
import subprocess
import hyperopt
from hyperopt import fmin, tpe, hp


class HyperTuning(object):
    def __init__(self, procedure_file, space=None, params_file=None, interpreter='python', algo=tpe.suggest, max_evals=100, bigger=True):
        self.filename = procedure_file
        self.interpreter = interpreter
        self.algo = algo
        self.max_evals = max_evals
        self.bigger = bigger
        if self.bigger:
            self.init_score = - float('inf')
        else:
            self.init_score = float('inf')
        self.best_score = self.init_score
        self.best_params = None
        if space:
            self.space = space
        elif params_file:
            self.space = self._build_space_from_file(params_file)
        else:
            raise ValueError('at least one of `space` and `params_file` is provided')

    @staticmethod
    def flush():
        sys.stdout.flush()
        sys.stderr.flush()

    @staticmethod
    def params2cmd(interpreter, filename, params):
        cmd = interpreter + ' ' + filename
        for param_name in params:
            param_value = params[param_name]
            cmd += ' --' + param_name
            if isinstance(param_value, str):
                cmd += '=%s' % param_value
            elif int(param_value) == param_value:
                cmd += '=%d' % int(param_value)
            else:
                cmd += '=%g' % float('%.1e' % float(param_value))
        return cmd

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

    def trial(self, params):
        cmd = self.params2cmd(self.interpreter, self.filename, params)
        try:
            print('\n\n running command: @ %s' % cmd, file=sys.stderr)
            self.flush()
            output = subprocess.check_output(cmd, shell=True)
        except subprocess.CalledProcessError:
            return {'loss': self.init_score, 'status': hyperopt.STATUS_FAIL}
        output = output.decode(encoding='UTF-8')
        score = float(output.strip().split('\n')[-1])
        if self.bigger:
            if score > self.best_score:
                self.best_score = score
                self.best_params = params
            score = - score
        else:
            if score < self.best_score:
                self.best_score = score
                self.best_params = params
        return {'loss': score, 'status': hyperopt.STATUS_OK}

    def run(self):
        fmin(self.trial, self.space, algo=self.algo, max_evals=self.max_evals)
