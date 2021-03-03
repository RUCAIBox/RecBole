# -*- coding: utf-8 -*-
# @Time   : 2020/7/19 19:06
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn
# @File   : hyper_tuning.py

"""
recbole.trainer.hyper_tuning
############################
"""

from functools import partial

import numpy as np

from recbole.utils.utils import dict2str


def _recursiveFindNodes(root, node_type='switch'):
    from hyperopt.pyll.base import Apply
    nodes = []
    if isinstance(root, (list, tuple)):
        for node in root:
            nodes.extend(_recursiveFindNodes(node, node_type))
    elif isinstance(root, dict):
        for node in root.values():
            nodes.extend(_recursiveFindNodes(node, node_type))
    elif isinstance(root, (Apply)):
        if root.name == node_type:
            nodes.append(root)

        for node in root.pos_args:
            if node.name == node_type:
                nodes.append(node)
        for _, node in root.named_args:
            if node.name == node_type:
                nodes.append(node)
    return nodes


def _parameters(space):
    # Analyze the domain instance to find parameters
    parameters = {}
    if isinstance(space, dict):
        space = list(space.values())
    for node in _recursiveFindNodes(space, 'switch'):
        # Find the name of this parameter
        paramNode = node.pos_args[0]
        assert paramNode.name == 'hyperopt_param'
        paramName = paramNode.pos_args[0].obj

        # Find all possible choices for this parameter
        values = [literal.obj for literal in node.pos_args[1:]]
        parameters[paramName] = np.array(range(len(values)))
    return parameters


def _spacesize(space):
    # Compute the number of possible combinations
    params = _parameters(space)
    return np.prod([len(values) for values in params.values()])


class ExhaustiveSearchError(Exception):
    r""" ExhaustiveSearchError

    """
    pass


def _validate_space_exhaustive_search(space):
    from hyperopt.pyll.base import dfs, as_apply
    from hyperopt.pyll.stochastic import implicit_stochastic_symbols
    supported_stochastic_symbols = ['randint', 'quniform', 'qloguniform', 'qnormal', 'qlognormal', 'categorical']
    for node in dfs(as_apply(space)):
        if node.name in implicit_stochastic_symbols:
            if node.name not in supported_stochastic_symbols:
                raise ExhaustiveSearchError(
                    'Exhaustive search is only possible with the following stochastic symbols: '
                    '' + ', '.join(supported_stochastic_symbols)
                )


def exhaustive_search(new_ids, domain, trials, seed, nbMaxSucessiveFailures=1000):
    r""" This is for exhaustive search in HyperTuning.

    """
    from hyperopt import pyll
    from hyperopt.base import miscs_update_idxs_vals
    # Build a hash set for previous trials
    hashset = set([
        hash(
            frozenset([(key, value[0]) if len(value) > 0 else ((key, None))
                       for key, value in trial['misc']['vals'].items()])
        ) for trial in trials.trials
    ])

    rng = np.random.RandomState(seed)
    rval = []
    for _, new_id in enumerate(new_ids):
        newSample = False
        nbSucessiveFailures = 0
        while not newSample:
            # -- sample new specs, idxs, vals
            idxs, vals = pyll.rec_eval(domain.s_idxs_vals, memo={
                domain.s_new_ids: [new_id],
                domain.s_rng: rng,
            })
            new_result = domain.new_result()
            new_misc = dict(tid=new_id, cmd=domain.cmd, workdir=domain.workdir)
            miscs_update_idxs_vals([new_misc], idxs, vals)

            # Compare with previous hashes
            h = hash(frozenset([(key, value[0]) if len(value) > 0 else ((key, None)) for key, value in vals.items()]))
            if h not in hashset:
                newSample = True
            else:
                # Duplicated sample, ignore
                nbSucessiveFailures += 1

            if nbSucessiveFailures > nbMaxSucessiveFailures:
                # No more samples to produce
                return []

        rval.extend(trials.new_trial_docs([new_id], [None], [new_result], [new_misc]))
    return rval


class HyperTuning(object):
    r"""HyperTuning Class is used to manage the parameter tuning process of recommender system models.
    Given objective funciton, parameters range and optimization algorithm, using HyperTuning can find
    the best result among these parameters

    Note:
        HyperTuning is based on the hyperopt (https://github.com/hyperopt/hyperopt)

        Thanks to sbrodeur for the exhaustive search code.
        https://github.com/hyperopt/hyperopt/issues/200
    """

    def __init__(
        self,
        objective_function,
        space=None,
        params_file=None,
        params_dict=None,
        fixed_config_file_list=None,
        algo='exhaustive',
        max_evals=100
    ):
        self.best_score = None
        self.best_params = None
        self.best_test_result = None
        self.params2result = {}

        self.objective_function = objective_function
        self.max_evals = max_evals
        self.fixed_config_file_list = fixed_config_file_list
        if space:
            self.space = space
        elif params_file:
            self.space = self._build_space_from_file(params_file)
        elif params_dict:
            self.space = self._build_space_from_dict(params_dict)
        else:
            raise ValueError('at least one of `space`, `params_file` and `params_dict` is provided')
        if isinstance(algo, str):
            if algo == 'exhaustive':
                self.algo = partial(exhaustive_search, nbMaxSucessiveFailures=1000)
                self.max_evals = _spacesize(self.space)
            else:
                raise ValueError('Illegal algo [{}]'.format(algo))
        else:
            self.algo = algo

    @staticmethod
    def _build_space_from_file(file):
        from hyperopt import hp
        space = {}
        with open(file, 'r') as fp:
            for line in fp:
                para_list = line.strip().split(' ')
                if len(para_list) < 3:
                    continue
                para_name, para_type, para_value = para_list[0], para_list[1], "".join(para_list[2:])
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
                    raise ValueError('Illegal param type [{}]'.format(para_type))
        return space

    @staticmethod
    def _build_space_from_dict(config_dict):
        from hyperopt import hp
        space = {}
        for para_type in config_dict:
            if para_type == 'choice':
                for para_name in config_dict['choice']:
                    para_value = config_dict['choice'][para_name]
                    space[para_name] = hp.choice(para_name, para_value)
            elif para_type == 'uniform':
                for para_name in config_dict['uniform']:
                    para_value = config_dict['uniform'][para_name]
                    low = para_value[0]
                    high = para_value[1]
                    space[para_name] = hp.uniform(para_name, float(low), float(high))
            elif para_type == 'quniform':
                for para_name in config_dict['quniform']:
                    para_value = config_dict['quniform'][para_name]
                    low = para_value[0]
                    high = para_value[1]
                    q = para_value[2]
                    space[para_name] = hp.quniform(para_name, float(low), float(high), float(q))
            elif para_type == 'loguniform':
                for para_name in config_dict['loguniform']:
                    para_value = config_dict['loguniform'][para_name]
                    low = para_value[0]
                    high = para_value[1]
                    space[para_name] = hp.loguniform(para_name, float(low), float(high))
            else:
                raise ValueError('Illegal param type [{}]'.format(para_type))
        return space

    @staticmethod
    def params2str(params):
        r""" convert dict to str

        Args:
            params (dict): parameters dict
        Returns:
            str: parameters string
        """
        params_str = ''
        for param_name in params:
            params_str += param_name + ':' + str(params[param_name]) + ', '
        return params_str[:-2]

    @staticmethod
    def _print_result(result_dict: dict):
        print('current best valid score: %.4f' % result_dict['best_valid_score'])
        print('current best valid result:')
        print(result_dict['best_valid_result'])
        print('current test result:')
        print(result_dict['test_result'])
        print()

    def export_result(self, output_file=None):
        r""" Write the searched parameters and corresponding results to the file

        Args:
            output_file (str): the output file

        """
        with open(output_file, 'w') as fp:
            for params in self.params2result:
                fp.write(params + '\n')
                fp.write('Valid result:\n' + dict2str(self.params2result[params]['best_valid_result']) + '\n')
                fp.write('Test result:\n' + dict2str(self.params2result[params]['test_result']) + '\n\n')

    def trial(self, params):
        r"""Given a set of parameters, return results and optimization status

        Args:
            params (dict): the parameter dictionary
        """
        import hyperopt
        config_dict = params.copy()
        params_str = self.params2str(params)
        print('running parameters:', config_dict)
        result_dict = self.objective_function(config_dict, self.fixed_config_file_list)
        self.params2result[params_str] = result_dict
        score, bigger = result_dict['best_valid_score'], result_dict['valid_score_bigger']

        if not self.best_score:
            self.best_score = score
            self.best_params = params
            self._print_result(result_dict)
        else:
            if bigger:
                if score > self.best_score:
                    self.best_score = score
                    self.best_params = params
                    self._print_result(result_dict)
            else:
                if score < self.best_score:
                    self.best_score = score
                    self.best_params = params
                    self._print_result(result_dict)

        if bigger:
            score = -score
        return {'loss': score, 'status': hyperopt.STATUS_OK}

    def run(self):
        r""" begin to search the best parameters

        """
        from hyperopt import fmin
        fmin(self.trial, self.space, algo=self.algo, max_evals=self.max_evals)
