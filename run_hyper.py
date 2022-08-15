# -*- coding: utf-8 -*-
# @Time   : 2020/7/24 15:57
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn
# @File   : run_hyper.py
# UPDATE:
# @Time   : 2020/8/20 21:17, 2020/8/29, 2022/7/13, 2022/7/18
# @Author : Zihan Lin, Yupeng Hou, Gaowei Zhang, Lei Wang
# @Email  : linzihan.super@foxmail.com, houyupeng@ruc.edu.cn, zgw15630559577@163.com, zxcptss@gmail.com

import argparse
import os
import numpy as np

from recbole.trainer import HyperTuning
from recbole.quick_start import objective_function
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler


def hyperopt_tune(args):

    # plz set algo='exhaustive' to use exhaustive search, in this case, max_evals is auto set
    # in other case, max_evals needs to be set manually
    config_file_list = (
        args.config_files.strip().split(" ") if args.config_files else None
    )
    hp = HyperTuning(
        objective_function,
        algo="exhaustive",
        early_stop=10,
        max_evals=100,
        params_file=args.params_file,
        fixed_config_file_list=config_file_list,
        display_file=args.display_file
    )
    hp.run()
    hp.export_result(output_file=args.output_file)
    print("best params: ", hp.best_params)
    print("best result: ")
    print(hp.params2result[hp.params2str(hp.best_params)])


def ray_tune(args):

    config_file_list = (
        args.config_files.strip().split(" ") if args.config_files else None
    )
    config_file_list = (
        [os.path.join(os.getcwd(), file) for file in config_file_list]
        if args.config_files
        else None
    )

    ray.init()
    tune.register_trainable("train_func", objective_function)
    config = {
        "embedding_size": tune.sample_from(lambda _: 2 ** np.random.randint(3, 4)),
        "learning_rate": tune.loguniform(1e-4, 1e-1),
    }
    # choose different schedulers to use different tuning optimization algorithms
    # For details, please refer to Ray's official website https://docs.ray.io
    scheduler = ASHAScheduler(
        metric="recall@10", mode="max", max_t=10, grace_period=1, reduction_factor=2
    )

    local_dir = "./ray_log"
    result = tune.run(
        tune.with_parameters(objective_function, config_file_list=config_file_list),
        config=config,
        num_samples=5,
        log_to_file=args.output_file,
        scheduler=scheduler,
        local_dir=local_dir,
    )

    best_trial = result.get_best_trial("recall@10", "max", "last")
    print("best params: ", best_trial.config)
    print("best result: ", best_trial.last_result)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_files", type=str, default=None, help="fixed config files"
    )
    parser.add_argument("--params_file", type=str, default=None, help="parameters file")
    parser.add_argument(
        "--output_file", type=str, default="hyper_example.result", help="output file"
    )
    parser.add_argument("--display_file", type=str, default=None, help="visualization file")
    parser.add_argument("--tool", type=str, default="Hyperopt", help="tuning tool")
    args, _ = parser.parse_known_args()
    if args.tool == "Hyperopt":
        hyperopt_tune(args)
    elif args.tool == "Ray":
        ray_tune(args)
    else:
        raise ValueError(f"The tool [{args.tool}] should in ['Hyperopt', 'Ray']")
