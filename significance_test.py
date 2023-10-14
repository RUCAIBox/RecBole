# @Time   : 2023/2/11
# @Author : Jiakai Tang
# @Email  : tangjiakai5704@ruc.edu.cn

# UPDATE
# @Time   :
# @Author :
# @Email  :

import argparse
import random
from collections import defaultdict

from scipy import stats

from recbole.quick_start import run


def run_test(
    model,
    dataset,
    config_files,
    seeds,
    nproc,
    world_size,
    ip,
    port,
    group_offset,
):
    results = defaultdict(list)
    for seed in seeds:
        res = run(
            model,
            dataset,
            config_files,
            config_dict={"seed": seed},
            nproc=nproc,
            world_size=world_size,
            ip=ip,
            port=port,
            group_offset=group_offset,
        )
        for _key, _value in res["test_result"].items():
            results[_key].append(_value)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_ours", type=str, default="BPR", help="name of our models"
    )
    parser.add_argument(
        "--model_baseline", type=str, default="NeuMF", help="name of baseline models"
    )
    parser.add_argument(
        "--dataset", "-d", type=str, default="ml-100k", help="name of datasets"
    )
    parser.add_argument(
        "--config_files",
        type=str,
        default=None,
        help="config files: 1st is our model and 2ed is baseline",
    )
    parser.add_argument(
        "--st_seed", type=int, default=2023, help="st_seed for generating random seeds"
    )
    parser.add_argument(
        "--run_times", type=int, default=10, help="run times for each model"
    )
    parser.add_argument(
        "--nproc", type=int, default=1, help="the number of process in this group"
    )
    parser.add_argument(
        "--ip", type=str, default="localhost", help="the ip of master node"
    )
    parser.add_argument(
        "--port", type=str, default="5678", help="the port of master node"
    )
    parser.add_argument(
        "--world_size", type=int, default=-1, help="total number of jobs"
    )
    parser.add_argument(
        "--group_offset",
        type=int,
        default=0,
        help="the global rank offset of this group",
    )

    args, _ = parser.parse_known_args()

    config_file_list = (
        args.config_files.strip().split(" ") if args.config_files else None
    )

    if len(config_file_list) != 2:
        raise ValueError("You have to specify 2 config files")

    random.seed(args.st_seed)
    random_seeds = [random.randint(0, 2**32 - 1) for _ in range(args.run_times)]

    config_file_ours, config_file_baseline = config_file_list

    result_ours = run_test(
        args.model_ours,
        args.dataset,
        [config_file_ours],
        random_seeds,
        args.nproc,
        args.world_size,
        args.ip,
        args.port,
        args.group_offset,
    )
    result_baseline = run_test(
        args.model_baseline,
        args.dataset,
        [config_file_baseline],
        random_seeds,
        args.nproc,
        args.world_size,
        args.ip,
        args.port,
        args.group_offset,
    )

    final_result = {}
    for key, value in result_ours.items():
        if key not in result_baseline:
            continue
        ours = value
        baseline = result_baseline[key]
        final_result[key] = stats.ttest_rel(ours, baseline, alternative="less")

    with open("significant_test.txt", "w") as f:
        for key, value in final_result.items():
            print(f"{key}: statistic={value.statistic}, pvalue={value.pvalue}\n")
            f.write(f"{key}: statistic={value.statistic}, pvalue={value.pvalue}\n")
