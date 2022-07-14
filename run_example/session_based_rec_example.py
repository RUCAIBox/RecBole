# @Time   : 2021/7/14
# @Author : Yupeng Hou
# @Email  : houyupeng@ruc.edu.cn

# UPDATE
# @Time   : 2021/7/21
# @Author : Yupeng Hou
# @Email  : houyupeng@ruc.edu.cn

"""
session-based recommendation example
========================
Here is the sample code for running session-based recommendation benchmarks using RecBole.

args.dataset can be one of diginetica-session/tmall-session/nowplaying-session
"""

import argparse
from logging import getLogger

from recbole.config import Config
from recbole.data import create_dataset
from recbole.data.utils import get_dataloader
from recbole.utils import init_logger, init_seed, get_model, get_trainer, set_color


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="GRU4Rec",
        help="Model for session-based rec.",
    )
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        default="diginetica-session",
        help="Benchmarks for session-based rec.",
    )
    parser.add_argument(
        "--validation",
        action="store_true",
        help="Whether evaluating on validation set (split from train set), otherwise on test set.",
    )
    parser.add_argument(
        "--valid_portion", type=float, default=0.1, help="ratio of validation set."
    )
    return parser.parse_known_args()[0]


if __name__ == "__main__":
    args = get_args()

    # configurations initialization
    config_dict = {
        "USER_ID_FIELD": "session_id",
        "load_col": None,
        "neg_sampling": None,
        "benchmark_filename": ["train", "test"],
        "alias_of_item_id": ["item_id_list"],
        "topk": [20],
        "metrics": ["Recall", "MRR"],
        "valid_metric": "MRR@20",
    }

    config = Config(
        model=args.model, dataset=f"{args.dataset}", config_dict=config_dict
    )
    init_seed(config["seed"], config["reproducibility"])

    # logger initialization
    init_logger(config)
    logger = getLogger()

    logger.info(args)
    logger.info(config)

    # dataset filtering
    dataset = create_dataset(config)
    logger.info(dataset)

    # dataset splitting
    train_dataset, test_dataset = dataset.build()
    if args.validation:
        train_dataset.shuffle()
        new_train_dataset, new_test_dataset = train_dataset.split_by_ratio(
            [1 - args.valid_portion, args.valid_portion]
        )
        train_data = get_dataloader(config, "train")(
            config, new_train_dataset, None, shuffle=True
        )
        test_data = get_dataloader(config, "test")(
            config, new_test_dataset, None, shuffle=False
        )
    else:
        train_data = get_dataloader(config, "train")(
            config, train_dataset, None, shuffle=True
        )
        test_data = get_dataloader(config, "test")(
            config, test_dataset, None, shuffle=False
        )

    # model loading and initialization
    model = get_model(config["model"])(config, train_data.dataset).to(config["device"])
    logger.info(model)

    # trainer loading and initialization
    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)

    # model training and evaluation
    test_score, test_result = trainer.fit(
        train_data, test_data, saved=True, show_progress=config["show_progress"]
    )

    logger.info(set_color("test result", "yellow") + f": {test_result}")
