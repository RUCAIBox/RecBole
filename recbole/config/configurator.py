# @Time   : 2020/6/28
# @Author : Zihan Lin
# @Email  : linzihan.super@foxmail.com

# UPDATE
# @Time   : 2020/10/04, 2021/3/2, 2021/2/17, 2021/6/30, 2022/7/6
# @Author : Shanlei Mu, Yupeng Hou, Jiawei Guan, Xingyu Pan, Gaowei Zhang
# @Email  : slmu@ruc.edu.cn, houyupeng@ruc.edu.cn, Guanjw@ruc.edu.cn, xy_pan@foxmail.com, zgw15630559577@163.com

"""
recbole.config.configurator
################################
"""

import re
import os
import sys
import yaml
from logging import getLogger
from typing import Literal

from recbole.evaluator import metric_types, smaller_metrics
from recbole.utils import (
    get_model,
    Enum,
    EvaluatorType,
    ModelType,
    InputType,
    general_arguments,
    training_arguments,
    evaluation_arguments,
    dataset_arguments,
    set_color,
)


class Config(object):
    """Configurator module that load the defined parameters.

    Configurator module will first load the default parameters from the fixed properties in RecBole and then
    load parameters from the external input.

    External input supports three kind of forms: config file, command line and parameter dictionaries.

    - config file: It's a file that record the parameters to be modified or added. It should be in ``yaml`` format,
      e.g. a config file is 'example.yaml', the content is:

        learning_rate: 0.001

        train_batch_size: 2048

    - command line: It should be in the format as '---learning_rate=0.001'

    - parameter dictionaries: It should be a dict, where the key is parameter name and the value is parameter value,
      e.g. config_dict = {'learning_rate': 0.001}

    Configuration module allows the above three kind of external input format to be used together,
    the priority order is as following:

    command line > parameter dictionaries > config file

    e.g. If we set learning_rate=0.01 in config file, learning_rate=0.02 in command line,
    learning_rate=0.03 in parameter dictionaries.

    Finally the learning_rate is equal to 0.02.
    """

    def __init__(
        self, model=None, dataset=None, config_file_list=None, config_dict=None
    ):
        """
        Args:
            model (str/AbstractRecommender): the model name or the model class, default is None, if it is None, config
            will search the parameter 'model' from the external input as the model name or model class.
            dataset (str): the dataset name, default is None, if it is None, config will search the parameter 'dataset'
            from the external input as the dataset name.
            config_file_list (list of str): the external config file, it allows multiple config files, default is None.
            config_dict (dict): the external parameter dictionaries, default is None.
        """
        self.compatibility_settings()
        self._init_parameters_category()
        self.yaml_loader = self._build_yaml_loader()
        self.file_config_dict = self._load_config_files(config_file_list)
        self.variable_config_dict = self._load_variable_config_dict(config_dict)
        self.cmd_config_dict = self._load_cmd_line()
        self._merge_external_config_dict()

        self.model, self.model_class, self.dataset = self._get_model_and_dataset(
            model, dataset
        )
        self._load_internal_config_dict(self.model, self.model_class, self.dataset)
        self.final_config_dict = self._get_final_config_dict()
        self._set_default_parameters()
        self._init_device()
        self._set_train_neg_sample_args()
        self._set_eval_neg_sample_args("valid")
        self._set_eval_neg_sample_args("test")

    def _init_parameters_category(self):
        self.parameters = dict()
        self.parameters["General"] = general_arguments
        self.parameters["Training"] = training_arguments
        self.parameters["Evaluation"] = evaluation_arguments
        self.parameters["Dataset"] = dataset_arguments

    def _build_yaml_loader(self):
        loader = yaml.FullLoader
        loader.add_implicit_resolver(
            "tag:yaml.org,2002:float",
            re.compile(
                """^(?:
             [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
            |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
            |\\.[0-9_]+(?:[eE][-+][0-9]+)?
            |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
            |[-+]?\\.(?:inf|Inf|INF)
            |\\.(?:nan|NaN|NAN))$""",
                re.X,
            ),
            list("-+0123456789."),
        )
        return loader

    def _convert_config_dict(self, config_dict):
        r"""This function convert the str parameters to their original type."""
        for key in config_dict:
            param = config_dict[key]
            if not isinstance(param, str):
                continue
            try:
                value = eval(param)
                if value is not None and not isinstance(
                    value, (str, int, float, list, tuple, dict, bool, Enum)
                ):
                    value = param
            except (NameError, SyntaxError, TypeError):
                if isinstance(param, str):
                    if param.lower() == "true":
                        value = True
                    elif param.lower() == "false":
                        value = False
                    else:
                        value = param
                else:
                    value = param
            config_dict[key] = value
        return config_dict

    def _load_config_files(self, file_list):
        file_config_dict = dict()
        if file_list:
            for file in file_list:
                with open(file, "r", encoding="utf-8") as f:
                    file_config_dict.update(
                        yaml.load(f.read(), Loader=self.yaml_loader)
                    )
        return file_config_dict

    def _load_variable_config_dict(self, config_dict):
        # HyperTuning may set the parameters such as mlp_hidden_size in NeuMF in the format of ['[]', '[]']
        # then config_dict will receive a str '[]', but indeed it's a list []
        # temporarily use _convert_config_dict to solve this problem
        return self._convert_config_dict(config_dict) if config_dict else dict()

    def _load_cmd_line(self):
        r"""Read parameters from command line and convert it to str."""
        cmd_config_dict = dict()
        unrecognized_args = []
        if "ipykernel_launcher" not in sys.argv[0]:
            for arg in sys.argv[1:]:
                if not arg.startswith("--") or len(arg[2:].split("=")) != 2:
                    unrecognized_args.append(arg)
                    continue
                cmd_arg_name, cmd_arg_value = arg[2:].split("=")
                if (
                    cmd_arg_name in cmd_config_dict
                    and cmd_arg_value != cmd_config_dict[cmd_arg_name]
                ):
                    raise SyntaxError(
                        "There are duplicate commend arg '%s' with different value."
                        % arg
                    )
                else:
                    cmd_config_dict[cmd_arg_name] = cmd_arg_value
        if len(unrecognized_args) > 0:
            logger = getLogger()
            logger.warning(
                "command line args [{}] will not be used in RecBole".format(
                    " ".join(unrecognized_args)
                )
            )
        cmd_config_dict = self._convert_config_dict(cmd_config_dict)
        return cmd_config_dict

    def _merge_external_config_dict(self):
        external_config_dict = dict()
        external_config_dict.update(self.file_config_dict)
        external_config_dict.update(self.variable_config_dict)
        external_config_dict.update(self.cmd_config_dict)
        self.external_config_dict = external_config_dict

    def _get_model_and_dataset(self, model, dataset):
        if model is None:
            try:
                model = self.external_config_dict["model"]
            except KeyError:
                raise KeyError(
                    "model need to be specified in at least one of the these ways: "
                    "[model variable, config file, config dict, command line] "
                )
        if not isinstance(model, str):
            final_model_class = model
            final_model = model.__name__
        else:
            final_model = model
            final_model_class = get_model(final_model)

        if dataset is None:
            try:
                final_dataset = self.external_config_dict["dataset"]
            except KeyError:
                raise KeyError(
                    "dataset need to be specified in at least one of the these ways: "
                    "[dataset variable, config file, config dict, command line] "
                )
        else:
            final_dataset = dataset

        return final_model, final_model_class, final_dataset

    def _update_internal_config_dict(self, file):
        with open(file, "r", encoding="utf-8") as f:
            config_dict = yaml.load(f.read(), Loader=self.yaml_loader)
            if config_dict is not None:
                self.internal_config_dict.update(config_dict)
        return config_dict

    def _load_internal_config_dict(self, model, model_class, dataset):
        current_path = os.path.dirname(os.path.realpath(__file__))
        overall_init_file = os.path.join(current_path, "../properties/overall.yaml")
        model_init_file = os.path.join(
            current_path, "../properties/model/" + model + ".yaml"
        )
        sample_init_file = os.path.join(
            current_path, "../properties/dataset/sample.yaml"
        )
        dataset_init_file = os.path.join(
            current_path, "../properties/dataset/" + dataset + ".yaml"
        )

        quick_start_config_path = os.path.join(
            current_path, "../properties/quick_start_config/"
        )
        context_aware_init = os.path.join(quick_start_config_path, "context-aware.yaml")
        context_aware_on_ml_100k_init = os.path.join(
            quick_start_config_path, "context-aware_ml-100k.yaml"
        )
        DIN_init = os.path.join(quick_start_config_path, "sequential_DIN.yaml")
        DIN_on_ml_100k_init = os.path.join(
            quick_start_config_path, "sequential_DIN_on_ml-100k.yaml"
        )
        sequential_init = os.path.join(quick_start_config_path, "sequential.yaml")
        special_sequential_on_ml_100k_init = os.path.join(
            quick_start_config_path, "special_sequential_on_ml-100k.yaml"
        )
        sequential_embedding_model_init = os.path.join(
            quick_start_config_path, "sequential_embedding_model.yaml"
        )
        knowledge_base_init = os.path.join(
            quick_start_config_path, "knowledge_base.yaml"
        )

        self.internal_config_dict = dict()
        for file in [
            overall_init_file,
            model_init_file,
            sample_init_file,
            dataset_init_file,
        ]:
            if os.path.isfile(file):
                config_dict = self._update_internal_config_dict(file)
                if file == dataset_init_file:
                    self.parameters["Dataset"] += [
                        key
                        for key in config_dict.keys()
                        if key not in self.parameters["Dataset"]
                    ]

        self.internal_config_dict["MODEL_TYPE"] = model_class.type
        if self.internal_config_dict["MODEL_TYPE"] == ModelType.GENERAL:
            pass
        elif self.internal_config_dict["MODEL_TYPE"] in {
            ModelType.CONTEXT,
            ModelType.DECISIONTREE,
        }:
            self._update_internal_config_dict(context_aware_init)
            if dataset == "ml-100k":
                self._update_internal_config_dict(context_aware_on_ml_100k_init)
        elif self.internal_config_dict["MODEL_TYPE"] == ModelType.SEQUENTIAL:
            if model in ["DIN", "DIEN"]:
                self._update_internal_config_dict(DIN_init)
                if dataset == "ml-100k":
                    self._update_internal_config_dict(DIN_on_ml_100k_init)
            elif model in ["GRU4RecKG", "KSR"]:
                self._update_internal_config_dict(sequential_embedding_model_init)
            else:
                self._update_internal_config_dict(sequential_init)
                if dataset == "ml-100k" and model in [
                    "GRU4RecF",
                    "SASRecF",
                    "FDSA",
                    "S3Rec",
                ]:
                    self._update_internal_config_dict(
                        special_sequential_on_ml_100k_init
                    )

        elif self.internal_config_dict["MODEL_TYPE"] == ModelType.KNOWLEDGE:
            self._update_internal_config_dict(knowledge_base_init)

    def _get_final_config_dict(self):
        final_config_dict = dict()
        final_config_dict.update(self.internal_config_dict)
        final_config_dict.update(self.external_config_dict)
        return final_config_dict

    def _set_default_parameters(self):
        self.final_config_dict["dataset"] = self.dataset
        self.final_config_dict["model"] = self.model
        if self.dataset == "ml-100k":
            current_path = os.path.dirname(os.path.realpath(__file__))
            self.final_config_dict["data_path"] = os.path.join(
                current_path, "../dataset_example/" + self.dataset
            )
        else:
            self.final_config_dict["data_path"] = os.path.join(
                self.final_config_dict["data_path"], self.dataset
            )

        if hasattr(self.model_class, "input_type"):
            self.final_config_dict["MODEL_INPUT_TYPE"] = self.model_class.input_type
        elif "loss_type" in self.final_config_dict:
            if self.final_config_dict["loss_type"] in ["CE"]:
                if (
                    self.final_config_dict["MODEL_TYPE"] == ModelType.SEQUENTIAL
                    and self.final_config_dict.get("train_neg_sample_args", None)
                    is not None
                ):
                    raise ValueError(
                        f"train_neg_sample_args [{self.final_config_dict['train_neg_sample_args']}] should be None "
                        f"when the loss_type is CE."
                    )
                self.final_config_dict["MODEL_INPUT_TYPE"] = InputType.POINTWISE
            elif self.final_config_dict["loss_type"] in ["BPR"]:
                self.final_config_dict["MODEL_INPUT_TYPE"] = InputType.PAIRWISE
        else:
            raise ValueError(
                "Either Model has attr 'input_type',"
                "or arg 'loss_type' should exist in config."
            )

        metrics = self.final_config_dict["metrics"]
        if isinstance(metrics, str):
            self.final_config_dict["metrics"] = [metrics]

        eval_type = set()
        for metric in self.final_config_dict["metrics"]:
            if metric.lower() in metric_types:
                eval_type.add(metric_types[metric.lower()])
            else:
                raise NotImplementedError(f"There is no metric named '{metric}'")
        if len(eval_type) > 1:
            raise RuntimeError(
                "Ranking metrics and value metrics can not be used at the same time."
            )
        self.final_config_dict["eval_type"] = eval_type.pop()

        if (
            self.final_config_dict["MODEL_TYPE"] == ModelType.SEQUENTIAL
            and not self.final_config_dict["repeatable"]
        ):
            raise ValueError(
                "Sequential models currently only support repeatable recommendation, "
                "please set `repeatable` as `True`."
            )

        valid_metric = self.final_config_dict["valid_metric"].split("@")[0]
        self.final_config_dict["valid_metric_bigger"] = (
            False if valid_metric.lower() in smaller_metrics else True
        )

        topk = self.final_config_dict["topk"]
        if isinstance(topk, (int, list)):
            if isinstance(topk, int):
                topk = [topk]
            for k in topk:
                if k <= 0:
                    raise ValueError(
                        f"topk must be a positive integer or a list of positive integers, but get `{k}`"
                    )
            self.final_config_dict["topk"] = topk
        else:
            raise TypeError(f"The topk [{topk}] must be a integer, list")

        if "additional_feat_suffix" in self.final_config_dict:
            ad_suf = self.final_config_dict["additional_feat_suffix"]
            if isinstance(ad_suf, str):
                self.final_config_dict["additional_feat_suffix"] = [ad_suf]

        # train_neg_sample_args checking
        default_train_neg_sample_args = {
            "distribution": "uniform",
            "sample_num": 1,
            "alpha": 1.0,
            "dynamic": False,
            "candidate_num": 0,
        }

        if (
            self.final_config_dict.get("neg_sampling") is not None
            or self.final_config_dict.get("training_neg_sample_num") is not None
        ):
            logger = getLogger()
            logger.warning(
                "Warning: Parameter 'neg_sampling' or 'training_neg_sample_num' has been deprecated in the new version, "
                "please use 'train_neg_sample_args' instead and check the API documentation for proper usage."
            )

        if self.final_config_dict.get("train_neg_sample_args") is not None:
            if not isinstance(self.final_config_dict["train_neg_sample_args"], dict):
                raise ValueError(
                    f"train_neg_sample_args:[{self.final_config_dict['train_neg_sample_args']}] should be a dict."
                )
            for op_args in default_train_neg_sample_args:
                if op_args not in self.final_config_dict["train_neg_sample_args"]:
                    self.final_config_dict["train_neg_sample_args"][op_args] = (
                        default_train_neg_sample_args[op_args]
                    )

        # eval_args checking
        default_eval_args = {
            "split": {"RS": [0.8, 0.1, 0.1]},
            "order": "RO",
            "group_by": "user",
            "mode": {"valid": "full", "test": "full"},
        }
        if not isinstance(self.final_config_dict["eval_args"], dict):
            raise ValueError(
                f"eval_args:[{self.final_config_dict['eval_args']}] should be a dict."
            )

        default_eval_args.update(self.final_config_dict["eval_args"])

        mode = default_eval_args["mode"]
        # backward compatible
        if isinstance(mode, str):
            default_eval_args["mode"] = {"valid": mode, "test": mode}

        # in case there is only one key in `mode`, e.g., mode: {'valid': 'uni100'} or mode: {'test': 'full'}
        if isinstance(mode, dict):
            default_mode = mode.get("valid", mode.get("test", "full"))
            default_eval_args["mode"] = {
                "valid": mode.get("valid", default_mode),
                "test": mode.get("test", default_mode),
            }

        self.final_config_dict["eval_args"] = default_eval_args
        if (
            self.final_config_dict["eval_type"] == EvaluatorType.VALUE
            and "full" in self.final_config_dict["eval_args"]["mode"].values()
        ):
            raise NotImplementedError(
                "Full sort evaluation do not match value-based metrics!"
            )

    def _init_device(self):
        if isinstance(self.final_config_dict["gpu_id"], tuple):
            self.final_config_dict["gpu_id"] = ",".join(
                map(str, list(self.final_config_dict["gpu_id"]))
            )
        else:
            self.final_config_dict["gpu_id"] = str(self.final_config_dict["gpu_id"])
        gpu_id = self.final_config_dict["gpu_id"]
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
        import torch

        if "local_rank" not in self.final_config_dict:
            self.final_config_dict["single_spec"] = True
            self.final_config_dict["local_rank"] = 0
            self.final_config_dict["device"] = (
                torch.device("cpu")
                if len(gpu_id) == 0 or not torch.cuda.is_available()
                else torch.device("cuda")
            )
        else:
            assert len(gpu_id.split(",")) >= self.final_config_dict["nproc"]
            torch.distributed.init_process_group(
                backend="nccl",
                rank=self.final_config_dict["local_rank"]
                + self.final_config_dict["offset"],
                world_size=self.final_config_dict["world_size"],
                init_method="tcp://"
                + self.final_config_dict["ip"]
                + ":"
                + str(self.final_config_dict["port"]),
            )
            self.final_config_dict["device"] = torch.device(
                "cuda", self.final_config_dict["local_rank"]
            )
            self.final_config_dict["single_spec"] = False
            torch.cuda.set_device(self.final_config_dict["local_rank"])
            if self.final_config_dict["local_rank"] != 0:
                self.final_config_dict["state"] = "error"
                self.final_config_dict["show_progress"] = False
                self.final_config_dict["verbose"] = False

    def _set_train_neg_sample_args(self):
        train_neg_sample_args = self.final_config_dict.get("train_neg_sample_args")
        if train_neg_sample_args is None or train_neg_sample_args == "None":
            self.final_config_dict["train_neg_sample_args"] = {
                "distribution": "none",
                "sample_num": "none",
                "alpha": "none",
                "dynamic": False,
                "candidate_num": 0,
            }
        else:
            if not isinstance(train_neg_sample_args, dict):
                raise ValueError(
                    f"train_neg_sample_args:[{train_neg_sample_args}] should be a dict."
                )

            distribution = train_neg_sample_args["distribution"]
            if distribution is None or distribution == "None":
                self.final_config_dict["train_neg_sample_args"] = {
                    "distribution": "none",
                    "sample_num": "none",
                    "alpha": "none",
                    "dynamic": False,
                    "candidate_num": 0,
                }
            elif distribution not in ["uniform", "popularity"]:
                raise ValueError(
                    f"The distribution [{distribution}] of train_neg_sample_args "
                    f"should in ['uniform', 'popularity']"
                )

    def _set_eval_neg_sample_args(self, phase: Literal["valid", "test"]):
        eval_mode = self.final_config_dict["eval_args"]["mode"][phase]
        if not isinstance(eval_mode, str):
            raise ValueError(f"mode [{eval_mode}] in eval_args should be a str.")
        if eval_mode == "labeled":
            eval_neg_sample_args = {"distribution": "none", "sample_num": "none"}
        elif eval_mode == "full":
            eval_neg_sample_args = {"distribution": "uniform", "sample_num": "none"}
        elif eval_mode[0:3] == "uni":
            sample_num = int(eval_mode[3:])
            eval_neg_sample_args = {"distribution": "uniform", "sample_num": sample_num}
        elif eval_mode[0:3] == "pop":
            sample_num = int(eval_mode[3:])
            eval_neg_sample_args = {
                "distribution": "popularity",
                "sample_num": sample_num,
            }
        else:
            raise ValueError(f"the mode [{eval_mode}] in eval_args is not supported.")
        self.final_config_dict[f"{phase}_neg_sample_args"] = eval_neg_sample_args

    def __setitem__(self, key, value):
        if not isinstance(key, str):
            raise TypeError("index must be a str.")
        self.final_config_dict[key] = value

    def __getattr__(self, item):
        if "final_config_dict" not in self.__dict__:
            raise AttributeError(
                f"'Config' object has no attribute 'final_config_dict'"
            )
        if item in self.final_config_dict:
            return self.final_config_dict[item]
        raise AttributeError(f"'Config' object has no attribute '{item}'")

    def __getitem__(self, item):
        return self.final_config_dict.get(item)

    def __contains__(self, key):
        if not isinstance(key, str):
            raise TypeError("index must be a str.")
        return key in self.final_config_dict

    def __str__(self):
        args_info = "\n"
        for category in self.parameters:
            args_info += set_color(category + " Hyper Parameters:\n", "pink")
            args_info += "\n".join(
                [
                    (
                        set_color("{}", "cyan") + " =" + set_color(" {}", "yellow")
                    ).format(arg, value)
                    for arg, value in self.final_config_dict.items()
                    if arg in self.parameters[category]
                ]
            )
            args_info += "\n\n"

        args_info += set_color("Other Hyper Parameters: \n", "pink")
        args_info += "\n".join(
            [
                (set_color("{}", "cyan") + " = " + set_color("{}", "yellow")).format(
                    arg, value
                )
                for arg, value in self.final_config_dict.items()
                if arg
                not in {_ for args in self.parameters.values() for _ in args}.union(
                    {"model", "dataset", "config_files"}
                )
            ]
        )
        args_info += "\n\n"
        return args_info

    def __repr__(self):
        return self.__str__()

    def compatibility_settings(self):
        import numpy as np

        np.bool = np.bool_
        np.int = np.int_
        np.float = np.float64
        np.complex = np.complex128
        np.object = np.object_
        np.str = np.str_
        np.long = np.int_
        np.unicode = np.str_
