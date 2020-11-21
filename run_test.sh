#!/bin/bash


python -m pytest -v tests/metrics
printf "metrics tests finished\n"
python -m pytest -v tests/config/test_config.py
python -m pytest -v tests/config/test_overall.py
export PYTHONPATH=.
python tests/config/test_command_line.py --use_gpu=False --valid_metric=Recall@10 --split_ratio=[0.7,0.2,0.1] --metrics=['Recall@10'] --epochs=200 --eval_setting='LO_RS' --learning_rate=0.3
printf "config tests finished\n"
python -m pytest -v tests/evaluation_setting
printf "evaluation_setting tests finished\n"
python -m pytest -v tests/model/test_model_auto.py
python -m pytest -v tests/model/test_model_manual.py
printf "model tests finished\n"

