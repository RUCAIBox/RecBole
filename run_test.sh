#!/bin/bash


python -m pytest -v tests/metrics
echo "metrics tests finished"

python -m pytest -v tests/config/test_config.py
python -m pytest -v tests/config/test_overall.py
export PYTHONPATH=.
python tests/config/test_command_line.py --use_gpu=False --valid_metric=Recall@10  --metrics=['Recall@10'] --epochs=200  --learning_rate=0.3
echo "config tests finished"

python -m pytest -v tests/evaluation_setting
echo "evaluation_setting tests finished"

python -m pytest -v tests/model/test_model_auto.py
python -m pytest -v tests/model/test_model_manual.py
echo "model tests finished"

python -m pytest -v tests/data/test_dataset.py
python -m pytest -v tests/data/test_dataloader.py
echo "data tests finished"