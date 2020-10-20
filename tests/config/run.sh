#!/bin/bash


python -m unittest

python test_command_line.py --use_gpu=False --valid_metric=Recall@10 --split_ratio=[0.7,0.2,0.1] \
--metrics=['Recall@10'] \
--epochs=200 --eval_setting='LO_RS' --learning_rate=0.3