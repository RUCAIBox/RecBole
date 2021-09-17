#!/bin/bash

conda-build --python 3.7 .
printf "python 3.7 version is released \n"
conda-build --python 3.8 .
printf "python 3.8 version is released \n"
conda-build --python 3.9 .
printf "python 3.9 version is released \n"
