#!/bin/bash

conda-build --python 3.6 .
printf "python 3.6 version is released \n"
conda-build --python 3.7 .
printf "python 3.7 version is released \n"
conda-build --python 3.8 .
printf "python 3.8 version is released \n"
