#!/bin/bash
# 临时修复 RecBole 导入问题的脚本

BASE_DIR="/home/charlie/project/RecBole"
cd $BASE_DIR

echo "=== 修复 RecBole 导入问题 ==="

# 方法1：安装缺失的依赖（最简单）
echo "方法1：安装 lightgbm 依赖..."
pip install lightgbm

# 如果上面失败，可以使用方法2
# 方法2：临时禁用 exlib_recommender 模块
# echo "方法2：临时重命名 exlib_recommender 目录..."
# mv recbole/model/exlib_recommender recbole/model/exlib_recommender.bak

echo "✓ 修复完成"
echo "现在可以运行 SASRec 实验了"
