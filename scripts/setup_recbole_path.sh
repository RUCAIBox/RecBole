#!/bin/bash
# 设置RecBole模型路径

BASE_DIR="/home/charlie/project/RecBole"
cd $BASE_DIR

echo "=== 设置RecBole模型文件 ==="

# 创建RecBole期望的文件名（小写，无下划线）
if [ ! -f "recbole/model/sequential_recommender/sasrecalign.py" ]; then
    echo "创建 sasrecalign.py ..."
    cp recbole/model/sequential_recommender/sasrec_align.py \
       recbole/model/sequential_recommender/sasrecalign.py
    echo "✓ 文件创建成功"
else
    echo "✓ sasrecalign.py 已存在"
fi

echo ""
echo "现在可以运行SASRecAlign实验了"
