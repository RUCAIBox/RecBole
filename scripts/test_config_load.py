#!/usr/bin/env python3
"""简单测试配置加载"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recbole.config import Config

# 测试配置加载
config = Config(
    model="SASRecAlign",
    dataset="Amazon_Beauty", 
    config_file_list=["sasrec_base_plain.yaml"],
    config_dict={"device": "cpu"}
)

print("=== 配置内容 ===")
# Config对象使用属性访问，不是字典
print(f"item_text_emb_path_base: {getattr(config, 'item_text_emb_path_base', 'NOT SET')}")
print(f"item_text_emb_path_llm: {getattr(config, 'item_text_emb_path_llm', 'NOT SET')}")
print(f"use_cross: {getattr(config, 'use_cross', 'NOT SET')}")
print(f"use_llm: {getattr(config, 'use_llm', 'NOT SET')}")

# 检查文件
base_path = getattr(config, 'item_text_emb_path_base', None)
if base_path:
    print(f"\nBase路径: {base_path}")
    print(f"当前目录: {os.getcwd()}")
    print(f"绝对路径: {os.path.abspath(base_path)}")
    print(f"文件存在: {os.path.exists(base_path)}")
    
    # 尝试不同的路径
    alt_paths = [
        base_path,
        os.path.join(os.getcwd(), base_path),
        f"/home/charlie/project/RecBole/{base_path}"
    ]
    
    print("\n尝试不同路径:")
    for p in alt_paths:
        print(f"  {p}: {os.path.exists(p)}")
