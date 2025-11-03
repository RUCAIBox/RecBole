#!/usr/bin/env python3
"""比较不同配置下的数据集大小"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recbole.config import Config
from recbole.data.utils import create_dataset

configs = [
    # 默认配置（export_internal_item_mapping.py 使用的）
    {
        "name": "默认配置（BPR）",
        "model": "BPR",
        "dataset": "Amazon_Beauty",
        "config_files": [],
        "config_dict": {}
    },
    # SASRec 配置
    {
        "name": "SASRec 配置",
        "model": "SASRec",
        "dataset": "Amazon_Beauty", 
        "config_files": [],
        "config_dict": {}
    },
    # SASRecAlign 配置
    {
        "name": "SASRecAlign 配置",
        "model": "SASRecAlign",
        "dataset": "Amazon_Beauty",
        "config_files": ["sasrec_base_plain.yaml"],
        "config_dict": {}
    }
]

print("=== 不同配置下的数据集大小 ===\n")

for cfg_info in configs:
    try:
        config = Config(
            model=cfg_info["model"],
            dataset=cfg_info["dataset"],
            config_file_list=cfg_info["config_files"] if cfg_info["config_files"] else None,
            config_dict=cfg_info["config_dict"]
        )
        
        dataset = create_dataset(config)
        n_items = dataset.num(dataset.iid_field)
        
        print(f"{cfg_info['name']}:")
        print(f"  Model: {cfg_info['model']}")
        print(f"  Config files: {cfg_info['config_files'] or 'None'}")
        print(f"  n_items: {n_items}")
        
        # 检查一些可能影响item数量的配置
        print(f"  load_col: {config.get('load_col', 'default')}")
        print(f"  min_item_inter_num: {config.get('min_item_inter_num', 'default')}")
        print(f"  min_user_inter_num: {config.get('min_user_inter_num', 'default')}")
        print()
        
    except Exception as e:
        print(f"{cfg_info['name']}: 错误 - {e}\n")
