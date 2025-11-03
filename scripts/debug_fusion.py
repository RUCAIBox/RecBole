#!/usr/bin/env python3
"""调试脚本：验证文本融合是否真的在工作"""

import sys
import os
# 添加 RecBole 到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from recbole.config import Config
from recbole.data.utils import create_dataset
from recbole.model.sequential_recommender.sasrec_align import SASRecAlign

def check_fusion_modules(config_file):
    """检查不同配置下的融合模块"""
    config = Config(
        model="SASRecAlign",
        dataset="Amazon_Beauty", 
        config_file_list=[config_file],
        config_dict={"device": "cpu"}
    )
    
    dataset = create_dataset(config)
    model = SASRecAlign(config, dataset)
    
    print(f"\n=== 配置文件: {config_file} ===")
    print(f"use_cross: {model.use_cross}")
    print(f"use_align: {model.use_align}")
    print(f"use_llm: {model.use_llm}")
    
    # 检查文本embeddings
    print(f"\n文本embedding状态:")
    print(f"- item_text_emb_base: {'已加载' if hasattr(model, 'item_text_emb_base') and model.item_text_emb_base is not None else '未加载'}")
    print(f"- item_text_emb_llm: {'已加载' if hasattr(model, 'item_text_emb_llm') and model.item_text_emb_llm is not None else '未加载'}")
    print(f"- _text_mode: {model._text_mode if hasattr(model, '_text_mode') else 'N/A'}")
    
    # 检查维度
    base_dim = model.item_text_emb_base.shape[1] if hasattr(model, 'item_text_emb_base') and model.item_text_emb_base is not None else 0
    llm_dim = model.item_text_emb_llm.shape[1] if hasattr(model, 'item_text_emb_llm') and model.item_text_emb_llm is not None else 0
    print(f"- base维度: {base_dim}")
    print(f"- llm维度: {llm_dim}")
    
    # 检查融合模块
    print(f"\n融合模块状态:")
    print(f"- item_fusion_cross: {'已创建' if model.item_fusion_cross is not None else '未创建'}")
    print(f"- item_fusion_deep: {'已创建' if model.item_fusion_deep is not None else '未创建'}")
    print(f"- item_fusion_predictor: {'已创建' if model.item_fusion_predictor is not None else '未创建'}")
    
    # 测试融合路径
    if model.item_fusion_predictor is not None:
        test_ids = torch.tensor([1, 2, 3])
        with torch.no_grad():
            # 获取原始 embeddings
            orig_emb = model.item_embedding(test_ids)
            # 获取融合后的 embeddings
            fused_emb = model._get_fused_item_embeddings(test_ids)
            
            # 检查是否不同
            is_different = not torch.allclose(orig_emb, fused_emb)
            print(f"\n融合测试:")
            print(f"- 原始 embedding 形状: {orig_emb.shape}")
            print(f"- 融合 embedding 形状: {fused_emb.shape}")
            print(f"- 是否经过融合: {'是' if is_different else '否'}")
            
            if is_different:
                diff_norm = (fused_emb - orig_emb).norm().item()
                print(f"- 差异 L2 范数: {diff_norm:.6f}")

if __name__ == "__main__":
    # 测试不同配置
    configs = [
        "sasrec_base_plain.yaml",   # exp2: 无 cross
        "sasrec_base_cross.yaml"    # exp3: 有 cross
    ]
    
    for cfg in configs:
        check_fusion_modules(cfg)
