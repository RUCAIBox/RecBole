#!/usr/bin/env python3
"""
07_summarize_results.py - 汇总所有实验结果并生成报告
"""

import os
import re
import pandas as pd
from glob import glob
import json

def extract_metrics(log_file):
    """从日志文件中提取评测指标"""
    metrics = {}
    
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 提取test结果 - 适配RecBole的输出格式
        # 查找 "test result" 后面的指标
        test_section = re.search(r'test result.*?({.*?})', content, re.DOTALL)
        if test_section:
            # 尝试解析JSON格式的结果
            try:
                test_dict = eval(test_section.group(1))
                for key in ['Recall@5', 'Recall@10', 'Recall@20', 
                           'MRR@5', 'MRR@10', 'MRR@20',
                           'NDCG@5', 'NDCG@10', 'NDCG@20',
                           'Hit@5', 'Hit@10', 'Hit@20']:
                    if key in test_dict:
                        metrics[key] = test_dict[key]
            except:
                pass
        
        # 备选方案：使用正则表达式提取
        if not metrics:
            patterns = {
                'Recall@10': r'Recall@10\s*:\s*([\d.]+)',
                'MRR@10': r'MRR@10\s*:\s*([\d.]+)',
                'NDCG@10': r'NDCG@10\s*:\s*([\d.]+)',
                'Hit@10': r'Hit@10\s*:\s*([\d.]+)'
            }
            
            for metric_name, pattern in patterns.items():
                match = re.search(pattern, content)
                if match:
                    metrics[metric_name] = float(match.group(1))
        
        # 提取训练时间
        time_match = re.search(r'Total training time:\s*([\d.]+)\s*s', content)
        if time_match:
            metrics['Training_Time'] = float(time_match.group(1))
        
        # 提取最佳epoch
        epoch_match = re.search(r'best valid.*?epoch\s*=\s*(\d+)', content, re.IGNORECASE)
        if epoch_match:
            metrics['Best_Epoch'] = int(epoch_match.group(1))
            
    except Exception as e:
        print(f"Error processing {log_file}: {e}")
    
    return metrics

def get_experiment_name(filename):
    """从文件名解析实验名称"""
    name_map = {
        'exp1_baseline_sasrec': 'SASRec (Baseline)',
        'exp2_sasrec_base': 'SASRec + Base',
        'exp3_sasrec_base_cross': 'SASRec + Base + Cross',
        'exp4_sasrec_base_cross_align': 'SASRec + Base + Cross + Align',
        'exp5_sasrec_llm': 'SASRec + LLM',
        'exp6_sasrec_llm_cross': 'SASRec + LLM + Cross',
        'exp7_sasrec_llm_cross_align': 'SASRec + LLM + Cross + Align'
    }
    
    base_name = os.path.basename(filename).replace('.log', '')
    return name_map.get(base_name, base_name)

def main():
    """主函数"""
    log_dir = 'results/sasrec_experiments'
    
    if not os.path.exists(log_dir):
        print(f"错误: 找不到日志目录 {log_dir}")
        return
    
    # 收集主实验结果
    print("=== 收集实验结果 ===")
    results = []
    
    # 主实验日志
    main_logs = glob(f'{log_dir}/exp*.log')
    
    for log_file in sorted(main_logs):
        print(f"处理: {os.path.basename(log_file)}")
        metrics = extract_metrics(log_file)
        
        if metrics:
            exp_name = get_experiment_name(log_file)
            results.append({
                'Experiment': exp_name,
                'Log_File': os.path.basename(log_file),
                **metrics
            })
    
    # 创建DataFrame
    if results:
        df = pd.DataFrame(results)
        
        # 按实验名称排序
        df = df.sort_values('Experiment')
        
        # 保存完整结果
        output_file = f'{log_dir}/summary_results.csv'
        df.to_csv(output_file, index=False)
        print(f"\n完整结果已保存到: {output_file}")
        
        # 打印主要指标表格
        print("\n=== 主要实验结果 (Test集) ===")
        main_metrics = ['Experiment', 'Recall@10', 'MRR@10', 'NDCG@10', 'Hit@10']
        display_cols = [col for col in main_metrics if col in df.columns]
        
        if len(display_cols) > 1:
            # 格式化数值
            display_df = df[display_cols].copy()
            for col in display_cols[1:]:
                if col in display_df.columns:
                    display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
            
            print(display_df.to_string(index=False))
        
        # 计算提升百分比
        print("\n=== 相对基线的提升 ===")
        if 'SASRec (Baseline)' in df['Experiment'].values and 'MRR@10' in df.columns:
            baseline_mrr = df[df['Experiment'] == 'SASRec (Baseline)']['MRR@10'].values[0]
            
            for _, row in df.iterrows():
                if row['Experiment'] != 'SASRec (Baseline)' and pd.notna(row.get('MRR@10')):
                    improvement = (row['MRR@10'] - baseline_mrr) / baseline_mrr * 100
                    print(f"{row['Experiment']}: +{improvement:.2f}%")
        
        # 训练效率统计
        if 'Training_Time' in df.columns:
            print("\n=== 训练时间统计 ===")
            time_df = df[['Experiment', 'Training_Time', 'Best_Epoch']].dropna()
            if not time_df.empty:
                print(time_df.to_string(index=False))
    else:
        print("\n没有找到有效的实验结果")
    
    # 处理超参数搜索结果
    hyperparam_dir = f'{log_dir}/hyperparam'
    if os.path.exists(hyperparam_dir):
        print("\n\n=== 超参数搜索结果 ===")
        hp_results = []
        
        for log_file in glob(f'{hyperparam_dir}/*.log'):
            metrics = extract_metrics(log_file)
            if metrics and 'MRR@10' in metrics:
                # 从文件名解析参数
                filename = os.path.basename(log_file)
                aw_match = re.search(r'aw([\d.]+)', filename)
                t_match = re.search(r't([\d.]+)', filename)
                
                if aw_match and t_match:
                    hp_results.append({
                        'Type': 'LLM' if 'llm' in filename else 'Base',
                        'Alignment_Weight': float(aw_match.group(1)),
                        'Temperature': float(t_match.group(1)),
                        'MRR@10': metrics['MRR@10']
                    })
        
        if hp_results:
            hp_df = pd.DataFrame(hp_results)
            hp_df = hp_df.sort_values('MRR@10', ascending=False)
            
            print("Top 5 配置:")
            print(hp_df.head().to_string(index=False))
            
            # 保存超参数结果
            hp_df.to_csv(f'{hyperparam_dir}/hyperparam_results.csv', index=False)

if __name__ == "__main__":
    main()
