#!/bin/bash
# 运行正则化系列实验 (exp8-exp11)
# 验证不同正则化技术对缓解过拟合的效果

echo "================================================"
echo "开始运行正则化系列实验"
echo "实验目的：验证门控、正则化和dropout对缓解过拟合的效果"
echo "开始时间: $(date)"
echo "================================================"

# 定义实验列表
EXPERIMENTS=(
    "exp8_sasrec_gate_l2_reg:门控L2正则化"
    "exp9_sasrec_gate_entropy_reg:门控熵正则化"
    "exp10_sasrec_cross_dropout:交叉网络dropout"
    "exp11_sasrec_combined_reg:组合正则化"
)

# 运行每个实验
for exp_info in "${EXPERIMENTS[@]}"; do
    exp_name="${exp_info%%:*}"
    exp_desc="${exp_info#*:}"
    
    echo ""
    echo "------------------------------------------------"
    echo "运行实验: $exp_name"
    echo "描述: $exp_desc"
    echo "------------------------------------------------"
    
    bash scripts/${exp_name}.sh
    
    echo "实验 $exp_name 完成"
done

echo ""
echo "================================================"
echo "所有正则化实验完成"
echo "结束时间: $(date)"
echo "================================================"

# 生成结果汇总
echo ""
echo "实验结果汇总："
echo "------------------------------------------------"
for exp_info in "${EXPERIMENTS[@]}"; do
    exp_name="${exp_info%%:*}"
    log_file="results/sasrec_experiments/${exp_name}.log"
    if [ -f "$log_file" ]; then
        echo ""
        echo "$exp_name 测试结果:"
        grep -A 5 "test result" "$log_file" | tail -6 || echo "未找到测试结果"
    fi
done
