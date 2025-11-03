# SASRec实验执行指引

本指引详细说明了如何运行SASRec相关实验，包括基线模型、文本增强版本以及各种消融实验。

## InfoNCE对齐损失

SASRecAlign中已实现InfoNCE对比学习损失，用于对齐ID embeddings和文本embeddings：

- **实现位置**: `recbole/model/sequential_recommender/sasrec_align.py` 的 `_info_nce_align` 方法
- **关键参数**:
  - `alignment_weight`: InfoNCE损失权重（默认0.1）
  - `temperature`: 温度参数τ（默认0.07）
- **特点**:
  - 使用in-batch负样本
  - L2归一化后计算相似度
  - 支持detach防止梯度回传

验证InfoNCE实现：
```bash
python scripts/verify_infonce.py
```

## 目录结构

```
scripts/
├── 00_quick_test.sh              # 环境和配置快速测试
├── 01_prepare_embeddings.sh      # 生成item映射和Base embeddings
├── 02_exp_baseline.sh            # 基线SASRec实验（已废弃，使用experiments/目录）
├── 03_exp_base.sh                # Base embeddings相关实验（已废弃）
├── 04_exp_llm.sh                 # LLM embeddings相关实验（已废弃）
├── 05_hyperparameter_search.sh   # 超参数网格搜索
├── 06_multiple_seeds.sh          # 多随机种子实验
├── 07_summarize_results.py       # 结果汇总脚本
├── generate_qwen3_embeddings.sh  # GPU上生成Qwen3 embeddings
├── verify_infonce.py             # InfoNCE实现验证
├── experiments/                  # 独立实验脚本目录
│   ├── exp1_baseline_sasrec.sh       # 实验1: 基线
│   ├── exp2_sasrec_base.sh           # 实验2: +Base
│   ├── exp3_sasrec_base_cross.sh     # 实验3: +Base+Cross
│   ├── exp4_sasrec_base_cross_align.sh  # 实验4: +Base+Cross+Align
│   ├── exp5_sasrec_llm.sh            # 实验5: +LLM
│   ├── exp6_sasrec_llm_cross.sh      # 实验6: +LLM+Cross
│   ├── exp7_sasrec_llm_cross_align.sh   # 实验7: +LLM+Cross+Align
│   └── run_all_experiments.sh        # 运行所有实验
└── README.md                     # 本文档
```

## 执行步骤

### 0. 环境准备和测试

首先确保环境配置正确：

```bash
cd /home/charlie/project/recbole
bash scripts/00_quick_test.sh
```

这会检查：
- Python环境
- 必要的代码文件
- 数据集文件
- 运行一个快速的测试训练

### 1. 生成Item Embeddings

#### 1.1 生成Base embeddings（在CPU机器上）

```bash
bash scripts/01_prepare_embeddings.sh
```

这会生成：
- `dataset/Amazon_Beauty/item_index_mapping.csv` - item ID映射文件
- `dataset/Amazon_Beauty/item_text_emb.base.npy` - TF-IDF+SVD embeddings

#### 1.2 生成Qwen3 embeddings（需要GPU）

在GPU机器上运行：

```bash
bash scripts/generate_qwen3_embeddings.sh
```

或者直接运行您已经使用的命令：

```bash
nohup python tools/build_item_text_emb_qwen3_hf.py \
  --mapping dataset/Amazon_Beauty/item_index_mapping.csv \
  --model_name_or_path /home/charlie/project/qwen/Model \
  --output dataset/Amazon_Beauty/item_text_emb.qwen3.npy \
  --batch_size 8 --max_length 128 --dtype float16 \
  --project_dim 256 \
  --dataset Amazon_Beauty \
  --config recbole/properties/model/SASRecAlign.yaml \
  --prompt_template "[TITLE] {text}" \
  --device_map auto > get_qwen3.log 2>&1 &
```

### 2. 运行主要实验

按顺序运行以下实验：

#### 2.1 基线实验

```bash
bash scripts/02_exp_baseline.sh
```

运行标准SASRec模型（无文本特征）作为基线。

#### 2.2 Base embeddings实验

```bash
bash scripts/03_exp_base.sh
```

包含三个实验：
- 实验2: SASRec + Base
- 实验3: SASRec + Base + Cross
- 实验4: SASRec + Base + Cross + Align

#### 2.3 LLM embeddings实验

确保Qwen3 embeddings已生成，然后运行：

```bash
bash scripts/04_exp_llm.sh
```

包含三个实验：
- 实验5: SASRec + LLM
- 实验6: SASRec + LLM + Cross
- 实验7: SASRec + LLM + Cross + Align

### 3. 超参数调优（可选）

对alignment_weight和temperature进行网格搜索：

```bash
bash scripts/05_hyperparameter_search.sh
```

测试的参数组合：
- alignment_weight: [0.05, 0.1, 0.2]
- temperature: [0.05, 0.07]

### 4. 稳定性验证（可选）

使用5个不同的随机种子验证最佳配置的稳定性：

```bash
bash scripts/06_multiple_seeds.sh
```

### 5. 汇总结果

运行Python脚本生成结果汇总：

```bash
python scripts/07_summarize_results.py
```

这会生成：
- `results/sasrec_experiments/summary_results.csv` - 所有实验结果的CSV文件
- 在终端打印格式化的结果表格
- 计算相对于基线的提升百分比

## 监控和调试

### 查看实时日志

```bash
# 查看某个实验的日志
tail -f results/sasrec_experiments/exp1_baseline_sasrec.log

# 查看Qwen3生成进度
tail -f get_qwen3.log
```

### 检查实验状态

```bash
# 查看所有日志文件
ls -la results/sasrec_experiments/*.log

# 快速查看所有实验的测试结果
grep -n "test result" results/sasrec_experiments/exp*.log
```

### 常见问题

1. **内存不足**：减小batch_size
   ```bash
   --config_dict "train_batch_size=256,eval_batch_size=256"
   ```

2. **GPU内存不足**（Qwen3生成时）：
   - 减小batch_size到4或2
   - 使用更小的max_length

3. **找不到embeddings文件**：
   - 检查文件路径是否正确
   - 确认embeddings生成步骤已完成

## 预期结果

根据论文指导，预期看到：
- Base embeddings相比基线提升显著（特别是在冷启动/长尾items上）
- LLM embeddings提供更稳定的语义增强
- Cross层带来额外提升
- Alignment进一步改善性能

典型的性能排序：
```
Baseline < +Base < +Base+Cross < +Base+Cross+Align
         < +LLM  < +LLM+Cross  < +LLM+Cross+Align
```

## 下一步

1. 分析结果，选择最佳配置
2. 在其他数据集（Yelp, MIND）上验证
3. 实现两阶段训练策略
4. 添加更多评测指标（Recall@100, Tail-Coverage, ILD）
