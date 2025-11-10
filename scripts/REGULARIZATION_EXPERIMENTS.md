# 正则化实验说明文档

## 背景
在SASRecAlign模型中，文本特征分支可能在采样评测上出现过拟合现象。为了缓解这个问题，我们设计了一系列正则化实验。

## 实验设计

### 实验8：门控L2正则化 (exp8_sasrec_gate_l2_reg)
- **配置文件**: `sasrec_align_qwen3_gate_l2.yaml`
- **关键参数**:
  - `text_gate_init: 0.3` - 较小的初始门控值
  - `text_gate_reg_l2: 0.01` - L2正则化系数
- **作用机制**: 通过L2正则惩罚过大的门控值，避免文本通道权重过高

### 实验9：门控熵正则化 (exp9_sasrec_gate_entropy_reg)
- **配置文件**: `sasrec_align_qwen3_gate_entropy.yaml`
- **关键参数**:
  - `text_gate_init: 0.5` - 中间初始值
  - `text_gate_reg_entropy: 0.1` - 熵正则化系数
- **作用机制**: 鼓励门控值接近0.5，避免极端值（接近0或1）

### 实验10：交叉网络Dropout (exp10_sasrec_cross_dropout)
- **配置文件**: `sasrec_align_qwen3_cross_dropout.yaml`
- **关键参数**:
  - `cross_dropout_prob: 0.3` - 30% dropout率
- **作用机制**: 对DCN-V2交叉网络的输出应用dropout，增强模型鲁棒性

### 实验11：组合正则化 (exp11_sasrec_combined_reg)
- **配置文件**: `sasrec_align_qwen3_combined_reg.yaml`
- **关键参数**:
  - `text_gate_init: 0.3`
  - `text_gate_reg_l2: 0.005` - 温和的L2正则
  - `cross_dropout_prob: 0.2` - 温和的dropout
- **作用机制**: 组合多种正则化技术，但使用较小的系数避免过度约束

## 正则化技术详解

### 1. 门控机制 (Learnable Gate)
```python
alpha = sigmoid(text_gate_param)  # α ∈ [0,1]
text_features = alpha * text_features
```
- 学习一个全局权重来控制文本特征的贡献度

### 2. L2正则化
```python
gate_l2_reg = text_gate_reg_l2 * (sigmoid(text_gate_param) ** 2)
```
- 惩罚过大的门控值，倾向于较小的文本权重

### 3. 熵正则化
```python
alpha = sigmoid(text_gate_param)
gate_entropy = -alpha * log(alpha) - (1-alpha) * log(1-alpha)
gate_entropy_reg = -text_gate_reg_entropy * gate_entropy
```
- 最大化门控值的熵，鼓励α接近0.5

### 4. 交叉网络Dropout
```python
cross_out = self.text_cross(raw)
if self.text_cross_dropout is not None:
    cross_out = self.text_cross_dropout(cross_out)
```
- 在训练时随机丢弃部分交叉特征，防止过度依赖特定交互模式

## 运行方式

1. **单独运行某个实验**：
```bash
bash scripts/exp8_sasrec_gate_l2_reg.sh
```

2. **批量运行所有正则化实验**：
```bash
bash scripts/run_regularization_experiments.sh
```

## 评估指标
- 主要关注valid集和test集性能差异
- 如果valid性能明显高于test，说明存在过拟合
- 理想情况下，正则化后valid和test性能应更接近

## 参数调优建议
1. L2正则系数：通常在[0.001, 0.1]范围内调整
2. 熵正则系数：通常在[0.01, 0.5]范围内调整
3. Dropout率：通常在[0.1, 0.5]范围内调整
4. 初始门控值：建议从较小值(0.1-0.3)开始
