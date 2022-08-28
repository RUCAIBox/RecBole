# MovieLens-Sequential

**数据集:** ml-1m_seq

**数据过滤:** 删除 rating 小于 3 的交互记录

**k-core 过滤:** 删除交互数小于 10 的用户或商品

**评测方式:** 时间顺序排列，leave one out 切分数据集，全排序

**评测指标:** Recall@10, NGCG@10, MRR@10, Hit@10, Precision@10

## 数据集信息

| Dataset | #Users | #Items | #Interactions | Sparsity |
| ------- | ------ | ------ | ------------- | -------- |
| ml-1m   | 6,040  | 3,124  | 834,449       | 95.57%   |

**配置文件 (ml-1m_seq.yaml):**

```yaml
# dataset config
field_separator: "\t"
seq_separator: " "
USER_ID_FIELD: user_id
ITEM_ID_FIELD: item_id
RATING_FIELD: rating
TIME_FIELD: timestamp
NEG_PREFIX: neg_
ITEM_LIST_LENGTH_FIELD: item_length
LIST_SUFFIX: _list
MAX_ITEM_LIST_LENGTH: 50
POSITION_FIELD: position_id
load_col:
    inter: [user_id, item_id, rating, timestamp]

# data filtering for interactions
val_interval:
    rating: "[3,inf)"    
unused_col: 
    inter: [rating]

user_inter_num_interval: "[10,inf)"
item_inter_num_interval: "[10,inf)"

# training and evaluation
epochs: 500
train_batch_size: 4096
eval_batch_size: 40960000
valid_metric: NDCG@10
eval_args:
    split: {'LS': 'valid_and_test'}
    mode: full
    order: TO

# disable negative sampling
train_neg_sample_args: ~
```

## Note

- 为了保证模型之间的公平性，我们对用户和商品的embedding维度做出限制，不同模型中此参数名称不同请自行调整。

  ```yaml
  embedding_size: 64 
  ```

- 对于 `FDSA`、`SASRecF` 和 `GRU4RecF` 这三个需要使用物品属性作为辅助数据的序列模型，我们选择物品的类型作为数据列，需额外设置 `selected_features: [genre]`，并在 `load_col` 中加入物品的加载列：

  ```yaml
  load_col:
      inter: [user_id, item_id, rating, timestamp]
      item: [item_id, genre]
  selected_features: [genre]
  ```

- 对于 `S3Rec` 模型，其需要使用物品列作为特征进行预训练，需将 `item_attribute` 设置为 `genre`，同样需要修改数据加载方式：

  ```yaml
  load_col:
      inter: [user_id, item_id, rating, timestamp]
      item: [item_id, genre]
  item_attribute: genre
  ```

- 一般的序列推荐模型使用交叉熵损失函数 `CE`，无需进行负采样；而对于 `TransRec` 和 `FPMC` 这两个模型使用成对的 `BPR` 损失函数，需要对其进行负采样，将参数 `train_neg_sample_args` 恢复为默认的配置，注释 `train_neg_sample_args: ~` 即可：

  ```yaml
  # train_neg_sample_args: ~
  ```
