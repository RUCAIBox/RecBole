# Yelp2022-General

**Dataset:** yelp2022_general

**Data filtering:** delete interactive records with rating less than 3

**K-core filtering:** delete inactive users or unpopular items with less than 10 interactions

**Evaluation method:** random ordering, ratio-based data splitting and full sorting

**Evaluation metric:** Recall@10, MRR@10, NGCG@10, Hit@10, Precision@10

## Dataset Information

| Dataset  | #Users | #Items | #Interactions | Sparsity |
| -------- | ------ | ------ | ------------- | -------- |
| yelp2022 | 72,488 | 43,749 | 2,043,402     | 99.9356% |

**Configuration file (ml-1m_seq.yaml):**

```yaml
# dataset config
field_separator: "\t"
seq_separator: " "
USER_ID_FIELD: user_id
ITEM_ID_FIELD: item_id
RATING_FIELD: rating
NEG_PREFIX: neg_
LABEL_FIELD: label
load_col:
    inter: [user_id, item_id, rating]

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
train_neg_sample_args: 
    distribution: uniform
    sample_num: 1
    dynamic: False

# model
embedding_size: 64
```

## Note

- In order to ensure fairness between models, we limit the embedding dimension of users and items to `64`. Please adjust the parameter name in different models.

  ```yaml
  embedding_size: 64 
  ```