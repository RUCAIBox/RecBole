# Yelp(2022) Sequential

**Dataset:** yelp_seq

**Data filtering:** delete interactive records with rating less than 3

**K-core filtering:** delete inactive users or unpopular items with less than 10 interactions

**Evaluation method:** chronological arrangement, leave one out split data set and full sorting

**Evaluation metric:** Recall@10, NGCG@10, MRR@10, Hit@10, Precision@10

## Dataset Information

| Dataset             | #Users | #Items | #Interactions | Sparsity |
| ------------------- | ------ | ------ | ------------- | -------- |
| Yelp(2022) | 72,488 | 43,749 | 2,043,402     | 99.94%   |

**Configuration file (yelp_seq.yaml):**

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
eval_batch_size: 4096
valid_metric: NDCG@10
eval_args:
    split: {'LS': 'valid_and_test'}
    mode: full
    order: TO

# disable negative sampling
train_neg_sample_args: ~
```

## Note

- In order to ensure fairness between models, we limit the embedding dimension of users and items to `64`. Please adjust the parameter name in different models.

  ```yaml
  embedding_size: 64 
  ```

- For the three sequential models that need to use item attributes as auxiliary data, namely `FDSA`, `SASRecF` and `GRU4RecF`, we select the item type `sales_type` as the data column, set `selected_features: [sales_type]` and load item columns as follows:

  ```yaml
  load_col:
      inter: [user_id, item_id, rating, timestamp]
      item: [item_id, sales_type]
  selected_features: [sales_type]
  ```

- For the `S3Rec` model, it needs to use the item column as a feature for pre-training and the `item_attribute` is set to `sales_type`. It is also necessary to load item columns as follows:

  ```yaml
  load_col:
      inter: [user_id, item_id, rating, timestamp]
      item: [item_id, sales_type]
  item_attribute: sales_type
  ```

- Most sequential recommendation models use the cross entropy loss function `CE` without negative sampling. For `TransRec` and `FPMC` models, the pairwise `BPR` loss function is employed, which needs negative sampling and the parameter `train_neg_sample_args` should restore to the default configuration as follows:

  ```yaml
  # train_neg_sample_args: ~
  ```
