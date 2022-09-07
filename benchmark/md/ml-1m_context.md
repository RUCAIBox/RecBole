# MovieLens-Context

**Dataset:** ml-1m_context

**Data filtering:** delete interactive records with rating of 3

**Threshold:** the ratings for 1s and 2s are normalized to be 0s; 4s and 5s to be 1s;

**Evaluation metric:** AUC, Log Loss

## Dataset Information

| Dataset | #Fields | #Features | #Instance | 
| ------- | ------ | ------ | ------------- | 
| ml-1m   | 7  | 10040  | 739,012       | 

**Configuration file (ml-1m_context.yaml):**

```yaml
# dataset config
field_separator: "\t"
seq_separator: " "
USER_ID_FIELD: user_id
ITEM_ID_FIELD: item_id
RATING_FIELD: rating
NEG_PREFIX: neg_
LABEL_FIELD: label
threshold:
  rating: 4
val_interval:
  rating: "[0,3);(3,inf)"
load_col:
    inter: [user_id, item_id, rating]
    user: [user_id, age, gender, occupation]
    item: [item_id, genre, release_year]

# training and evaluation
epochs: 500
train_batch_size: 4096
eval_batch_size: 40960000

eval_args:
  split: {'RS':[0.8, 0.1, 0.1]}
  group_by: ~
  mode: labeled
  order: RO
valid_metric: AUC
metrics: ['AUC', 'LogLoss']
```

## Note

- In order to ensure fairness between models, we limit the embedding dimension of users and items to `16`. Please adjust the parameter name in different models.

  ```yaml
  embedding_size: 16
  ```