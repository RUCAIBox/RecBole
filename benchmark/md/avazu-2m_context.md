# Avazu-Context

**Dataset:** avazu-2m_context

**Normalization:** we normalize the scalar value of the numerical fields `timestamp` and `banner_pos`.

**Evaluation metric:** AUC, Log Loss

## Dataset Information

| Dataset | #Fields | #Features | #Instance | 
| ------- | ------ | ------ | ------------- | 
| avazu-2m   | 23  | 2,736,107  | 2,000,000  | 

**Configuration file (avazu-2m_context.yaml):**

```yaml
# dataset config
field_separator: "\t"
seq_separator: " "
USER_ID_FIELD: ~
ITEM_ID_FIELD: ~
LABEL_FIELD: label
fill_nan: True
normalize_all: True
numerical_features: ['timestamp','banner_pos']
load_col:
    inter: '*'

save_dataloaders: True

# training and evaluation
epochs: 500
train_batch_size: 4096
eval_batch_size: 40960000

eval_args:
    group_by: ~
    split: {'RS':[0.8, 0.1, 0.1]}
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