# Criteo-Context

**Dataset:** criteo-4m_context

**Discretization:** we lognormalize the scalar value of the numerical fields `I1` ~ `I13`.

**Evaluation metric:** AUC, Log Loss

## Dataset Information

| Dataset | #Fields | #Features | #Instance | 
| ------- | ------ | ------ | ------------- | 
| criteo-4m   | 39  | 4,156,307  | 4,000,000  | 

**Configuration file (criteo-4m_context.yaml):**

```yaml
# dataset config
field_separator: "\t"
seq_separator: " "
USER_ID_FIELD: ~
ITEM_ID_FIELD: ~
LABEL_FIELD: label
fill_nan: True
numerical_features: ['I1','I2','I3','I4','I5','I6','I7','I8','I9','I10','I11','I12','I13']
discretization:
  I1:
    method: 'LD'
  I2:
    method: 'LD'
  I3:
    method: 'LD'
  I4:
    method: 'LD'
  I5:
    method: 'LD'
  I6:
    method: 'LD'
  I7:
    method: 'LD'
  I8:
    method: 'LD'
  I9:
    method: 'LD'
  I10:
    method: 'LD'
  I11:
    method: 'LD'
  I12:
    method: 'LD'
  I13:
    method: 'LD'
load_col: 
    inter: '*'

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