## Time and memory cost of context-aware recommendation models 

### Datasets information:

| Dataset | #Interaction | #Feature Field |  #Feature |
| ------- | -----------: | -------------: | --------: |
| ml-1m   |       739012 |              5 |       134 |
| Criteo  |    1,000,000 |             39 | 2,572,192 |
| Avazu   |    4,218,938 |             21 | 1,326,631 |

### Device information

```
OS:                   Linux
Python Version:       3.8.10
PyTorch Version:      1.8.1
cudatoolkit Version:  10.1
GPU:                  TITAN V（12GB）
Machine Specs:        14 CPU machine, 256GB RAM
```

### 1) ml-1m dataset:

#### Time and memory cost on ml-1m dataset:

| Method           | Training Time (sec/epoch) | Evaluation Time (sec/epoch) | Training GPU Memory (GB) | Evaluation GPU Memory (GB) |
| ---------------- | ------------------------- | --------------------------- | ------------------------ | -------------------------- |
| AFM              | 2.29                      | 0.44                        | 0.09                     | 0.86                       |
| AutoInt          | 3.15                      | 0.42                        | 0.09                     | 0.45                       |
| DCN              | 2.93                      | 0.39                        | 0.14                     | 0.74                       |
| DCN V2 (stacked) | 1.76                      | 0.29                        | 0.43                     | 0.43                       |
| DeepFM           | 2.12                      | 0.44                        | 0.05                     | 0.34                       |
| DIEN             | 191.24                    | 16.96                       | 3.48                     | 3.48                       |
| DIN              | 31.47                     | 9.62                        | 1.47                     | 1.47                       |
| DSSM             | 2.77                      | 0.35                        | 0.12                     | 0.59                       |
| FFM              | 4.10                      | 0.39                        | 0.07                     | 0.69                       |
| FM               | 1.75                      | 0.35                        | 0.03                     | 0.17                       |
| FNN              | 1.74                      | 0.39                        | 0.05                     | 0.41                       |
| FwFM             | 4.02                      | 0.58                        | 0.08                     | 0.54                       |
| LR               | 1.10                      | 0.32                        | 0.01                     | 0.03                       |
| NFM              | 2.56                      | 0.45                        | 0.04                     | 0.18                       |
| PNN(inner)       | 2.14                      | 0.38                        | 0.07                     | 0.49                       |
| PNN(outer)       | 2.40                      | 0.46                        | 0.21                     | 1.88                       |
| WideDeep         | 2.28                      | 0.44                        | 0.05                     | 0.34                       |
| xDeepFM          | 7.58                      | 0.49                        | 0.54                     | 4.72                       |

#### Config file of ml-1m dataset:

```
# dataset config
field_separator: "\t"
seq_separator: " "
USER_ID_FIELD: user_id
ITEM_ID_FIELD: item_id
RATING_FIELD: rating
NEG_PREFIX: neg_
LABEL_FIELD: label
val_interval:
  rating: "[0,3);(3,inf)"
threshold:
    rating: 4
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

# model
embedding_size: 16
```

Other parameters (including model parameters) are default value. 

### 2）Avazu dataset:

#### Time and memory cost on Avazu dataset:

| Method           | Training Time (sec/epoch) | Evaluation Time (sec/epoch) | Training GPU Memory (GB) | Evaluation GPU Memory (GB) |
| ---------------- | ------------------------- | --------------------------- | ------------------------ | -------------------------- |
| AFM              | 15.34                     | 0.62                        | 1.76                     | 1.76                       |
| AutoInt          | 18.69                     | 0.80                        | 1.62                     | 1.62                       |
| DCN              | 9.60                      | 0.31                        | 1.36                     | 1.36                       |
| DCN V2 (stacked) | 7.41                      | 0.26                        | 1.04                     | 1.04                       |
| DeepFM           | 7.35                      | 0.36                        | 1.24                     | 1.24                       |
| DIEN             |                           |                             |                          |                            |
| DIN              |                           |                             |                          |                            |
| DSSM             |                           |                             |                          |                            |
| FFM              |                           |                             |                          |                            |
| FM               | 7.22                      | 0.27                        | 1.23                     | 1.23                       |
| FNN              | 6.36                      | 0.25                        | 1.22                     | 1.22                       |
| FwFM             | 277.00                    | 3.18                        | 1.75                     | 1.75                       |
| LR               | 4.54                      | 0.25                        | 0.42                     | 0.42                       |
| NFM              | 8.05                      | 0.34                        | 1.24                     | 1.24                       |
| PNN(inner)       | 9.16                      | 0.42                        | 1.45                     | 1.45                       |
| PNN(outer)       | 22.78                     | 1.02                        | 3.54                     | 3.54                       |
| WideDeep         | 7.37                      | 0.34                        | 1.24                     | 1.24                       |
| xDeepFM          | 27.92                     | 0.64                        | 2.02                     | 2.02                       |

Note: Avazu dataset is not suitable for DIN model and DSSM model.

#### Config file of Avazu dataset:

```
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



# model
embedding_size: 16
```

Other parameters (including model parameters) are default value. 

### 3）Criteo dataset:

#### Time and memory cost on Criteo dataset:

| Method           | Training Time (sec/epoch) | Evaluation Time (sec/epoch) | Training GPU Memory (GB) | Evaluation GPU Memory (GB) |
| ---------------- | ------------------------- | --------------------------- | ------------------------ | -------------------------- |
| AFM              | 59.21                     | 2.81                        | 2.98                     | 2.98                       |
| AutoInt          | 50.73                     | 2.64                        | 2.14                     | 2.14                       |
| DCN              | 27.69                     | 1.38                        | 1.54                     | 1.54                       |
| DCN V2 (stacked) | 17.58                     | 0.68                        | 1.9                      | 1.9                        |
| DeepFM           | 21.01                     | 1.38                        | 1.78                     | 1.78                       |
| DIEN             |                           |                             |                          |                            |
| DIN              |                           |                             |                          |                            |
| DSSM             |                           |                             |                          |                            |
| FFM              |                           |                             |                          |                            |
| FM               | 19.32                     | 1.23                        | 1.78                     | 1.78                       |
| FNN              | 19.29                     | 1.20                        | 1.53                     | 1.53                       |
| FwFM             | 4069.58                   | 17.48                       | 3.88                     | 3.88                       |
| LR               | 16.71                     | 1.12                        | 0.54                     | 0.54                       |
| NFM              | 22.09                     | 1.28                        | 1.57                     | 1.57                       |
| PNN(inner)       | 33.82                     | 1.53                        | 2.08                     | 2.08                       |
| PNN(outer)       | 106.90                    | 5.50                        | 9.59                     | 9.59                       |
| WideDeep         | 22.53                     | 1.43                        | 1.78                     | 1.78                       |
| xDeepFM          | 80.28                     | 2.68                        | 4.26                     | 4.26                       |

Note: Criteo dataset is not suitable for DIN model and DSSM model.
#### Config file of Criteo dataset:

```
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
# model
embedding_size: 16
```

Other parameters (including model parameters) are default value. 







