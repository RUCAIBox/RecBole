## Time and memory cost of context-aware recommendation models 

### Datasets information:

| Dataset | #Interaction | #Feature Field | #Feature   |
| ------- | -----------: | -------------: | ---------: |
| ml-1m   |   1,000,209  |             5  |       134  |
| Criteo  |   1,000,000  |            39  | 2,572,192  |
| Avazu   |   4,218,938  |            21  | 1,326,631  |

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

| Method    | Training Time (sec/epoch) | Evaluation Time (sec/epoch) | GPU Memory (GB) |
| --------- | ------------------------: | --------------------------: | --------------: |
| LR        |                     1.02  |                       1.38  |           0.03  |
| DIN       |                    24.26  |                       0.87  |           4.61  |
| DSSM      |                     5.69  |                       1.17  |           0.19  |
| FM        |                     1.08  |                       1.34  |           0.03  |
| DeepFM    |                     2.08  |                       1.50  |           0.06  |
| Wide&Deep |                     2.12  |                       1.25  |           0.03  |
| NFM       |                     3.79  |                       1.12  |           0.05  |
| AFM       |                     1.77  |                       1.36  |           0.15  |
| AutoInt   |                     3.84  |                       1.44  |           0.17  |
| DCN       |                     4.98  |                       1.12  |           0.16  |
| FNN(DNN)  |                     1.95  |                       1.32  |           0.10  |
| PNN       |                     2.45  |                       1.50  |           0.13  |
| FFM       |                     2.39  |                       1.17  |           0.13  |
| FwFM      |                     2.25  |                       1.22  |           0.10  |
| xDeepFM   |                     7.20  |                       1.17  |           0.87  |

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
threshold:
    rating: 4
load_col:
    inter: [user_id, item_id, rating]
    user: [user_id, age, gender, occupation]
    item: [item_id, genre]

# training and evaluation
epochs: 500
train_batch_size: 4096
eval_batch_size: 25600
eval_args:
  split: {'RS':[0.8, 0.1, 0.1]}
  group_by: ~
  mode: labeled
  order: RO
valid_metric: AUC
metrics: ['AUC', 'LogLoss']

# model
embedding_size: 10
```

Other parameters (including model parameters) are default value. 

### 2）Criteo dataset:

#### Time and memory cost on Criteo dataset:

| Method    | Training Time (sec/epoch) | Evaluation Time (sec/epoch) | GPU Memory (GB) |
| --------- | ------------------------: | --------------------------: | --------------: |
| LR        |                     1.16  |                       0.10  |           0.10  |
| DIN       |                         - |                           - |               - |
| DSSM      |                         - |                           - |               - |
| FM        |                     1.67  |                       0.13  |           0.34  |
| DeepFM    |                     3.55  |                       0.13  |           0.34  |
| Wide&Deep |                     3.41  |                       0.13  |           0.34  |
| NFM       |                     3.58  |                       0.14  |           0.35  |
| AFM       |                     5.69  |                       0.27  |           2.13  |
| AutoInt   |                     5.42  |                       0.22  |           1.14  |
| DCN       |                     4.20  |                       0.15  |           0.42  |
| FNN(DNN)  |                     2.16  |                       0.11  |           0.36  |
| PNN       |                     3.32  |                       0.14  |           0.77  |
| FFM       |                    57.66  |                       0.71  |           8.60  |
| FwFM      |                   482.04  |                       3.21  |           1.59  |
| xDeepFM   |                    10.55  |                       0.34  |           1.91  |

Note: Criteo dataset is not suitable for DIN model and DSSM model.
#### Config file of Criteo dataset:

```
# dataset config
field_separator: "\t"
seq_separator: " "
USER_ID_FIELD: ~
ITEM_ID_FIELD: ~
LABEL_FIELD: label

load_col: 
    inter: '*'

fill_nan: True
normalize_all: True

# training and evaluation
epochs: 500
train_batch_size: 4096
eval_batch_size: 4096
eval_args:
  mode: labeled
  group_by: ~
valid_metric: AUC
metrics: ['AUC', 'LogLoss']

# model
embedding_size: 10
```

Other parameters (including model parameters) are default value. 

### 3）Avazu dataset:

#### Time and memory cost on Avazu dataset:

| Method    | Training Time (sec/epoch) | Evaluation Time (sec/epoch) | GPU Memory (GB)      |
| --------- | ------------------------: | --------------------------: | -------------------: |
| LR        |                     4.01  |                       0.37  |                0.36  |
| DIN       |                         - |                           - |                    - |
| DSSM      |                         - |                           - |                    - |
| FM        |                    13.14  |                       0.40  |                1.35  |
| DeepFM    |                    14.69  |                       0.48  |                1.38  |
| Wide&Deep |                    14.20  |                       0.41  |                1.36  |
| NFM       |                    17.40  |                       0.48  |                1.36  |
| AFM       |                    18.25  |                       0.55  |                1.89  |
| AutoInt   |                    21.42  |                       0.68  |                1.67  |
| DCN       |                    18.95  |                       0.44  |                1.37  |
| FNN(DNN)  |                    12.13  |                       0.40  |                1.31  |
| PNN       |                    14.19  |                       0.41  |                1.45  |
| FFM       |                         - |                           - |   CUDA out of memory |
| FwFM      |                   292.43  |                       3.83  |                1.74  |
| xDeepFM   |                    35.60  |                       0.93  |                2.20  |

Note: Avazu dataset is not suitable for DIN model and DSSM model.
#### Config file of Avazu dataset:

```
# dataset config
field_separator: "\t"
seq_separator: " "
LABEL_FIELD: label

load_col: 
    inter: '*'

fill_nan: True
USER_ID_FIELD: ~
ITEM_ID_FIELD: ~
normalize_all: True
val_interval: 
  timestamp: "[14102931, inf)"

# training and evaluation
epochs: 500
train_batch_size: 4096
eval_batch_size: 4096
eval_args:
  group_by: ~
  split: {'RS':[0.8, 0.1, 0.1]}
  mode: labeled
  order: RO
group_by_user: False
valid_metric: AUC
metrics: ['AUC', 'LogLoss']

# model
embedding_size: 10    
attention_size: 30
dropout_prob: 0.1
learning_rate: 5e-5
reg_weight: 5
```

Other parameters (including model parameters) are default value. 







