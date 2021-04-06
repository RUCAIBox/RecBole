## Time and memory cost of knowledge-based recommendation models 

### Datasets information:

| Dataset | #User  | #Item   | #Interaction | Sparsity | #Entity   | #Relation | #Triple   |
| ------- | ------: | -------: | ------------: | --------: | ---------: | ---------: | ---------: |
| ml-1m   | 6,040  | 3,629   | 836,478      | 0.9618   | 79,388    | 51        | 385,923   |
| ml-10m  | 69,864 | 10,599  | 8,242,124    | 0.9889   | 181,941   | 51        | 1,051,385 |
| LFM-1b | 64,536 | 156,343 | 6,544,312    | 0.9994   | 1,751,586 | 10        | 3,054,516 |

### Device information

```
OS:                   Linux
Python Version:       3.8.3
PyTorch Version:      1.7.0
cudatoolkit Version:  10.1
GPU:                  TITAN RTX（24GB）
Machine Specs:        32 CPU machine, 64GB RAM
```

### 1) ml-1m dataset:

#### Time and memory cost on ml-1m dataset:

| Method    | Training Time (sec/epoch) | Evaluation Time (sec/epoch) | GPU Memory (GB) |
| --------- | -------------------------: | ---------------------------: | ---------------: |
| CKE       | 3.76                      | 8.73                        | 1.16            |
| KTUP      | 3.82                      | 17.68                       | 1.04            |
| RippleNet | 9.39                      | 13.13                       | 4.57            |
| KGAT      | 9.59                      | 8.63                        | 3.52            |
| KGNN-LS   | 4.78                      | 15.09                       | 1.04            |
| KGCN      | 2.25                      | 13.71                       | 1.04            |
| MKR       | 6.25                      | 14.89                       | 1.29            |
| CFKG      | 1.49                      | 9.76                        | 0.97            |

#### Config file of ml-1m dataset:

```
# dataset config
field_separator: "\t"
seq_separator: " "
USER_ID_FIELD: user_id
ITEM_ID_FIELD: item_id
RATING_FIELD: rating
HEAD_ENTITY_ID_FIELD: head_id
TAIL_ENTITY_ID_FIELD: tail_id
RELATION_ID_FIELD: relation_id
ENTITY_ID_FIELD: entity_id
NEG_PREFIX: neg_
LABEL_FIELD: label
load_col:
    inter: [user_id, item_id, rating]
    kg: [head_id, relation_id, tail_id]
    link: [item_id, entity_id]
lowest_val:
    rating: 3
unused_col:[rating]

# training and evaluation
epochs: 500
train_batch_size: 2048
eval_batch_size: 2048
valid_metric: MRR@10
```

Other parameters (including model parameters) are default value. 

### 2）ml-10m dataset:

#### Time and memory cost on ml-10m dataset:

| Method    | Training Time (sec/epoch) | Evaluation Time (sec/epoch) | GPU Memory (GB) |
| --------- | -------------------------: | ---------------------------: | ---------------: |
| CKE       | 8.65                      | 85.53                       | 1.46            |
| KTUP      | 40.71                     | 507.56                      | 1.43            |
| RippleNet | 32.01                     | 152.40                      | 4.71            |
| KGAT      | 298.22                    | 80.94                       | 22.44           |
| KGNN-LS   | 15.47                     | 241.57                      | 1.42            |
| KGCN      | 7.73                      | 244.93                      | 1.42            |
| MKR       | 61.05                     | 383.29                      | 1.80            |
| CFKG      | 5.99                      | 140.74                      | 1.35            |

#### Config file of ml-10m dataset:

```
# dataset config
field_separator: "\t"
seq_separator: " "
USER_ID_FIELD: user_id
ITEM_ID_FIELD: item_id
RATING_FIELD: rating
HEAD_ENTITY_ID_FIELD: head_id
TAIL_ENTITY_ID_FIELD: tail_id
RELATION_ID_FIELD: relation_id
ENTITY_ID_FIELD: entity_id
NEG_PREFIX: neg_
LABEL_FIELD: label
load_col:
    inter: [user_id, item_id, rating]
    kg: [head_id, relation_id, tail_id]
    link: [item_id, entity_id]
lowest_val:
    rating: 3
unused_col:[rating]

# training and evaluation
epochs: 500
train_batch_size: 2048
eval_batch_size: 2048
valid_metric: MRR@10
```

Other parameters (including model parameters) are default value. 

### 3）LFM-1b dataset:

#### Time and memory cost on LFM-1b dataset:

| Method    | Training Time (sec/epoch) | Evaluation Time (sec/epoch) | GPU Memory (GB) |
| --------- | -------------------------: | ---------------------------: | ---------------: |
| CKE       | 62.99                     | 82.93                       | 4.45            |
| KTUP      | 91.79                     | 3218.69                     | 4.36            |
| RippleNet | 126.26                    | 188.38                      | 6.49            |
| KGAT      | 626.07                    | 75.70                       | 23.28           |
| KGNN-LS   | 62.55                     | 1709.10                     | 4.73            |
| KGCN      | 52.54                     | 1763.03                     | 4.71            |
| MKR       | 290.01                    | 2341.91                     | 6.96            |
| CFKG      | 53.35                     | 553.58                      | 4.22            |

#### Config file of LFM-1b  dataset:

```
# dataset config
field_separator: "\t"
seq_separator: " "
USER_ID_FIELD: user_id
ITEM_ID_FIELD: tracks_id
RATING_FIELD: rating
HEAD_ENTITY_ID_FIELD: head_id
TAIL_ENTITY_ID_FIELD: tail_id
RELATION_ID_FIELD: relation_id
ENTITY_ID_FIELD: entity_id
NEG_PREFIX: neg_
LABEL_FIELD: label
load_col:
    inter: [user_id, tracks_id, timestamp]
    kg: [head_id, relation_id, tail_id]
    link: [tracks_id, entity_id]
lowest_val:
    timestamp: 1356969600
  
highest_val:
    timestamp: 1362067200
unused_col: [timestamp]
min_user_inter_num: 2
min_item_inter_num: 15

# training and evaluation
epochs: 500
train_batch_size: 2048
eval_batch_size: 2048
valid_metric: MRR@10
```

Other parameters (including model parameters) are default value. 

