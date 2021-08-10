## Time and memory cost of knowledge-based recommendation models 

### Datasets information:

| Dataset    | #User   | #Item   | #Interaction | Sparsity | #Entity  | #Relation | #Triple    |
| ---------- | ------: | ------: | -----------: | -------: | -------: | --------: | ---------: |
| ml-1m      |  6,040  |  3,629  |     836,478  |  0.9618  |  79,388  |       51  |   385,923  |
| ml-10m     | 69,864  | 10,599  |   8,242,124  |  0.9889  | 181,941  |       51  | 1,051,385  |
| LFM-1b2013 | 28,150  | 64,583  |   1,907,900  |  0.9990  | 181,112  |        7  |   281,900  |

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
| CKE       |                     4.88  |                       0.44  |           0.38  |
| KTUP      |                     3.76  |                       1.70  |           0.47  |
| RippleNet |                    35.85  |                       0.84  |           7.26  |
| KGAT      |                     6.68  |                       0.37  |           2.10  |
| KGNN-LS   |                     8.20  |                       1.14  |           0.57  |
| KGCN      |                     3.56  |                       1.14  |           0.56  |
| MKR       |                     4.36  |                       5.57  |           3.68  |
| CFKG      |                     1.60  |                       0.57  |           0.27  |

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
val_interval:
    rating: "[3,inf)"

# training and evaluation
epochs: 500
train_batch_size: 4096
eval_batch_size: 102400
valid_metric: MRR@10

# model
embedding_size: 64
```

Other parameters (including model parameters) are default value. 

### 2）ml-10m dataset:

#### Time and memory cost on ml-10m dataset:

| Method    | Training Time (sec/epoch) | Evaluation Time (sec/epoch) | GPU Memory (GB)    |
| --------- | ------------------------: | --------------------------: | -----------------: |
| CKE       |                    43.41  |                      10.72  |              0.70  |
| KTUP      |                    33.82  |                      38.87  |              0.66  |
| RippleNet |                   360.16  |                      23.35  |              7.38  |
| KGAT      |                         - |                           - | CUDA out of memory |
| KGNN-LS   |                    84.51  |                      47.31  |              0.73  |
| KGCN      |                    20.13  |                      53.33  |              0.74  |
| MKR       |                    31.74  |                     207.12  |              3.85  |
| CFKG      |                    16.33  |                      16.88  |              0.46  |

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
val_interval:
    rating: "[3,inf)"

# training and evaluation
epochs: 500
train_batch_size: 4096
eval_batch_size: 102400
valid_metric: MRR@10

# model
embedding_size: 64
```

Other parameters (including model parameters) are default value. 

### 3）LFM-1b dataset:

#### Time and memory cost on LFM-1b dataset:

| Method    | Training Time (sec/epoch) | Evaluation Time (sec/epoch) | GPU Memory (GB) |
| --------- | ------------------------: | --------------------------: | --------------: |
| CKE       |                     8.73  |                      41.84  |           0.69  |
| KTUP      |                     4.56  |                      87.67  |           0.48  |
| RippleNet |                    82.53  |                      69.37  |           7.32  |
| KGAT      |                    15.69  |                      40.75  |           4.19  |
| KGNN-LS   |                    15.65  |                     436.84  |           0.61  |
| KGCN      |                     8.04  |                     443.80  |           0.60  |
| MKR       |                     9.06  |                     456.11  |           2.87  |
| CFKG      |                     3.81  |                      50.48  |           0.45  |

#### Config file of LFM-1b  dataset:

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
val_interval:
  rating: "[10,inf)"
user_inter_num_interval: "[10,inf)"
item_inter_num_interval: "[10,inf)"

# training and evaluation
epochs: 500
train_batch_size: 4096
eval_batch_size: 102400
valid_metric: MRR@10

# model
embedding_size: 64
```

Other parameters (including model parameters) are default value. 

